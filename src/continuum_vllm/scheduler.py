"""Custom vLLM schedulers that implement Continuum TTL pinning."""

import time
from collections.abc import Callable, Sequence
from typing import Any

from vllm.logger import init_logger
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.request_queue import RequestQueue
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus

from continuum_vllm.compat import assert_compatible_vllm, assert_scheduler_contract
from continuum_vllm.policy import (
    ContinuumRequestMetadata,
    PaperTTLPolicy,
    PinManager,
    PinnedRequest,
    TTLDecision,
    TTLPolicy,
    _optional_bool,
    paper_ttl_policy_from_env,
)
from continuum_vllm.request_queue import ContinuumRequestQueue
from continuum_vllm.telemetry import ContinuumTelemetry, telemetry_dump_path_from_env

logger = init_logger("vllm.continuum")
OutputDecoder = Callable[[Sequence[int]], str]
AllocateSlots = Callable[..., Any]
GetNumNewMatchedTokens = Callable[..., Any]

# Connectors that always load synchronously inside the model forward pass.
# The scheduler never sees an asynchronous load window for these, so reload
# latency cannot be measured from here.
SYNC_ONLY_CONNECTORS = frozenset({"SimpleCPUOffloadConnector"})

# TTL decisions to observe with a tier attached before warning that the
# connector produced no reload samples at all.
SILENT_RELOAD_THRESHOLD = 64

# Upper bound on in-flight reload timers, so a connector that starts loads it
# never finishes cannot grow the dict without limit.
MAX_PENDING_RELOADS = 8192


class ContinuumSchedulerMixin:
    """Add paper TTL pinning without replacing files in vLLM."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        assert_compatible_vllm()
        assert_scheduler_contract()
        self._continuum_clock: Callable[[], float] = kwargs.pop(
            "continuum_clock", time.monotonic
        )
        ttl_policy = kwargs.pop("continuum_ttl_policy", None)
        self._continuum_ttl_policy: TTLPolicy = (
            ttl_policy if ttl_policy is not None else paper_ttl_policy_from_env()
        )
        output_decoder = kwargs.pop("continuum_output_decoder", None)
        telemetry_environ = kwargs.pop("continuum_environ", None)
        super().__init__(*args, **kwargs)

        self._continuum_pins = PinManager()
        self._continuum_finish_decisions: dict[str, TTLDecision | None] = {}
        self._continuum_job_arrival: dict[str, float] = {}
        self._continuum_program_turns: dict[str, int] = {}
        self._continuum_request_arrival: dict[str, float] = {}
        self._continuum_request_jobs: dict[str, str] = {}
        self._continuum_evicted_jobs: set[str] = set()
        self._continuum_evicted_waiting: dict[str, float] = {}
        self._continuum_reload_pending: dict[str, tuple[float, int]] = {}
        self._continuum_output_decoder = output_decoder or self._make_output_decoder()

        self._continuum_reload_mode, self._continuum_reload_detail = (
            self._detect_reload_mode()
        )
        self._continuum_telemetry = self._make_telemetry(telemetry_environ)
        # Retained under the historical name; the fields are identical.
        self._continuum_pin_stats = self._continuum_telemetry.counters

        self._replace_waiting_queues()
        self._install_allocate_slots_wrapper()
        self._install_connector_wrapper()
        logger.info(
            "Continuum scheduler active: class=%s ttl_policy=%s "
            "allocate_slots_wrapper=enabled reload_estimator=%s (%s)",
            type(self).__name__,
            type(self._continuum_ttl_policy).__name__,
            self._continuum_reload_mode,
            self._continuum_reload_detail,
        )

    # ---- installation ----------------------------------------------

    def _replace_waiting_queues(self) -> None:
        self.waiting = ContinuumRequestQueue(self.waiting)
        self.skipped_waiting = ContinuumRequestQueue(self.skipped_waiting)
        self.waiting.set_ranker(self._continuum_rank)
        self.skipped_waiting.set_ranker(self._continuum_rank)

    def _install_allocate_slots_wrapper(self) -> None:
        self._continuum_original_allocate_slots: AllocateSlots = (
            self.kv_cache_manager.allocate_slots
        )
        self.kv_cache_manager.allocate_slots = self._continuum_allocate_slots

    def _install_connector_wrapper(self) -> None:
        """Time asynchronous KV loads by wrapping the connector's lookup."""
        self._continuum_original_get_matched: GetNumNewMatchedTokens | None = None
        connector = getattr(self, "connector", None)
        if connector is None or self._continuum_reload_mode != "online":
            return
        self._continuum_original_get_matched = connector.get_num_new_matched_tokens
        connector.get_num_new_matched_tokens = (
            self._continuum_get_num_new_matched_tokens
        )

    def _detect_reload_mode(self) -> tuple[str, str]:
        """Decide whether reload latency is measurable from the scheduler."""
        connector = getattr(self, "connector", None)
        if connector is None:
            return (
                "disabled",
                "no kv_transfer_config; CacheMissCost uses the prefill profile",
            )
        name = type(connector).__name__
        transfer_config = getattr(self.vllm_config, "kv_transfer_config", None)
        extra = getattr(transfer_config, "kv_connector_extra_config", None) or {}
        if _optional_bool(extra.get("load_async")) is False:
            return "prefill_fallback", f"{name} configured with load_async=false"
        if name in SYNC_ONLY_CONNECTORS:
            return "prefill_fallback", f"{name} always loads synchronously"
        return "online", name

    def _make_telemetry(self, environ: Any = None) -> ContinuumTelemetry:
        from os import environ as process_environ

        values = process_environ if environ is None else environ
        try:
            interval = float(values.get("CONTINUUM_STATS_INTERVAL_SECONDS", "60"))
        except ValueError:
            logger.warning(
                "CONTINUUM_STATS_INTERVAL_SECONDS is not a number; using 60."
            )
            interval = 60.0
        return ContinuumTelemetry(
            policy=self._continuum_ttl_policy,
            reload_mode=self._continuum_reload_mode,
            reload_detail=self._continuum_reload_detail,
            log_interval_seconds=interval,
            dump_path=telemetry_dump_path_from_env(values),
            clock=self._continuum_clock,
        )

    def _make_output_decoder(self) -> OutputDecoder | None:
        try:
            tokenizer = cached_tokenizer_from_config(self.vllm_config.model_config)
        except Exception:
            logger.warning(
                "Continuum could not initialize the tokenizer; requests must "
                "provide this_func_call to enable pinning.",
                exc_info=True,
            )
            return None
        if tokenizer is None:
            return None
        return lambda token_ids: tokenizer.decode(token_ids, skip_special_tokens=True)

    # ---- scheduling ------------------------------------------------

    def add_request(self, request: Request) -> None:
        self._expire_continuum_pins()
        metadata = ContinuumRequestMetadata.from_request(request)
        is_new_request = request.request_id not in self.requests
        if metadata.enabled and is_new_request:
            assert metadata.job_id is not None
            now = self._continuum_clock()
            self._continuum_job_arrival.setdefault(
                metadata.job_id, request.arrival_time
            )
            self._continuum_program_turns[metadata.job_id] = (
                self._continuum_program_turns.get(metadata.job_id, 0) + 1
            )
            self._continuum_request_arrival[request.request_id] = now
            self._continuum_request_jobs[request.request_id] = metadata.job_id
            self._continuum_ttl_policy.on_request_arrival(metadata, now)
            if metadata.job_id in self._continuum_evicted_jobs:
                self._continuum_evicted_waiting[request.request_id] = now
                self._continuum_evicted_jobs.discard(metadata.job_id)
            self._continuum_pins.claim(metadata.job_id, request.request_id)
        super().add_request(request)

    def _continuum_rank(self, request: Request) -> tuple[int, int, float, float, str]:
        metadata = ContinuumRequestMetadata.from_request(request)
        job_id = metadata.job_id
        is_preempted = request.status == RequestStatus.PREEMPTED
        is_pinned = job_id is not None and job_id in self._continuum_pins
        job_arrival = (
            self._continuum_job_arrival.get(job_id, request.arrival_time)
            if job_id is not None
            else request.arrival_time
        )
        return (
            0 if is_preempted else 1,
            0 if is_pinned and not is_preempted else 1,
            job_arrival,
            request.arrival_time,
            request.request_id,
        )

    def _select_waiting_queue_for_scheduling(self) -> RequestQueue | None:
        if self.waiting and self.skipped_waiting:
            waiting_request = self.waiting.peek_request()
            skipped_request = self.skipped_waiting.peek_request()
            if self._continuum_rank(waiting_request) <= self._continuum_rank(
                skipped_request
            ):
                return self.waiting
            return self.skipped_waiting
        return self.waiting or self.skipped_waiting or None

    def schedule(self, *args: Any, **kwargs: Any):
        self._expire_continuum_pins()
        self._continuum_report()
        return super().schedule(*args, **kwargs)

    def _continuum_report(self) -> None:
        if self._continuum_telemetry.silent_reload_detected(SILENT_RELOAD_THRESHOLD):
            logger.warning(
                "Continuum made %d TTL decisions with KV tier %s attached but "
                "observed no asynchronous reload samples. The connector is "
                "loading synchronously, so CacheMissCost keeps using the "
                "prefill profile and TTLs will be longer than the tier "
                "warrants. Measure the tier offline and point "
                "CONTINUUM_PREFILL_PROFILE at the result, or accept the "
                "conservative estimate.",
                SILENT_RELOAD_THRESHOLD,
                self._continuum_reload_detail,
            )
        if self._continuum_telemetry.due_for_log():
            logger.info("%s", self._continuum_telemetry.log_line())
            # Refresh the dump on the same cadence so the file is readable
            # without stopping the engine.
            try:
                self._continuum_telemetry.dump()
            except OSError:
                logger.warning(
                    "Continuum could not refresh the stats dump.", exc_info=True
                )

    # ---- pin lifecycle ---------------------------------------------

    @property
    def _continuum_tier_available(self) -> bool:
        """True when an external KV tier would hold this prefix."""
        return getattr(self, "connector", None) is not None

    def _free_request(
        self, request: Request, delay_free_blocks: bool = False
    ) -> dict[str, Any] | None:
        metadata = ContinuumRequestMetadata.from_request(request)
        if metadata.enabled:
            assert metadata.job_id is not None
            self._continuum_pins.unclaim(metadata.job_id, request.request_id)
            successful = request.status in (
                RequestStatus.FINISHED_STOPPED,
                RequestStatus.FINISHED_LENGTH_CAPPED,
            )
            if successful:
                decision = self._continuum_ttl_policy.on_request_finished(
                    metadata,
                    self._decode_output(request),
                    self._continuum_clock(),
                    request.num_tokens,
                    self._continuum_tier_available,
                )
                self._continuum_finish_decisions[request.request_id] = decision
                self._continuum_telemetry.record_decision(decision)
                if metadata.is_last_step is True:
                    num_turns = self._continuum_program_turns.pop(metadata.job_id, 0)
                    self._continuum_ttl_policy.observe_program(num_turns)
            if metadata.is_last_step is True or not successful:
                previous = self._continuum_pins.pop(metadata.job_id)
                if previous is not None:
                    self._release_continuum_pin(previous, "final")
                self._forget_continuum_program(metadata.job_id)
            else:
                self._forget_continuum_request(request.request_id)
        self._continuum_reload_pending.pop(request.request_id, None)
        return super()._free_request(request, delay_free_blocks)

    def _decode_output(self, request: Request) -> str | None:
        if self._continuum_output_decoder is None or not request.output_token_ids:
            return None
        try:
            return self._continuum_output_decoder(request.output_token_ids)
        except Exception:
            logger.warning(
                "Continuum could not decode output for request %s.",
                request.request_id,
                exc_info=True,
            )
            return None

    def _free_blocks(self, request: Request) -> None:
        metadata = ContinuumRequestMetadata.from_request(request)
        decision = self._continuum_finish_decisions.pop(request.request_id, None)
        now = self._continuum_clock()
        if metadata.job_id is not None and decision and decision.is_active(now):
            previous = self._continuum_pins.pin(
                job_id=metadata.job_id,
                request=request,
                expires_at=decision.expires_at,
                program_arrival=self._continuum_job_arrival.get(
                    metadata.job_id, request.arrival_time
                ),
                tool_name=decision.tool_name,
            )
            if previous is not None:
                self._release_continuum_pin(previous, "replaced")
            del self.requests[request.request_id]
            self._continuum_telemetry.record_pin()
            logger.debug(
                "Continuum pinned job=%s tool=%s ttl=%.3f ttl_source=%s "
                "recon=%.4fs recon_source=%s tokens=%d",
                metadata.job_id,
                decision.tool_name,
                decision.ttl_seconds,
                decision.source,
                decision.reconstruction_seconds,
                decision.reconstruction_source,
                request.num_tokens,
            )
            return
        if metadata.job_id is not None and decision is not None:
            self._mark_continuum_evicted(metadata.job_id)
        super()._free_blocks(request)

    def _continuum_allocate_slots(
        self, request: Request, *args: Any, **kwargs: Any
    ) -> Any:
        blocks = self._continuum_original_allocate_slots(request, *args, **kwargs)
        while blocks is None:
            pin = self._continuum_pins.release_for_pressure()
            if pin is None:
                break
            self._mark_continuum_evicted(pin.job_id)
            self._release_continuum_pin(pin, "pressure")
            logger.debug(
                "Continuum released pinned KV for job=%s under pressure",
                pin.job_id,
            )
            blocks = self._continuum_original_allocate_slots(request, *args, **kwargs)
        if blocks is not None:
            self._on_request_allocated(request)
        return blocks

    def _expire_continuum_pins(self) -> None:
        for pin in self._continuum_pins.expire(self._continuum_clock()):
            self._mark_continuum_evicted(pin.job_id)
            self._release_continuum_pin(pin, "ttl")

    def _on_request_allocated(self, request: Request) -> None:
        metadata = ContinuumRequestMetadata.from_request(request)
        waiting_started = self._continuum_evicted_waiting.pop(request.request_id, None)
        if waiting_started is not None:
            self._continuum_ttl_policy.observe_queue_delay(
                max(0.0, self._continuum_clock() - waiting_started)
            )
        self._continuum_request_arrival.pop(request.request_id, None)
        if metadata.job_id is None:
            return
        self._continuum_evicted_jobs.discard(metadata.job_id)
        pin = self._continuum_pins.handoff(metadata.job_id)
        if pin is None:
            return
        self._release_continuum_pin(pin, "handoff")

    def _mark_continuum_evicted(self, job_id: str) -> None:
        self._continuum_evicted_jobs.add(job_id)
        for queue in (self.waiting, self.skipped_waiting):
            for request in queue:
                metadata = ContinuumRequestMetadata.from_request(request)
                if metadata.job_id == job_id:
                    waiting_started = self._continuum_request_arrival.get(
                        request.request_id, self._continuum_clock()
                    )
                    self._continuum_evicted_waiting.setdefault(
                        request.request_id, waiting_started
                    )

    def _forget_continuum_request(self, request_id: str) -> None:
        self._continuum_request_jobs.pop(request_id, None)
        self._continuum_request_arrival.pop(request_id, None)
        self._continuum_evicted_waiting.pop(request_id, None)

    def _forget_continuum_program(self, job_id: str) -> None:
        self._continuum_job_arrival.pop(job_id, None)
        self._continuum_program_turns.pop(job_id, None)
        self._continuum_evicted_jobs.discard(job_id)
        request_ids = [
            request_id
            for request_id, request_job_id in self._continuum_request_jobs.items()
            if request_job_id == job_id
        ]
        for request_id in request_ids:
            self._forget_continuum_request(request_id)

    def _release_continuum_pin(self, pin: PinnedRequest, reason: str) -> None:
        self._continuum_telemetry.record_release(reason)
        self._free_request_blocks(pin.request)

    def _release_all_continuum_pins(self) -> None:
        for pin in self._continuum_pins.pop_all():
            self._release_continuum_pin(pin, "final")

    # ---- reload measurement ----------------------------------------

    def _continuum_get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> Any:
        """Start a reload timer whenever the connector schedules an async load."""
        assert self._continuum_original_get_matched is not None
        result = self._continuum_original_get_matched(request, num_computed_tokens)
        try:
            num_external_tokens, load_async = result
        except (TypeError, ValueError):
            return result
        if load_async and num_external_tokens:
            pending = self._continuum_reload_pending
            if len(pending) >= MAX_PENDING_RELOADS:
                pending.pop(next(iter(pending)), None)
            pending[request.request_id] = (
                self._continuum_clock(),
                int(num_external_tokens),
            )
        return result

    def _update_from_kv_xfer_finished(self, kv_connector_output: Any) -> Any:
        """Close reload timers for requests whose transfer just completed."""
        if self._continuum_original_get_matched is None:
            return super()._update_from_kv_xfer_finished(kv_connector_output)
        before = set(self.finished_recving_kv_req_ids)
        result = super()._update_from_kv_xfer_finished(kv_connector_output)
        now = self._continuum_clock()
        for request_id in self.finished_recving_kv_req_ids - before:
            pending = self._continuum_reload_pending.pop(request_id, None)
            if pending is None:
                continue
            started_at, num_tokens = pending
            self._continuum_ttl_policy.observe_reload(
                num_tokens, max(0.0, now - started_at)
            )
            self._continuum_telemetry.record_reload_sample()
        return result

    # ---- teardown --------------------------------------------------

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        self._release_all_continuum_pins()
        return super().reset_prefix_cache(reset_running_requests, reset_connector)

    def shutdown(self) -> None:
        self._release_all_continuum_pins()
        self.kv_cache_manager.allocate_slots = self._continuum_original_allocate_slots
        connector = getattr(self, "connector", None)
        if connector is not None and self._continuum_original_get_matched is not None:
            connector.get_num_new_matched_tokens = self._continuum_original_get_matched
            self._continuum_original_get_matched = None
        logger.info("%s", self._continuum_telemetry.log_line())
        try:
            written = self._continuum_telemetry.dump()
        except OSError:
            logger.warning("Continuum could not write the stats dump.", exc_info=True)
        else:
            if written is not None:
                logger.info("Continuum stats written to %s", written)
        self._continuum_finish_decisions.clear()
        self._continuum_job_arrival.clear()
        self._continuum_program_turns.clear()
        self._continuum_request_arrival.clear()
        self._continuum_request_jobs.clear()
        self._continuum_evicted_jobs.clear()
        self._continuum_evicted_waiting.clear()
        self._continuum_reload_pending.clear()
        super().shutdown()


class ContinuumScheduler(ContinuumSchedulerMixin, Scheduler):
    """Synchronous vLLM 0.25.1 scheduler with Continuum enabled."""


class AsyncContinuumScheduler(ContinuumSchedulerMixin, AsyncScheduler):
    """Asynchronous vLLM 0.25.1 scheduler with Continuum enabled."""


__all__ = [
    "AsyncContinuumScheduler",
    "ContinuumScheduler",
    "ContinuumSchedulerMixin",
    "PaperTTLPolicy",
]
