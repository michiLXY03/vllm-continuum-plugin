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
    ContinuumPinStats,
    ContinuumRequestMetadata,
    PinManager,
    PinnedRequest,
    TTLDecision,
    TTLPolicy,
    paper_ttl_policy_from_env,
)
from continuum_vllm.request_queue import ContinuumRequestQueue

logger = init_logger(__name__)
OutputDecoder = Callable[[Sequence[int]], str]
AllocateSlots = Callable[..., Any]


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
        super().__init__(*args, **kwargs)

        self._continuum_pins = PinManager()
        self._continuum_pin_stats = ContinuumPinStats()
        self._continuum_finish_decisions: dict[str, TTLDecision | None] = {}
        self._continuum_job_arrival: dict[str, float] = {}
        self._continuum_program_turns: dict[str, int] = {}
        self._continuum_request_arrival: dict[str, float] = {}
        self._continuum_request_jobs: dict[str, str] = {}
        self._continuum_evicted_jobs: set[str] = set()
        self._continuum_evicted_waiting: dict[str, float] = {}
        self._continuum_output_decoder = output_decoder or self._make_output_decoder()
        self._replace_waiting_queues()
        self._install_allocate_slots_wrapper()

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
        return super().schedule(*args, **kwargs)

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
                self._continuum_finish_decisions[request.request_id] = (
                    self._continuum_ttl_policy.on_request_finished(
                        metadata,
                        self._decode_output(request),
                        self._continuum_clock(),
                        request.num_tokens,
                    )
                )
                if metadata.is_last_step is True:
                    num_turns = self._continuum_program_turns.pop(metadata.job_id, 0)
                    self._continuum_ttl_policy.observe_program(num_turns)
            if metadata.is_last_step is True or not successful:
                previous = self._continuum_pins.pop(metadata.job_id)
                if previous is not None:
                    self._release_continuum_pin(previous)
                self._forget_continuum_program(metadata.job_id)
            else:
                self._forget_continuum_request(request.request_id)
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
                self._release_continuum_pin(previous)
            del self.requests[request.request_id]
            logger.debug(
                "Continuum pinned job=%s tool=%s ttl=%.3f source=%s",
                metadata.job_id,
                decision.tool_name,
                decision.ttl_seconds,
                decision.source,
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
            self._continuum_pin_stats.pressure_unpins += 1
            self._mark_continuum_evicted(pin.job_id)
            self._release_continuum_pin(pin)
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
            self._continuum_pin_stats.ttl_expirations += 1
            self._mark_continuum_evicted(pin.job_id)
            self._release_continuum_pin(pin)

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
        self._continuum_pin_stats.handoffs += 1
        self._release_continuum_pin(pin)

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

    def _release_continuum_pin(self, pin: PinnedRequest) -> None:
        self._free_request_blocks(pin.request)

    def _release_all_continuum_pins(self) -> None:
        for pin in self._continuum_pins.pop_all():
            self._release_continuum_pin(pin)

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        self._release_all_continuum_pins()
        return super().reset_prefix_cache(reset_running_requests, reset_connector)

    def shutdown(self) -> None:
        self._release_all_continuum_pins()
        self.kv_cache_manager.allocate_slots = self._continuum_original_allocate_slots
        self._continuum_finish_decisions.clear()
        self._continuum_job_arrival.clear()
        self._continuum_program_turns.clear()
        self._continuum_request_arrival.clear()
        self._continuum_request_jobs.clear()
        self._continuum_evicted_jobs.clear()
        self._continuum_evicted_waiting.clear()
        super().shutdown()


class ContinuumScheduler(ContinuumSchedulerMixin, Scheduler):
    """Synchronous vLLM 0.25.1 scheduler with Continuum enabled."""


class AsyncContinuumScheduler(ContinuumSchedulerMixin, AsyncScheduler):
    """Asynchronous vLLM 0.25.1 scheduler with Continuum enabled."""
