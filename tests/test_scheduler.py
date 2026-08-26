import importlib
import sys
from collections import deque
from enum import IntEnum, auto
from types import ModuleType, SimpleNamespace
from typing import Any


class _RequestStatus(IntEnum):
    WAITING = auto()
    RUNNING = auto()
    PREEMPTED = auto()
    FINISHED_STOPPED = auto()
    FINISHED_LENGTH_CAPPED = auto()
    FINISHED_ABORTED = auto()


class _RequestQueue:
    pass


class _FCFSRequestQueue(deque, _RequestQueue):
    def add_request(self, request: Any) -> None:
        self.append(request)

    def pop_request(self) -> Any:
        return self.popleft()

    def peek_request(self) -> Any:
        return self[0]

    def prepend_request(self, request: Any) -> None:
        self.appendleft(request)

    def prepend_requests(self, requests: _RequestQueue) -> None:
        self.extendleft(requests)

    def remove_request(self, request: Any) -> None:
        self.remove(request)

    def remove_requests(self, requests: Any) -> None:
        for request in tuple(requests):
            if request in self:
                self.remove(request)


class _KVCacheManager:
    def __init__(self) -> None:
        self.freed: list[Any] = []
        self.allocation_results: deque[Any] = deque()
        self.allocation_calls = 0

    def allocate_slots(self, request: Any, *args: Any, **kwargs: Any) -> Any:
        self.allocation_calls += 1
        if self.allocation_results:
            return self.allocation_results.popleft()
        return object()

    def free(self, request: Any) -> None:
        self.freed.append(request)


class _Connector:
    """Minimal KVConnectorBase_V1 surface used by the reload estimator."""

    def __init__(self, results: Any = ()) -> None:
        self.results = deque(results)
        self.calls: list[tuple[str, int]] = []

    def get_num_new_matched_tokens(
        self, request: Any, num_computed_tokens: int
    ) -> tuple[int, bool]:
        self.calls.append((request.request_id, num_computed_tokens))
        if self.results:
            return self.results.popleft()
        return 0, False


class _Scheduler:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(skip_tokenizer_init=True),
            kv_transfer_config=kwargs.pop("kv_transfer_config", None),
        )
        self.kv_cache_manager = kwargs.pop("kv_cache_manager", _KVCacheManager())
        self.connector = kwargs.pop("connector", None)
        self.finished_recving_kv_req_ids: set[str] = set()
        self.requests: dict[str, Any] = {}
        self.waiting = _FCFSRequestQueue()
        self.skipped_waiting = _FCFSRequestQueue()
        self.shutdown_called = False

    def _update_from_kv_xfer_finished(self, kv_connector_output: Any) -> None:
        for request_id in kv_connector_output.finished_recving or ():
            self.finished_recving_kv_req_ids.add(request_id)

    def add_request(self, request: Any) -> None:
        self.requests[request.request_id] = request
        self.waiting.add_request(request)

    def schedule(self, *args: Any, **kwargs: Any) -> str:
        return "scheduled"

    def _free_request(
        self, request: Any, delay_free_blocks: bool = False
    ) -> dict[str, Any] | None:
        if request in self.waiting:
            self.waiting.remove_request(request)
        if not delay_free_blocks:
            self._free_blocks(request)
        return None

    def _free_blocks(self, request: Any) -> None:
        self._free_request_blocks(request)
        del self.requests[request.request_id]

    def _free_request_blocks(self, request: Any) -> None:
        self.kv_cache_manager.free(request)

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        return True

    def shutdown(self) -> None:
        self.shutdown_called = True


class _AsyncScheduler(_Scheduler):
    pass


class _Logger:
    def info(self, *args: Any, **kwargs: Any) -> None:
        pass

    def warning(self, *args: Any, **kwargs: Any) -> None:
        pass

    def debug(self, *args: Any, **kwargs: Any) -> None:
        pass


def _module(name: str, **attributes: Any) -> ModuleType:
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def _install_vllm_stubs() -> None:
    modules = {
        "vllm": _module("vllm", __path__=[]),
        "vllm.logger": _module("vllm.logger", init_logger=lambda name: _Logger()),
        "vllm.tokenizers": _module(
            "vllm.tokenizers", cached_tokenizer_from_config=lambda config: None
        ),
        "vllm.v1": _module("vllm.v1", __path__=[]),
        "vllm.v1.core": _module("vllm.v1.core", __path__=[]),
        "vllm.v1.core.kv_cache_manager": _module(
            "vllm.v1.core.kv_cache_manager", KVCacheManager=_KVCacheManager
        ),
        "vllm.v1.core.sched": _module("vllm.v1.core.sched", __path__=[]),
        "vllm.v1.core.sched.request_queue": _module(
            "vllm.v1.core.sched.request_queue",
            FCFSRequestQueue=_FCFSRequestQueue,
            RequestQueue=_RequestQueue,
        ),
        "vllm.v1.core.sched.scheduler": _module(
            "vllm.v1.core.sched.scheduler", Scheduler=_Scheduler
        ),
        "vllm.v1.core.sched.async_scheduler": _module(
            "vllm.v1.core.sched.async_scheduler", AsyncScheduler=_AsyncScheduler
        ),
        "vllm.v1.request": _module(
            "vllm.v1.request", Request=object, RequestStatus=_RequestStatus
        ),
    }
    sys.modules.update(modules)


_install_vllm_stubs()
sys.modules.pop("continuum_vllm.request_queue", None)
sys.modules.pop("continuum_vllm.scheduler", None)
scheduler_module = importlib.import_module("continuum_vllm.scheduler")
scheduler_module.assert_compatible_vllm = lambda: "0.25.1"
scheduler_module.assert_scheduler_contract = lambda: None

ContinuumScheduler = scheduler_module.ContinuumScheduler
ContinuumRequestQueue = importlib.import_module(
    "continuum_vllm.request_queue"
).ContinuumRequestQueue


class _Clock:
    def __init__(self, now: float) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _request(
    request_id: str,
    *,
    job_id: str = "job",
    is_last_step: int = 0,
    this_func_call: str | None = "pytest",
    arrival_time: float = 0.0,
    status: _RequestStatus = _RequestStatus.FINISHED_STOPPED,
    num_tokens: int = 100,
) -> Any:
    extra_args = {
        "job_id": job_id,
        "is_last_step": is_last_step,
        "this_func_call": this_func_call,
    }
    return SimpleNamespace(
        request_id=request_id,
        sampling_params=SimpleNamespace(extra_args=extra_args),
        output_token_ids=[],
        arrival_time=arrival_time,
        num_tokens=num_tokens,
        status=status,
    )


def test_scheduler_replaces_vllm_queues_without_patching_module() -> None:
    scheduler = ContinuumScheduler(continuum_clock=_Clock(10.0))

    assert isinstance(scheduler.waiting, ContinuumRequestQueue)
    assert isinstance(scheduler.skipped_waiting, ContinuumRequestQueue)


def test_scheduler_keeps_kv_until_ttl_expires() -> None:
    clock = _Clock(10.0)
    scheduler = ContinuumScheduler(continuum_clock=clock)
    request = _request("request")

    scheduler.add_request(request)
    scheduler._free_request(request)

    assert scheduler.kv_cache_manager.freed == []
    assert request.request_id not in scheduler.requests
    assert len(scheduler._continuum_pins) == 1

    clock.now = 12.0
    scheduler.schedule()

    assert scheduler.kv_cache_manager.freed == [request]
    assert len(scheduler._continuum_pins) == 0


def test_scheduler_releases_last_step_immediately() -> None:
    scheduler = ContinuumScheduler(continuum_clock=_Clock(10.0))
    request = _request("request", is_last_step=1)

    scheduler.add_request(request)
    scheduler._free_request(request)

    assert scheduler.kv_cache_manager.freed == [request]
    assert len(scheduler._continuum_pins) == 0


def test_scheduler_releases_claimed_pin_when_next_turn_fails() -> None:
    scheduler = ContinuumScheduler(continuum_clock=_Clock(10.0))
    previous = _request("previous")
    scheduler._continuum_pins.pin("job", previous, 12.0, 1.0, "pytest")
    failed = _request("failed", status=_RequestStatus.FINISHED_ABORTED)

    scheduler.add_request(failed)
    scheduler._free_request(failed)

    assert scheduler.kv_cache_manager.freed == [previous, failed]
    assert len(scheduler._continuum_pins) == 0


def test_queue_prioritizes_preempted_then_pinned_then_program_fcfs() -> None:
    scheduler = ContinuumScheduler(continuum_clock=_Clock(10.0))
    old = _request("old", job_id="old", arrival_time=1.0)
    pinned = _request("pinned", job_id="pinned", arrival_time=3.0)
    preempted = _request(
        "preempted",
        job_id="preempted",
        arrival_time=4.0,
        status=_RequestStatus.PREEMPTED,
    )
    for request in (old, pinned, preempted):
        scheduler.add_request(request)
    scheduler._continuum_pins.pin("pinned", object(), 12.0, 3.0, "pytest")

    assert scheduler.waiting.pop_request() is preempted
    assert scheduler.waiting.pop_request() is pinned
    assert scheduler.waiting.pop_request() is old


def test_allocate_wrapper_hands_off_pin_after_success() -> None:
    scheduler = ContinuumScheduler(continuum_clock=_Clock(10.0))
    old_request = _request("finished")
    scheduler._continuum_pins.pin("job", old_request, 12.0, 1.0, "pytest")
    next_request = _request("next", arrival_time=11.0)
    scheduler.add_request(next_request)
    allocated = object()
    scheduler.kv_cache_manager.allocation_results.append(allocated)

    result = scheduler.kv_cache_manager.allocate_slots(next_request, 1)

    assert result is allocated
    assert scheduler.kv_cache_manager.freed == [old_request]
    assert scheduler._continuum_pin_stats.handoffs == 1


def test_allocate_wrapper_releases_pin_and_retries_under_pressure() -> None:
    scheduler = ContinuumScheduler(continuum_clock=_Clock(10.0))
    pinned_request = _request("finished", job_id="pinned")
    scheduler._continuum_pins.pin("pinned", pinned_request, 12.0, 2.0, "pytest")
    waiting_request = _request("waiting", job_id="waiting")
    allocated = object()
    scheduler.kv_cache_manager.allocation_results.extend([None, allocated])

    result = scheduler.kv_cache_manager.allocate_slots(waiting_request, 1)

    assert result is allocated
    assert scheduler.kv_cache_manager.allocation_calls == 2
    assert scheduler.kv_cache_manager.freed == [pinned_request]
    assert scheduler._continuum_pin_stats.pressure_unpins == 1


def test_allocate_wrapper_releases_multiple_pins_until_allocation_fits() -> None:
    scheduler = ContinuumScheduler(continuum_clock=_Clock(10.0))
    older = _request("older", job_id="older")
    newer = _request("newer", job_id="newer")
    scheduler._continuum_pins.pin("older", older, 12.0, 1.0, "pytest")
    scheduler._continuum_pins.pin("newer", newer, 12.0, 2.0, "pytest")
    allocated = object()
    scheduler.kv_cache_manager.allocation_results.extend([None, None, allocated])

    result = scheduler.kv_cache_manager.allocate_slots(
        _request("waiting", job_id="waiting"), 1
    )

    assert result is allocated
    assert scheduler.kv_cache_manager.freed == [newer, older]
    assert scheduler._continuum_pin_stats.pressure_unpins == 2


def test_shutdown_releases_pins_and_restores_allocate_slots() -> None:
    manager = _KVCacheManager()
    original = manager.allocate_slots
    scheduler = ContinuumScheduler(
        kv_cache_manager=manager, continuum_clock=_Clock(10.0)
    )
    pinned_request = _request("finished")
    scheduler._continuum_pins.pin("job", pinned_request, 12.0, 1.0, "pytest")

    scheduler.shutdown()

    assert manager.freed == [pinned_request]
    assert manager.allocate_slots == original
    assert scheduler.shutdown_called


def test_installation_check_loads_both_scheduler_classes(capsys) -> None:
    check_module = importlib.import_module("continuum_vllm.check")
    check_module.assert_compatible_vllm = lambda: "0.25.1+empty"
    check_module.assert_scheduler_contract = lambda: None

    check_module.main()

    output = capsys.readouterr().out
    assert "vLLM: 0.25.1+empty" in output
    assert "AsyncContinuumScheduler" in output


def _xfer(*request_ids: str) -> Any:
    return SimpleNamespace(finished_recving=list(request_ids), finished_sending=[])


def _reload_scheduler(clock: _Clock, connector: _Connector) -> Any:
    return ContinuumScheduler(
        continuum_clock=clock, connector=connector, continuum_environ={}
    )


def test_reload_mode_is_disabled_without_a_kv_connector() -> None:
    scheduler = ContinuumScheduler(continuum_clock=_Clock(0.0), continuum_environ={})

    assert scheduler._continuum_reload_mode == "disabled"
    assert scheduler._continuum_original_get_matched is None
    assert scheduler._continuum_tier_available is False


def test_reload_timer_records_a_sample_across_the_async_load_window() -> None:
    clock = _Clock(100.0)
    connector = _Connector([(4096, True)])
    scheduler = _reload_scheduler(clock, connector)
    assert scheduler._continuum_reload_mode == "online"
    assert scheduler._continuum_tier_available is True

    request = _request("request")
    scheduler.connector.get_num_new_matched_tokens(request, 0)
    assert scheduler._continuum_reload_pending == {"request": (100.0, 4096)}

    clock.now = 100.25
    scheduler._update_from_kv_xfer_finished(_xfer("request"))

    assert scheduler._continuum_reload_pending == {}
    assert scheduler._continuum_pin_stats.reload_samples == 1
    estimator = scheduler._continuum_ttl_policy.reconstruction.reload
    assert estimator.samples() == ((4096.0, 0.25),)


def test_reload_timer_ignores_synchronous_loads() -> None:
    clock = _Clock(100.0)
    connector = _Connector([(4096, False)])
    scheduler = _reload_scheduler(clock, connector)

    scheduler.connector.get_num_new_matched_tokens(_request("request"), 0)
    clock.now = 100.25
    scheduler._update_from_kv_xfer_finished(_xfer("request"))

    assert scheduler._continuum_reload_pending == {}
    assert scheduler._continuum_pin_stats.reload_samples == 0


def test_reload_estimator_falls_back_when_load_async_is_disabled() -> None:
    connector = _Connector([(4096, True)])
    scheduler = ContinuumScheduler(
        continuum_clock=_Clock(0.0),
        connector=connector,
        continuum_environ={},
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config={"load_async": "false"}
        ),
    )

    assert scheduler._continuum_reload_mode == "prefill_fallback"
    # The wrapper is not installed, so the connector keeps its own method.
    assert scheduler._continuum_original_get_matched is None
    assert scheduler.connector.get_num_new_matched_tokens.__self__ is connector
    # A tier still exists, so CacheMissCost knows it is estimating conservatively.
    assert scheduler._continuum_tier_available is True


def test_silent_reload_guard_fires_once_after_enough_decisions() -> None:
    scheduler = _reload_scheduler(_Clock(0.0), _Connector())
    telemetry = scheduler._continuum_telemetry

    telemetry.counters.decisions = 63
    assert telemetry.silent_reload_detected(64) is False
    telemetry.counters.decisions = 64
    assert telemetry.silent_reload_detected(64) is True
    assert telemetry.silent_reload_detected(64) is False


def test_silent_reload_guard_stays_quiet_once_samples_arrive() -> None:
    scheduler = _reload_scheduler(_Clock(0.0), _Connector())
    telemetry = scheduler._continuum_telemetry

    telemetry.counters.decisions = 200
    telemetry.counters.reload_samples = 1
    assert telemetry.silent_reload_detected(64) is False


def test_shutdown_restores_the_connector_lookup() -> None:
    connector = _Connector()
    original = connector.get_num_new_matched_tokens
    scheduler = _reload_scheduler(_Clock(0.0), connector)
    assert connector.get_num_new_matched_tokens is not original

    scheduler.shutdown()

    assert connector.get_num_new_matched_tokens == original
    assert scheduler._continuum_original_get_matched is None


def test_tier_availability_reaches_the_ttl_policy() -> None:
    seen: list[bool] = []

    class _RecordingPolicy:
        def on_request_arrival(self, metadata: Any, now: float) -> None:
            return None

        def on_request_finished(
            self,
            metadata: Any,
            output_text: Any,
            now: float,
            num_context_tokens: int = 0,
            tier_available: bool = False,
        ) -> Any:
            seen.append(tier_available)
            return None

        def observe_queue_delay(self, queue_delay: float) -> None:
            return None

        def observe_program(self, num_turns: int) -> None:
            return None

        def observe_reload(self, num_tokens: int, seconds: float) -> None:
            return None

        def on_program_finished(self, job_id: str, completed: bool) -> None:
            return None

    scheduler = ContinuumScheduler(
        continuum_clock=_Clock(0.0),
        connector=_Connector(),
        continuum_environ={},
        continuum_ttl_policy=_RecordingPolicy(),
    )
    request = _request("request")
    scheduler.add_request(request)
    scheduler._free_request(request)

    assert seen == [True]


class _ProgramPolicy:
    """Records the program lifecycle calls the scheduler makes."""

    def __init__(self) -> None:
        self.finished: list[tuple[str, bool]] = []
        self.programs: list[int] = []
        self.queue_delays: list[float] = []

    def on_request_arrival(self, metadata: Any, now: float) -> None:
        return None

    def on_request_finished(
        self,
        metadata: Any,
        output_text: Any,
        now: float,
        num_context_tokens: int = 0,
        tier_available: bool = False,
    ) -> Any:
        return None

    def observe_queue_delay(self, queue_delay: float) -> None:
        self.queue_delays.append(queue_delay)

    def observe_program(self, num_turns: int) -> None:
        self.programs.append(num_turns)

    def observe_reload(self, num_tokens: int, seconds: float) -> None:
        return None

    def on_program_finished(self, job_id: str, completed: bool) -> None:
        self.finished.append((job_id, completed))


def test_last_step_reports_a_completed_program() -> None:
    policy = _ProgramPolicy()
    scheduler = ContinuumScheduler(
        continuum_clock=_Clock(10.0), continuum_environ={}, continuum_ttl_policy=policy
    )
    first = _request("first")
    last = _request("last", is_last_step=1)

    scheduler.add_request(first)
    scheduler.add_request(last)
    scheduler._free_request(last)

    assert policy.finished == [("job", True)]
    assert policy.programs == [2]


def test_failed_turn_reports_an_abandoned_program() -> None:
    policy = _ProgramPolicy()
    scheduler = ContinuumScheduler(
        continuum_clock=_Clock(10.0), continuum_environ={}, continuum_ttl_policy=policy
    )
    failed = _request("failed", status=_RequestStatus.FINISHED_ABORTED)

    scheduler.add_request(failed)
    scheduler._free_request(failed)

    assert policy.finished == [("job", False)]
    assert policy.programs == []


def test_queue_delay_is_measured_from_eviction_to_readmission() -> None:
    clock = _Clock(10.0)
    policy = _ProgramPolicy()
    scheduler = ContinuumScheduler(
        continuum_clock=clock, continuum_environ={}, continuum_ttl_policy=policy
    )
    # A pin for this job is dropped, so the job is marked evicted.
    scheduler._continuum_pins.pin("job", _request("old"), 12.0, 1.0, "pytest")
    clock.now = 13.0
    scheduler.schedule()
    assert "job" in scheduler._continuum_evicted_jobs

    # The next turn arrives and waits before being admitted.
    clock.now = 20.0
    nxt = _request("next")
    scheduler.add_request(nxt)
    clock.now = 23.5
    scheduler._continuum_allocate_slots(nxt)

    assert policy.queue_delays == [3.5]
    # The sample is taken once, not on every later allocation.
    clock.now = 30.0
    scheduler._continuum_allocate_slots(nxt)
    assert policy.queue_delays == [3.5]


def test_eviction_marks_only_the_matching_job_already_waiting() -> None:
    clock = _Clock(10.0)
    policy = _ProgramPolicy()
    scheduler = ContinuumScheduler(
        continuum_clock=clock, continuum_environ={}, continuum_ttl_policy=policy
    )
    mine = _request("mine", job_id="mine")
    other = _request("other", job_id="other")
    scheduler.add_request(mine)
    scheduler.add_request(other)

    scheduler._mark_continuum_evicted("mine")

    assert set(scheduler._continuum_evicted_waiting) == {"mine"}

    clock.now = 14.0
    scheduler._continuum_allocate_slots(mine)
    scheduler._continuum_allocate_slots(other)
    assert policy.queue_delays == [4.0]


def test_unranked_scan_does_not_sort_the_waiting_queue() -> None:
    scheduler = ContinuumScheduler(continuum_clock=_Clock(10.0), continuum_environ={})
    late = _request("late", job_id="late", arrival_time=9.0)
    early = _request("early", job_id="early", arrival_time=1.0)
    scheduler.add_request(late)
    scheduler.add_request(early)

    # Ranked order is by program arrival; storage order is insertion order.
    assert [r.request_id for r in scheduler.waiting] == ["early", "late"]
    assert [r.request_id for r in scheduler.waiting.unranked()] == ["late", "early"]
