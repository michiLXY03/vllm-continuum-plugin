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


class _Scheduler:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(skip_tokenizer_init=True)
        )
        self.kv_cache_manager = kwargs.pop("kv_cache_manager", _KVCacheManager())
        self.requests: dict[str, Any] = {}
        self.waiting = _FCFSRequestQueue()
        self.skipped_waiting = _FCFSRequestQueue()
        self.shutdown_called = False

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
