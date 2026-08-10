"""Runtime compatibility checks for vLLM internal APIs."""

from importlib.metadata import PackageNotFoundError, version
from inspect import signature
from typing import Any

SUPPORTED_VLLM_RELEASE = (0, 25, 1)


def assert_compatible_vllm(installed_version: str | None = None) -> str:
    """Fail before loading private scheduler APIs from another vLLM release."""
    if installed_version is None:
        try:
            installed_version = version("vllm")
        except PackageNotFoundError as exc:
            raise RuntimeError("continuum-vllm requires vLLM 0.25.1") from exc

    release = _release_tuple(installed_version)
    if release != SUPPORTED_VLLM_RELEASE:
        expected = ".".join(str(part) for part in SUPPORTED_VLLM_RELEASE)
        raise RuntimeError(
            f"continuum-vllm supports vLLM {expected}, found {installed_version}"
        )
    return installed_version


def assert_scheduler_contract() -> None:
    """Validate the private vLLM hooks used by the custom Scheduler."""
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.sched.scheduler import Scheduler

    _assert_scheduler_contract(Scheduler, KVCacheManager)


def _assert_scheduler_contract(scheduler_cls: Any, kv_cache_manager_cls: Any) -> None:
    required_methods = {
        "add_request",
        "schedule",
        "_select_waiting_queue_for_scheduling",
        "_free_request",
        "_free_blocks",
        "_free_request_blocks",
        "reset_prefix_cache",
        "shutdown",
    }
    missing = sorted(
        name for name in required_methods if not hasattr(scheduler_cls, name)
    )
    if missing:
        raise RuntimeError(
            "vLLM Scheduler contract mismatch; missing methods: " + ", ".join(missing)
        )

    parameters = signature(kv_cache_manager_cls.allocate_slots).parameters
    required_parameters = {"self", "request", "num_new_tokens"}
    missing_parameters = sorted(required_parameters - parameters.keys())
    if missing_parameters:
        raise RuntimeError(
            "vLLM KVCacheManager contract mismatch; allocate_slots is missing: "
            + ", ".join(missing_parameters)
        )


def _release_tuple(value: str) -> tuple[int, int, int] | None:
    public = value.split("+", maxsplit=1)[0]
    parts = public.split(".")
    if len(parts) < 3:
        return None
    try:
        return tuple(int(part) for part in parts[:3])
    except ValueError:
        return None
