"""Installation check that does not initialize a model or device."""

from continuum_vllm.compat import assert_compatible_vllm, assert_scheduler_contract


def main() -> None:
    """Validate version and Scheduler imports, then print the active contract."""
    installed_version = assert_compatible_vllm()
    assert_scheduler_contract()

    from vllm.v1.core.sched.async_scheduler import AsyncScheduler
    from vllm.v1.core.sched.scheduler import Scheduler

    from continuum_vllm.scheduler import (
        AsyncContinuumScheduler,
        ContinuumScheduler,
    )

    if not issubclass(ContinuumScheduler, Scheduler):
        raise TypeError("ContinuumScheduler is not compatible with Scheduler")
    if not issubclass(AsyncContinuumScheduler, AsyncScheduler):
        raise TypeError("AsyncContinuumScheduler is not compatible with AsyncScheduler")

    from continuum_vllm.policy import paper_ttl_policy_from_env

    policy = paper_ttl_policy_from_env()
    prefill = policy.reconstruction.prefill
    reload_estimator = policy.reconstruction.reload

    print(f"vLLM: {installed_version}")
    print("sync scheduler: continuum_vllm.scheduler.ContinuumScheduler")
    print("async scheduler: continuum_vllm.scheduler.AsyncContinuumScheduler")
    print(f"prefill estimator: {type(prefill).__name__}")
    for tokens in (1000, 8000, 32000):
        print(f"  prefill({tokens}) = {prefill.estimate_seconds(tokens):.4f}s")
    print(
        f"reload estimator: online, warm after {reload_estimator.min_samples} samples"
    )
