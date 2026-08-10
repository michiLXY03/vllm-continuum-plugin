from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from continuum_vllm.policy import (
    BashToolCallParser,
    ConstantReconstructionCostEstimator,
    ContinuumRequestMetadata,
    MemoryfulnessEstimator,
    PaperTTLConfig,
    PaperTTLPolicy,
    PinManager,
    QuadraticReconstructionCostEstimator,
    SlidingWindowMean,
    ToolDurationHistory,
    paper_ttl_policy_from_env,
)


@dataclass
class _Request:
    request_id: str


class _FixedMemoryfulness(MemoryfulnessEstimator):
    def __init__(self, value: float) -> None:
        super().__init__()
        self._value = value

    @property
    def value(self) -> float:
        return self._value


def test_metadata_reads_native_vllm_xargs_values() -> None:
    request = SimpleNamespace(
        sampling_params=SimpleNamespace(
            extra_args={
                "job_id": 42,
                "last_func_call": "pytest",
                "is_last_step": 1,
            }
        )
    )

    metadata = ContinuumRequestMetadata.from_request(request)

    assert metadata.job_id == "42"
    assert metadata.last_func_call == "pytest"
    assert metadata.is_last_step is True


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0, False), (1, True), ("false", False), ("true", True), (2, None)],
)
def test_metadata_normalizes_vllm_xargs_booleans(value, expected) -> None:
    metadata = ContinuumRequestMetadata.from_extra_args({"is_last_step": value})
    assert metadata.is_last_step is expected


def test_pin_manager_replaces_and_expires_by_job() -> None:
    pins = PinManager()
    first = _Request("first")
    second = _Request("second")

    assert pins.pin("job", first, 2.0, 1.0, "pytest") is None
    replaced = pins.pin("job", second, 3.0, 1.0, "pytest")

    assert replaced is not None
    assert replaced.request is first
    assert pins.expire(2.9) == []
    assert [pin.request for pin in pins.expire(3.0)] == [second]


def test_claimed_pin_survives_ttl_until_handoff() -> None:
    pins = PinManager()
    request = _Request("finished")
    pins.pin("job", request, 2.0, 1.0, "pytest")

    claimed = pins.claim("job", "next-request")

    assert claimed is not None
    assert claimed.claimed_by == "next-request"
    assert pins.expire(3.0) == []
    assert pins.handoff("job") is claimed


def test_pressure_release_prefers_latest_program_arrival() -> None:
    pins = PinManager()
    older = _Request("older")
    newer = _Request("newer")
    pins.pin("older", older, 5.0, 1.0, "pytest")
    pins.pin("newer", newer, 2.0, 2.0, "pytest")
    pins.claim("newer", "next-request")

    released = pins.release_for_pressure()

    assert released is not None
    assert released.request is newer
    assert "older" in pins


def test_paper_ttl_uses_cold_start_until_global_history_exceeds_k() -> None:
    config = PaperTTLConfig(
        history_threshold=2,
        cold_start_ttl_seconds=3.0,
        max_history_samples=10,
    )
    policy = PaperTTLPolicy(config=config)
    metadata = ContinuumRequestMetadata(job_id="job", this_func_call="pytest")

    decision = policy.on_request_finished(metadata, None, 10.0, 1000)

    assert decision is not None
    assert decision.ttl_seconds == 3.0
    assert decision.source == "cold_start"


def test_paper_ttl_switches_from_global_to_per_tool_history() -> None:
    config = PaperTTLConfig(history_threshold=2, max_history_samples=10)
    policy = PaperTTLPolicy(
        config=config,
        reconstruction_cost=ConstantReconstructionCostEstimator(10.0),
    )
    for duration in (1.0, 2.0, 4.0):
        policy.tool_history.observe("other", duration)
    metadata = ContinuumRequestMetadata(job_id="job", this_func_call="pytest")

    global_decision = policy.on_request_finished(metadata, None, 10.0, 1000)
    assert global_decision is not None
    assert global_decision.source == "global"

    for duration in (1.0, 2.0, 4.0):
        policy.tool_history.observe("pytest", duration)
    tool_decision = policy.on_request_finished(metadata, None, 20.0, 1000)
    assert tool_decision is not None
    assert tool_decision.source == "tool"


def test_paper_ttl_maximizes_empirical_utility_and_prefers_shorter_ties() -> None:
    config = PaperTTLConfig(history_threshold=0, max_history_samples=10)
    policy = PaperTTLPolicy(
        config=config,
        reconstruction_cost=ConstantReconstructionCostEstimator(10.0),
    )
    for duration in (1.0, 2.0, 4.0, 6.0, 12.0):
        policy.tool_history.observe("pytest", duration)

    decision = policy.on_request_finished(
        ContinuumRequestMetadata(job_id="job", this_func_call="pytest"),
        None,
        10.0,
        1000,
    )

    assert decision is not None
    assert decision.ttl_seconds == 2.0
    assert decision.finish_probability == 0.4
    assert decision.expected_utility == 2.0


def test_paper_ttl_returns_zero_when_pinning_has_no_utility() -> None:
    config = PaperTTLConfig(history_threshold=0, max_history_samples=10)
    policy = PaperTTLPolicy(
        config=config,
        reconstruction_cost=ConstantReconstructionCostEstimator(0.0),
    )
    policy.tool_history.observe("pytest", 5.0)

    decision = policy.on_request_finished(
        ContinuumRequestMetadata(job_id="job", this_func_call="pytest"),
        None,
        10.0,
        1000,
    )

    assert decision is not None
    assert decision.ttl_seconds == 0.0
    assert decision.expected_utility == 0.0


def test_paper_ttl_uses_queue_delay_and_memoryfulness() -> None:
    config = PaperTTLConfig(history_threshold=0, max_history_samples=10)
    policy = PaperTTLPolicy(
        config=config,
        reconstruction_cost=ConstantReconstructionCostEstimator(0.0),
    )
    policy.tool_history.observe("pytest", 5.0)
    metadata = ContinuumRequestMetadata(job_id="job", this_func_call="pytest")

    without_delay = policy.on_request_finished(metadata, None, 10.0)
    assert without_delay is not None
    assert without_delay.ttl_seconds == 0.0

    policy.observe_queue_delay(10.0)
    with_memoryfulness = policy.on_request_finished(metadata, None, 20.0)
    assert with_memoryfulness is not None
    assert with_memoryfulness.ttl_seconds == 5.0

    policy.memoryfulness = _FixedMemoryfulness(0.0)
    memoryless = policy.on_request_finished(metadata, None, 30.0)
    assert memoryless is not None
    assert memoryless.ttl_seconds == 0.0


def test_tool_duration_is_recorded_on_next_request_arrival() -> None:
    policy = PaperTTLPolicy()
    policy.on_request_finished(
        ContinuumRequestMetadata(job_id="job", this_func_call="pytest"),
        None,
        10.0,
    )

    policy.on_request_arrival(
        ContinuumRequestMetadata(job_id="job", last_func_call="pytest"), 13.5
    )

    assert policy.tool_history.tool_samples("pytest") == (3.5,)


def test_estimators_and_parser() -> None:
    queue_delay = SlidingWindowMean(window_size=2)
    for value in (2.0, 4.0, 6.0):
        queue_delay.observe(value)
    assert queue_delay.value == 5.0

    memoryfulness = MemoryfulnessEstimator()
    memoryfulness.observe_program(4)
    assert memoryfulness.value == 1.0
    assert memoryfulness.sample_count == 4

    quadratic = QuadraticReconstructionCostEstimator(0.001, 0.1, 1.0)
    assert quadratic.estimate_seconds(10) == 2.1
    assert ToolDurationHistory.finish_probability((1.0, 2.0, 4.0), 2.0) == 2 / 3

    parser = BashToolCallParser()
    assert parser.parse("Run:\n```bash\npytest tests/unit\n```") == "pytest"
    assert parser.parse("No tool call") is None


def test_policy_configuration_reads_quadratic_profile_from_environment() -> None:
    policy = paper_ttl_policy_from_env(
        {
            "CONTINUUM_HISTORY_THRESHOLD": "2",
            "CONTINUUM_COLD_START_TTL_SECONDS": "3.5",
            "CONTINUUM_MAX_HISTORY_SAMPLES": "20",
            "CONTINUUM_QUEUE_DELAY_WINDOW_SIZE": "4",
            "CONTINUUM_PREFILL_QUADRATIC": "0.001",
            "CONTINUUM_PREFILL_LINEAR": "0.1",
            "CONTINUUM_PREFILL_CONSTANT": "1.0",
        }
    )

    assert policy.config.history_threshold == 2
    assert policy.config.cold_start_ttl_seconds == 3.5
    assert policy.reconstruction_cost.estimate_seconds(10) == 2.1


def test_policy_configuration_requires_complete_quadratic_profile() -> None:
    with pytest.raises(ValueError, match="requires quadratic, linear, and constant"):
        paper_ttl_policy_from_env({"CONTINUUM_PREFILL_QUADRATIC": "0.001"})


def test_policy_configuration_rejects_invalid_numbers() -> None:
    with pytest.raises(ValueError, match="CONTINUUM_HISTORY_THRESHOLD"):
        paper_ttl_policy_from_env({"CONTINUUM_HISTORY_THRESHOLD": "many"})
