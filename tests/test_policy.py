from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from continuum_vllm.policy import (
    BashToolCallParser,
    ConstantPrefillCostEstimator,
    ContinuumRequestMetadata,
    MemoryfulnessEstimator,
    OnlineReloadCostEstimator,
    PaperTTLConfig,
    PaperTTLPolicy,
    PendingToolCalls,
    PinManager,
    QuadraticPrefillCostEstimator,
    ReconstructionCostModel,
    SlidingWindowMean,
    ToolDurationHistory,
    paper_ttl_policy_from_env,
)


def _prefill_only(seconds: float) -> ReconstructionCostModel:
    return ReconstructionCostModel(prefill=ConstantPrefillCostEstimator(seconds))


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
        reconstruction=_prefill_only(10.0),
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
        reconstruction=_prefill_only(10.0),
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
        reconstruction=_prefill_only(0.0),
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
        reconstruction=_prefill_only(0.0),
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

    quadratic = QuadraticPrefillCostEstimator(0.001, 0.1, 1.0)
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
    assert policy.reconstruction.prefill.estimate_seconds(10) == 2.1


def test_policy_configuration_requires_complete_quadratic_profile() -> None:
    with pytest.raises(ValueError, match="requires quadratic, linear, and constant"):
        paper_ttl_policy_from_env({"CONTINUUM_PREFILL_QUADRATIC": "0.001"})


def test_policy_configuration_rejects_invalid_numbers() -> None:
    with pytest.raises(ValueError, match="CONTINUUM_HISTORY_THRESHOLD"):
        paper_ttl_policy_from_env({"CONTINUUM_HISTORY_THRESHOLD": "many"})


def test_reload_estimator_is_cold_until_the_minimum_sample_count() -> None:
    estimator = OnlineReloadCostEstimator(max_samples=8, min_samples=4)

    for tokens in (1000, 2000, 3000):
        estimator.observe(tokens, tokens * 1e-6)
    assert not estimator.is_warm
    assert estimator.estimate_seconds(4000) is None

    estimator.observe(4000, 4000e-6)
    assert estimator.is_warm
    assert estimator.estimate_seconds(4000) == pytest.approx(4000e-6, rel=1e-6)


def test_reload_estimator_fits_a_line_and_forgets_old_samples() -> None:
    estimator = OnlineReloadCostEstimator(max_samples=3, min_samples=2)
    for tokens in (1000, 2000, 3000):
        estimator.observe(tokens, 0.01 + tokens * 2e-6)

    slope, intercept = estimator.fit()
    assert slope == pytest.approx(2e-6, rel=1e-6)
    assert intercept == pytest.approx(0.01, abs=1e-9)

    # A bounded window drops the oldest sample rather than growing.
    estimator.observe(4000, 0.02 + 4000 * 4e-6)
    assert estimator.sample_count == 3
    assert estimator.samples()[0][0] == 2000

    estimator.observe(0, 1.0)
    estimator.observe(1000, -1.0)
    assert estimator.sample_count == 3


def test_reconstruction_model_selects_reload_only_when_a_warm_tier_exists() -> None:
    model = ReconstructionCostModel(
        prefill=ConstantPrefillCostEstimator(5.0),
        reload=OnlineReloadCostEstimator(max_samples=8, min_samples=2),
    )

    assert model.estimate(1000, tier_available=False) == (5.0, "prefill")
    assert model.estimate(1000, tier_available=True) == (5.0, "prefill_fallback")

    model.observe_reload(1000, 0.1)
    model.observe_reload(2000, 0.2)
    seconds, source = model.estimate(1000, tier_available=True)
    assert source == "reload"
    assert seconds == pytest.approx(0.1, rel=1e-6)
    # No tier still means a full recompute.
    assert model.estimate(1000, tier_available=False) == (5.0, "prefill")


def test_tier_reload_shortens_ttl_relative_to_recompute() -> None:
    config = PaperTTLConfig(
        history_threshold=0, max_history_samples=10, reload_min_samples=2
    )
    model = ReconstructionCostModel(
        prefill=ConstantPrefillCostEstimator(10.0),
        reload=OnlineReloadCostEstimator(max_samples=8, min_samples=2),
    )
    policy = PaperTTLPolicy(config=config, reconstruction=model)
    for duration in (1.0, 2.0, 4.0, 6.0, 12.0):
        policy.tool_history.observe("pytest", duration)
    metadata = ContinuumRequestMetadata(job_id="job", this_func_call="pytest")

    recompute = policy.on_request_finished(metadata, None, 10.0, 1000, False)
    assert recompute is not None
    assert recompute.reconstruction_source == "prefill"
    assert recompute.reconstruction_seconds == 10.0

    policy.observe_reload(1000, 0.05)
    policy.observe_reload(2000, 0.10)
    tiered = policy.on_request_finished(metadata, None, 20.0, 1000, True)
    assert tiered is not None
    assert tiered.reconstruction_source == "reload"
    assert tiered.reconstruction_seconds == pytest.approx(0.05, rel=1e-6)
    assert tiered.ttl_seconds < recompute.ttl_seconds


def test_policy_configuration_reads_a_prefill_profile_file(tmp_path) -> None:
    profile = tmp_path / "prefill.json"
    profile.write_text(
        '{"prefill_coefficients": '
        '{"quadratic": 0.001, "linear": 0.1, "constant": 1.0}}',
        encoding="utf-8",
    )

    policy = paper_ttl_policy_from_env({"CONTINUUM_PREFILL_PROFILE": str(profile)})

    assert policy.reconstruction.prefill.estimate_seconds(10) == 2.1


def test_policy_configuration_rejects_a_malformed_prefill_profile(tmp_path) -> None:
    profile = tmp_path / "prefill.json"
    profile.write_text('{"nope": 1}', encoding="utf-8")

    with pytest.raises(ValueError, match="not a continuum prefill profile"):
        paper_ttl_policy_from_env({"CONTINUUM_PREFILL_PROFILE": str(profile)})


def test_policy_configuration_reads_reload_window_from_environment() -> None:
    policy = paper_ttl_policy_from_env(
        {
            "CONTINUUM_RELOAD_MAX_SAMPLES": "64",
            "CONTINUUM_RELOAD_MIN_SAMPLES": "8",
        }
    )

    assert policy.config.reload_max_samples == 64
    assert policy.reconstruction.reload.min_samples == 8


def test_pending_tool_calls_expire_instead_of_recording_a_fake_duration() -> None:
    pending = PendingToolCalls(max_entries=8, max_age_seconds=60.0)
    pending.start("job", 100.0, "pytest")

    assert pending.finish("job", 100.0 + 61.0) is None
    assert pending.expired == 1
    assert len(pending) == 0


def test_pending_tool_calls_record_a_duration_inside_the_window() -> None:
    pending = PendingToolCalls(max_entries=8, max_age_seconds=60.0)
    pending.start("job", 100.0, "pytest")

    assert pending.finish("job", 103.5) == (3.5, "pytest")
    assert pending.expired == 0
    assert pending.finish("job", 104.0) is None


def test_pending_tool_calls_are_bounded_and_pruned() -> None:
    pending = PendingToolCalls(max_entries=2, max_age_seconds=60.0)
    pending.start("a", 100.0, "x")
    pending.start("b", 100.0, "y")
    pending.start("c", 100.0, "z")

    assert len(pending) == 2
    assert pending.evicted == 1
    assert "a" not in pending

    # Starting a new call also ages out anything past the window.
    pending.start("d", 200.0, "w")
    assert pending.expired == 2
    assert "d" in pending
    assert "b" not in pending
    assert len(pending) == 1


def test_abandoned_program_does_not_leak_a_pending_tool_call() -> None:
    policy = PaperTTLPolicy()
    metadata = ContinuumRequestMetadata(job_id="job", this_func_call="pytest")
    policy.on_request_finished(metadata, None, 10.0)
    assert len(policy.pending_tools) == 1

    policy.on_program_finished("job", completed=False)

    assert len(policy.pending_tools) == 0
    assert policy.abandoned_programs == 1
    assert policy.completed_programs == 0


def test_completed_program_is_counted_separately() -> None:
    policy = PaperTTLPolicy()

    policy.on_program_finished("job", completed=True)

    assert policy.completed_programs == 1
    assert policy.abandoned_programs == 0


def test_stale_program_does_not_pollute_the_tool_history() -> None:
    config = PaperTTLConfig(pending_tool_max_age_seconds=60.0)
    policy = PaperTTLPolicy(config=config)
    policy.on_request_finished(
        ContinuumRequestMetadata(job_id="job", this_func_call="pytest"), None, 10.0
    )

    # The next turn arrives an hour later; that gap is an abandoned session,
    # not a tool execution.
    policy.on_request_arrival(
        ContinuumRequestMetadata(job_id="job", last_func_call="pytest"), 10.0 + 3600.0
    )

    assert policy.tool_history.tool_samples("pytest") == ()
    assert policy.pending_tools.expired == 1


def test_ttl_stage_walks_the_three_cold_start_tiers() -> None:
    config = PaperTTLConfig(history_threshold=2, max_history_samples=10)
    policy = PaperTTLPolicy(config=config)
    assert policy.ttl_stage == "cold_start"

    # Enough global samples, but spread thin so no single tool has graduated.
    for name in ("a", "b", "c"):
        policy.tool_history.observe(name, 1.0)
    assert policy.tool_history.global_count == 3
    assert policy.ttl_stage == "global"

    # Once one tool passes the threshold, the per-tool tier is reachable.
    for duration in (1.0, 2.0, 4.0):
        policy.tool_history.observe("pytest", duration)
    assert policy.ttl_stage == "tool"


def test_queue_delay_reports_whether_it_has_any_samples() -> None:
    window = SlidingWindowMean(window_size=4)

    assert window.is_warm is False
    assert window.value == 0.0
    assert window.sample_count == 0
    assert window.window_size == 4

    window.observe(2.0)
    assert window.is_warm is True
    assert window.sample_count == 1


def test_memoryfulness_is_uninformative_until_lengths_differ() -> None:
    memoryfulness = MemoryfulnessEstimator()

    memoryfulness.observe_program(4)
    # Within one program k and N-k are perfectly anti-correlated.
    assert memoryfulness.value == 1.0
    assert memoryfulness.program_count == 1
    assert memoryfulness.distinct_length_count == 1
    assert memoryfulness.is_informative is False

    memoryfulness.observe_program(4)
    assert memoryfulness.is_informative is False

    memoryfulness.observe_program(9)
    assert memoryfulness.distinct_length_count == 2
    assert memoryfulness.is_informative is True


def test_inline_and_public_finish_probability_agree() -> None:
    samples = (0.5, 1.0, 1.0, 2.5, 4.0, 9.0)
    config = PaperTTLConfig(history_threshold=0, max_history_samples=10)
    policy = PaperTTLPolicy(config=config, reconstruction=_prefill_only(10.0))
    for duration in samples:
        policy.tool_history.observe("pytest", duration)

    decision = policy.on_request_finished(
        ContinuumRequestMetadata(job_id="job", this_func_call="pytest"), None, 10.0
    )

    assert decision is not None
    assert decision.finish_probability == ToolDurationHistory.finish_probability(
        samples, decision.ttl_seconds
    )
