import json

import pytest

from continuum_vllm.policy import (
    ConstantPrefillCostEstimator,
    ContinuumRequestMetadata,
    OnlineReloadCostEstimator,
    PaperTTLConfig,
    PaperTTLPolicy,
    QuadraticPrefillCostEstimator,
    ReconstructionCostModel,
)
from continuum_vllm.profile import (
    ProfileError,
    evaluate,
    fit_quadratic,
    r_squared,
    solve_3x3,
)
from continuum_vllm.report import main as report_main
from continuum_vllm.telemetry import ContinuumTelemetry


class _Clock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def test_solve_3x3_handles_pivoting() -> None:
    solution = solve_3x3([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 2.0]], [2, 3, 8])

    assert solution == pytest.approx([3.0, 2.0, 4.0])


def test_fit_quadratic_recovers_known_coefficients_at_scale() -> None:
    truth = {"quadratic": 2.5e-9, "linear": 3.1e-5, "constant": 0.012}
    points = [
        (tokens, evaluate(truth, tokens)) for tokens in range(2048, 65536 + 1, 2048)
    ]

    fitted = fit_quadratic(points)

    for key, value in truth.items():
        assert fitted[key] == pytest.approx(value, rel=1e-6, abs=1e-12)
    assert r_squared(points, fitted) == pytest.approx(1.0)


def test_fit_quadratic_rejects_too_few_distinct_lengths() -> None:
    with pytest.raises(ProfileError, match="three distinct context lengths"):
        fit_quadratic([(1000, 0.1), (1000, 0.2), (2000, 0.3)])


def _telemetry(tmp_path, warm: bool) -> ContinuumTelemetry:
    policy = PaperTTLPolicy(
        config=PaperTTLConfig(history_threshold=0, reload_min_samples=2),
        reconstruction=ReconstructionCostModel(
            prefill=QuadraticPrefillCostEstimator(2.5e-9, 3.1e-5, 0.012),
            reload=OnlineReloadCostEstimator(max_samples=8, min_samples=2),
        ),
    )
    policy.tool_history.observe("pytest", 1.5)
    policy.tool_history.observe("pytest", 3.0)
    if warm:
        policy.observe_reload(1000, 0.02)
        policy.observe_reload(4000, 0.05)
    clock = _Clock(10.0)
    telemetry = ContinuumTelemetry(
        policy=policy,
        reload_mode="online" if warm else "disabled",
        reload_detail="OffloadingConnector" if warm else "no kv_transfer_config",
        dump_path=str(tmp_path / "stats.json"),
        clock=clock,
        wall_clock=lambda: 1000.0,
    )
    telemetry.record_decision(
        policy.on_request_finished(
            ContinuumRequestMetadata(job_id="job", this_func_call="pytest"),
            None,
            10.0,
            4000,
            warm,
        )
    )
    telemetry.record_pin()
    telemetry.record_release("handoff")
    return telemetry


def test_telemetry_snapshot_reports_both_cost_curves(tmp_path) -> None:
    telemetry = _telemetry(tmp_path, warm=True)

    document = telemetry.snapshot()

    assert document["reload"]["is_warm"] is True
    assert document["reload"]["microseconds_per_token"] == pytest.approx(10.0, rel=1e-6)
    assert document["prefill"]["coefficients"]["linear"] == pytest.approx(3.1e-5)
    assert document["reconstruction_sources"] == {"reload": 1}
    assert document["tools"]["by_tool"]["pytest"]["count"] == 2
    assert document["counters"]["handoffs"] == 1


def test_telemetry_dump_is_readable_by_the_report(tmp_path, capsys) -> None:
    telemetry = _telemetry(tmp_path, warm=True)
    written = telemetry.dump()
    assert written is not None
    with open(written, encoding="utf-8") as handle:
        assert json.load(handle)["schema"].startswith("continuum-telemetry/")

    assert report_main([written]) == 0

    output = capsys.readouterr().out
    assert "Continuum stats report" in output
    assert "Reload estimator" in output
    assert "CacheMissCost by context length" in output
    assert "Pin outcomes" in output


def test_report_explains_a_cold_deployment(tmp_path, capsys) -> None:
    telemetry = _telemetry(tmp_path, warm=False)
    written = telemetry.dump()
    assert written is not None

    assert report_main([written]) == 0

    output = capsys.readouterr().out
    assert "no KV tier attached" in output


def test_report_rejects_a_foreign_document(tmp_path, capsys) -> None:
    path = tmp_path / "other.json"
    path.write_text('{"schema": "something-else/1"}', encoding="utf-8")

    assert report_main([str(path)]) == 2
    assert "not a Continuum stats dump" in capsys.readouterr().err


def test_telemetry_degrades_without_a_paper_policy(tmp_path) -> None:
    class _Policy:
        pass

    telemetry = ContinuumTelemetry(policy=_Policy(), clock=_Clock(0.0))

    assert "policy=_Policy" in telemetry.log_line()
    assert telemetry.snapshot()["policy"] == "_Policy"


def test_prefill_estimator_stays_constant_without_a_profile() -> None:
    estimator = ConstantPrefillCostEstimator(2.0)

    assert estimator.estimate_seconds(1000) == 2.0
    assert estimator.estimate_seconds(64000) == 2.0
