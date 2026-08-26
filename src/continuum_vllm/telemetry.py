"""Continuum runtime counters, periodic logging, and offline dump.

This module never imports vLLM so that it stays testable on a CPU-only host.
"""

import json
import os
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from continuum_vllm.policy import (
    ConstantPrefillCostEstimator,
    PaperTTLPolicy,
    QuadraticPrefillCostEstimator,
    TTLDecision,
)

SCHEMA = "continuum-telemetry/1"
DEFAULT_ESTIMATE_POINTS = (1000, 4000, 16000, 32000, 65536)
ReleaseReason = str


def _percentile(ordered: Sequence[float], fraction: float) -> float:
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


@dataclass
class ContinuumCounters:
    """Counters for every pin lifecycle transition."""

    decisions: int = 0
    pins: int = 0
    ttl_expirations: int = 0
    handoffs: int = 0
    pressure_unpins: int = 0
    final_releases: int = 0
    reload_samples: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "decisions": self.decisions,
            "pins": self.pins,
            "ttl_expirations": self.ttl_expirations,
            "handoffs": self.handoffs,
            "pressure_unpins": self.pressure_unpins,
            "final_releases": self.final_releases,
            "reload_samples": self.reload_samples,
        }


@dataclass
class ContinuumTelemetry:
    """Aggregate Continuum runtime state for logging and offline analysis."""

    policy: PaperTTLPolicy
    reload_mode: str = "disabled"
    reload_detail: str = ""
    log_interval_seconds: float = 60.0
    dump_path: str | None = None
    clock: Callable[[], float] = time.monotonic
    wall_clock: Callable[[], float] = time.time
    estimate_points: tuple[int, ...] = DEFAULT_ESTIMATE_POINTS

    counters: ContinuumCounters = field(default_factory=ContinuumCounters)
    ttl_sources: Counter = field(default_factory=Counter)
    reconstruction_sources: Counter = field(default_factory=Counter)
    _started_at: float = field(default=0.0, init=False)
    _last_log_at: float = field(default=0.0, init=False)
    _silent_reload_warned: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self._started_at = self.clock()
        self._last_log_at = self._started_at
        self._introspectable = isinstance(self.policy, PaperTTLPolicy)

    # ---- recording -------------------------------------------------

    def record_decision(self, decision: TTLDecision | None) -> None:
        self.counters.decisions += 1
        if decision is None:
            return
        self.ttl_sources[decision.source] += 1
        self.reconstruction_sources[decision.reconstruction_source] += 1

    def record_pin(self) -> None:
        self.counters.pins += 1

    def record_release(self, reason: ReleaseReason) -> None:
        if reason == "ttl":
            self.counters.ttl_expirations += 1
        elif reason == "handoff":
            self.counters.handoffs += 1
        elif reason == "pressure":
            self.counters.pressure_unpins += 1
        elif reason == "final":
            self.counters.final_releases += 1

    def record_reload_sample(self) -> None:
        self.counters.reload_samples += 1

    # ---- guard -----------------------------------------------------

    def silent_reload_detected(self, threshold: int) -> bool:
        """True once, when a tier is attached but produced no samples.

        Catches synchronous connectors that never enter the scheduler's
        asynchronous load window, whatever their class name is.
        """
        if self._silent_reload_warned or self.reload_mode != "online":
            return False
        if self.counters.decisions < threshold:
            return False
        if self.counters.reload_samples > 0:
            return False
        self._silent_reload_warned = True
        return True

    # ---- reporting -------------------------------------------------

    def due_for_log(self) -> bool:
        if self.log_interval_seconds <= 0:
            return False
        now = self.clock()
        if now - self._last_log_at < self.log_interval_seconds:
            return False
        self._last_log_at = now
        return True

    def log_line(self) -> str:
        counters = self.counters
        if not self._introspectable:
            return (
                f"Continuum stats: mode={self.reload_mode} "
                f"policy={type(self.policy).__name__} {counters.as_dict()}"
            )
        estimator = self.policy.reconstruction.reload
        fitted = estimator.fit()
        if fitted is None:
            curve = "reload_fit=none"
        else:
            slope, intercept = fitted
            curve = (
                f"reload_us_per_token={slope * 1e6:.3f} "
                f"reload_base_ms={intercept * 1e3:.2f}"
            )
        return (
            f"Continuum stats: mode={self.reload_mode} "
            f"decisions={counters.decisions} pins={counters.pins} "
            f"ttl_expired={counters.ttl_expirations} handoff={counters.handoffs} "
            f"pressure={counters.pressure_unpins} "
            f"reload_window={estimator.sample_count} "
            f"reload_total={counters.reload_samples} "
            f"reload_warm={estimator.is_warm} {curve} "
            f"ttl_src={dict(self.ttl_sources)} "
            f"recon_src={dict(self.reconstruction_sources)} "
            f"queue_delay_s={self.policy.queue_delay.value:.3f} "
            f"eta={self.policy.memoryfulness.value:.3f}"
        )

    def _prefill_section(self) -> dict[str, Any]:
        estimator = self.policy.reconstruction.prefill
        section: dict[str, Any] = {"estimator": type(estimator).__name__}
        if isinstance(estimator, QuadraticPrefillCostEstimator):
            section["coefficients"] = {
                "quadratic": estimator.quadratic,
                "linear": estimator.linear,
                "constant": estimator.constant,
            }
        elif isinstance(estimator, ConstantPrefillCostEstimator):
            section["coefficients"] = {"seconds": estimator.seconds}
        section["estimates_seconds"] = {
            str(point): estimator.estimate_seconds(point)
            for point in self.estimate_points
        }
        return section

    def _reload_section(self) -> dict[str, Any]:
        estimator = self.policy.reconstruction.reload
        fitted = estimator.fit()
        section: dict[str, Any] = {
            "mode": self.reload_mode,
            "detail": self.reload_detail,
            "sample_count": estimator.sample_count,
            "is_warm": estimator.is_warm,
            "observed_total": self.counters.reload_samples,
        }
        if fitted is not None:
            slope, intercept = fitted
            section["slope_seconds_per_token"] = slope
            section["intercept_seconds"] = intercept
            section["microseconds_per_token"] = slope * 1e6
        section["estimates_seconds"] = {
            str(point): estimator.estimate_seconds(point)
            for point in self.estimate_points
        }
        section["samples"] = [[int(x), y] for x, y in estimator.samples()]
        return section

    def _tool_section(self) -> dict[str, Any]:
        history = self.policy.tool_history
        tools: dict[str, Any] = {}
        for name in sorted(history.tool_names()):
            samples = sorted(history.tool_samples(name))
            if not samples:
                continue
            tools[name] = {
                "count": len(samples),
                "mean_seconds": sum(samples) / len(samples),
                "p50_seconds": _percentile(samples, 0.50),
                "p90_seconds": _percentile(samples, 0.90),
                "p99_seconds": _percentile(samples, 0.99),
                "max_seconds": samples[-1],
            }
        return {
            "global_count": history.global_count,
            "threshold": self.policy.config.history_threshold,
            "by_tool": tools,
        }

    def snapshot(self) -> dict[str, Any]:
        document: dict[str, Any] = {
            "schema": SCHEMA,
            "generated_at": self.wall_clock(),
            "uptime_seconds": self.clock() - self._started_at,
            "counters": self.counters.as_dict(),
            "ttl_sources": dict(self.ttl_sources),
            "reconstruction_sources": dict(self.reconstruction_sources),
        }
        if not self._introspectable:
            document["policy"] = type(self.policy).__name__
            return document
        document.update(
            {
                "queue_delay_seconds": self.policy.queue_delay.value,
                "memoryfulness": self.policy.memoryfulness.value,
                "prefill": self._prefill_section(),
                "reload": self._reload_section(),
                "tools": self._tool_section(),
            }
        )
        return document

    def dump(self) -> str | None:
        """Write the snapshot atomically. Returns the path, or None."""
        if not self.dump_path:
            return None
        payload = json.dumps(self.snapshot(), indent=2, sort_keys=True)
        directory = os.path.dirname(os.path.abspath(self.dump_path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        temporary = f"{self.dump_path}.{os.getpid()}.tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            handle.write(payload)
        os.replace(temporary, self.dump_path)
        return self.dump_path


def telemetry_dump_path_from_env(
    values: dict[str, str] | Any = None,
) -> str | None:
    values = os.environ if values is None else values
    path = values.get("CONTINUUM_STATS_DUMP_PATH")
    if not path:
        return None
    # One file per engine process so data-parallel ranks do not overwrite.
    root, extension = os.path.splitext(path)
    return f"{root}.{os.getpid()}{extension or '.json'}"
