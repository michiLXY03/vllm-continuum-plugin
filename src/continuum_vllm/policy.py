"""Continuum request metadata, pin lifecycle, and paper TTL model."""

import re
from bisect import bisect_right
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from math import sqrt
from os import environ as process_environ
from typing import Any, Literal, Protocol

DEFAULT_TTL_SECONDS = 2.0
TTLSource = Literal["cold_start", "global", "tool"]


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes"}:
            return True
        if normalized in {"0", "false", "no"}:
            return False
    return None


@dataclass(frozen=True, slots=True)
class ContinuumRequestMetadata:
    """Metadata carried through SamplingParams.extra_args."""

    job_id: str | None = None
    last_func_call: str | None = None
    is_last_step: bool | None = None
    this_func_call: str | None = None

    @classmethod
    def from_extra_args(
        cls, extra_args: dict[str, Any] | None
    ) -> "ContinuumRequestMetadata":
        if not extra_args:
            return cls()
        return cls(
            job_id=_optional_text(extra_args.get("job_id")),
            last_func_call=_optional_text(extra_args.get("last_func_call")),
            is_last_step=_optional_bool(extra_args.get("is_last_step")),
            this_func_call=_optional_text(extra_args.get("this_func_call")),
        )

    @classmethod
    def from_request(cls, request: Any) -> "ContinuumRequestMetadata":
        sampling_params = getattr(request, "sampling_params", None)
        extra_args = getattr(sampling_params, "extra_args", None)
        return cls.from_extra_args(extra_args)

    @property
    def enabled(self) -> bool:
        return self.job_id is not None


@dataclass(frozen=True, slots=True)
class TTLDecision:
    """Result of evaluating the paper TTL objective."""

    expires_at: float
    tool_name: str
    ttl_seconds: float
    source: TTLSource
    finish_probability: float | None = None
    expected_utility: float | None = None

    def is_active(self, now: float) -> bool:
        return self.expires_at > now


@dataclass(frozen=True, slots=True)
class PinnedRequest:
    """A finished request whose KV blocks remain scheduler-owned."""

    job_id: str
    request: Any
    expires_at: float
    program_arrival: float
    tool_name: str
    claimed_by: str | None = None


@dataclass(slots=True)
class ContinuumPinStats:
    """Local counters for normal pin release paths."""

    ttl_expirations: int = 0
    handoffs: int = 0
    pressure_unpins: int = 0


class PinManager:
    """Track at most one pinned request for each agent program."""

    def __init__(self) -> None:
        self._pins: dict[str, PinnedRequest] = {}

    def pin(
        self,
        job_id: str,
        request: Any,
        expires_at: float,
        program_arrival: float,
        tool_name: str,
    ) -> PinnedRequest | None:
        previous = self._pins.get(job_id)
        self._pins[job_id] = PinnedRequest(
            job_id=job_id,
            request=request,
            expires_at=expires_at,
            program_arrival=program_arrival,
            tool_name=tool_name,
            claimed_by=previous.claimed_by if previous is not None else None,
        )
        return previous

    def get(self, job_id: str) -> PinnedRequest | None:
        return self._pins.get(job_id)

    def pop(self, job_id: str) -> PinnedRequest | None:
        return self._pins.pop(job_id, None)

    def claim(self, job_id: str, request_id: str) -> PinnedRequest | None:
        pin = self._pins.get(job_id)
        if pin is None:
            return None
        if pin.claimed_by not in (None, request_id):
            return pin
        claimed = replace(pin, claimed_by=request_id)
        self._pins[job_id] = claimed
        return claimed

    def unclaim(self, job_id: str, request_id: str) -> PinnedRequest | None:
        pin = self._pins.get(job_id)
        if pin is None or pin.claimed_by != request_id:
            return pin
        unclaimed = replace(pin, claimed_by=None)
        self._pins[job_id] = unclaimed
        return unclaimed

    def handoff(self, job_id: str) -> PinnedRequest | None:
        return self._pins.pop(job_id, None)

    def expire(self, now: float) -> list[PinnedRequest]:
        expired = [
            pin
            for pin in self._pins.values()
            if pin.claimed_by is None and pin.expires_at <= now
        ]
        for pin in expired:
            del self._pins[pin.job_id]
        return expired

    def release_for_pressure(self) -> PinnedRequest | None:
        if not self._pins:
            return None
        pin = max(
            self._pins.values(),
            key=lambda item: (item.program_arrival, item.expires_at, item.job_id),
        )
        del self._pins[pin.job_id]
        return pin

    def pop_all(self) -> list[PinnedRequest]:
        pins = list(self._pins.values())
        self._pins.clear()
        return pins

    def __contains__(self, job_id: str) -> bool:
        return job_id in self._pins

    def __len__(self) -> int:
        return len(self._pins)


class TTLPolicy(Protocol):
    """Scheduler-facing TTL policy contract."""

    def on_request_arrival(
        self, metadata: ContinuumRequestMetadata, now: float
    ) -> None: ...

    def on_request_finished(
        self,
        metadata: ContinuumRequestMetadata,
        output_text: str | None,
        now: float,
        num_context_tokens: int = 0,
    ) -> TTLDecision | None: ...

    def observe_queue_delay(self, queue_delay: float) -> None: ...

    def observe_program(self, num_turns: int) -> None: ...


class ReconstructionCostEstimator(Protocol):
    """Estimate Prefill-Reload cost in seconds."""

    def estimate_seconds(self, num_context_tokens: int) -> float: ...


@dataclass(frozen=True, slots=True)
class ConstantReconstructionCostEstimator:
    """Fallback when no hardware/model profile is configured."""

    seconds: float = DEFAULT_TTL_SECONDS

    def __post_init__(self) -> None:
        if self.seconds < 0:
            raise ValueError("reconstruction cost must be non-negative")

    def estimate_seconds(self, num_context_tokens: int) -> float:
        return self.seconds


@dataclass(frozen=True, slots=True)
class QuadraticReconstructionCostEstimator:
    """Evaluate the paper's offline quadratic prefill profile."""

    quadratic: float
    linear: float
    constant: float

    def estimate_seconds(self, num_context_tokens: int) -> float:
        tokens = max(0, num_context_tokens)
        estimate = self.quadratic * tokens**2 + self.linear * tokens + self.constant
        return max(0.0, estimate)


class SlidingWindowMean:
    """O(1) sliding-window average used for queueing delay T."""

    def __init__(self, window_size: int, initial_value: float = 0.0) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self._values: deque[float] = deque()
        self._window_size = window_size
        self._total = 0.0
        self._initial_value = initial_value

    def observe(self, value: float) -> None:
        if value < 0:
            raise ValueError("observed value must be non-negative")
        if len(self._values) == self._window_size:
            self._total -= self._values.popleft()
        self._values.append(value)
        self._total += value

    @property
    def value(self) -> float:
        if not self._values:
            return self._initial_value
        return self._total / len(self._values)

    def __len__(self) -> int:
        return len(self._values)


class MemoryfulnessEstimator:
    """Online estimator for eta = -Corr(k, N-k)."""

    def __init__(self, initial_value: float = 1.0) -> None:
        self._initial_value = initial_value
        self._count = 0
        self._sum_x = 0.0
        self._sum_y = 0.0
        self._sum_x2 = 0.0
        self._sum_y2 = 0.0
        self._sum_xy = 0.0

    def observe_program(self, num_turns: int) -> None:
        if num_turns <= 0:
            return
        for completed_turns in range(1, num_turns + 1):
            self._observe(completed_turns, num_turns - completed_turns)

    def _observe(self, completed_turns: int, remaining_turns: int) -> None:
        x = float(completed_turns)
        y = float(remaining_turns)
        self._count += 1
        self._sum_x += x
        self._sum_y += y
        self._sum_x2 += x * x
        self._sum_y2 += y * y
        self._sum_xy += x * y

    @property
    def value(self) -> float:
        if self._count < 2:
            return self._initial_value
        covariance = self._count * self._sum_xy - self._sum_x * self._sum_y
        variance_x = self._count * self._sum_x2 - self._sum_x**2
        variance_y = self._count * self._sum_y2 - self._sum_y**2
        denominator = sqrt(max(0.0, variance_x * variance_y))
        if denominator == 0:
            return self._initial_value
        eta = -covariance / denominator
        return min(1.0, max(-1.0, eta))

    @property
    def sample_count(self) -> int:
        return self._count


class ToolDurationHistory:
    """Bounded global and per-tool execution-time histories."""

    def __init__(self, max_samples: int) -> None:
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")
        self._global: deque[float] = deque(maxlen=max_samples)
        self._by_tool: defaultdict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=max_samples)
        )

    def observe(self, tool_name: str, duration: float) -> None:
        duration = max(0.0, duration)
        self._global.append(duration)
        self._by_tool[tool_name].append(duration)

    @property
    def global_count(self) -> int:
        return len(self._global)

    def tool_count(self, tool_name: str) -> int:
        return len(self._by_tool.get(tool_name, ()))

    def global_samples(self) -> tuple[float, ...]:
        return tuple(self._global)

    def tool_samples(self, tool_name: str) -> tuple[float, ...]:
        return tuple(self._by_tool.get(tool_name, ()))

    @staticmethod
    def finish_probability(samples: Sequence[float], ttl_seconds: float) -> float:
        if not samples:
            return 0.0
        ordered = sorted(samples)
        return bisect_right(ordered, ttl_seconds) / len(ordered)


@dataclass(frozen=True, slots=True)
class PaperTTLConfig:
    """Public parameters and explicit fallbacks for the paper policy."""

    history_threshold: int = 100
    cold_start_ttl_seconds: float = DEFAULT_TTL_SECONDS
    max_history_samples: int = 4096
    queue_delay_window_size: int = 100

    def __post_init__(self) -> None:
        if self.history_threshold < 0:
            raise ValueError("history_threshold must be non-negative")
        if self.cold_start_ttl_seconds < 0:
            raise ValueError("cold_start_ttl_seconds must be non-negative")
        if self.max_history_samples <= self.history_threshold:
            raise ValueError("max_history_samples must exceed history_threshold")
        if self.queue_delay_window_size <= 0:
            raise ValueError("queue_delay_window_size must be positive")


class BashToolCallParser:
    """Extract the command name from one fenced bash block."""

    _ACTION_PATTERN = re.compile(r"```bash\s*\n(.*?)\n```", re.DOTALL)

    def parse(self, text: str) -> str | None:
        actions = self._ACTION_PATTERN.findall(text)
        if len(actions) != 1:
            return None
        words = actions[0].strip().split()
        return words[0] if words else None


class PaperTTLPolicy:
    """Continuum utility model from paper Sections 4.1 and 4.2."""

    def __init__(
        self,
        config: PaperTTLConfig | None = None,
        reconstruction_cost: ReconstructionCostEstimator | None = None,
        parser: BashToolCallParser | None = None,
    ) -> None:
        self.config = config or PaperTTLConfig()
        self.reconstruction_cost = (
            reconstruction_cost or ConstantReconstructionCostEstimator()
        )
        self.parser = parser or BashToolCallParser()
        self.tool_history = ToolDurationHistory(self.config.max_history_samples)
        self.queue_delay = SlidingWindowMean(
            self.config.queue_delay_window_size, initial_value=0.0
        )
        self.memoryfulness = MemoryfulnessEstimator(initial_value=1.0)
        self._pending_tools: dict[str, tuple[float, str | None]] = {}

    def on_request_arrival(
        self, metadata: ContinuumRequestMetadata, now: float
    ) -> None:
        if metadata.job_id is None:
            return
        previous = self._pending_tools.pop(metadata.job_id, None)
        if previous is None:
            return
        finished_at, parsed_tool = previous
        tool_name = metadata.last_func_call or parsed_tool
        if tool_name is not None:
            self.tool_history.observe(tool_name, max(0.0, now - finished_at))

    def on_request_finished(
        self,
        metadata: ContinuumRequestMetadata,
        output_text: str | None,
        now: float,
        num_context_tokens: int = 0,
    ) -> TTLDecision | None:
        if metadata.job_id is None:
            return None

        tool_name = metadata.this_func_call
        if tool_name is None and output_text:
            tool_name = self.parser.parse(output_text)
        if metadata.is_last_step is True:
            self._pending_tools.pop(metadata.job_id, None)
            return None

        self._pending_tools[metadata.job_id] = (now, tool_name)
        if tool_name is None:
            return None

        ttl_seconds, source, probability, utility = self._select_ttl(
            tool_name, num_context_tokens
        )
        return TTLDecision(
            expires_at=now + ttl_seconds,
            tool_name=tool_name,
            ttl_seconds=ttl_seconds,
            source=source,
            finish_probability=probability,
            expected_utility=utility,
        )

    def _select_ttl(
        self, tool_name: str, num_context_tokens: int
    ) -> tuple[float, TTLSource, float | None, float | None]:
        if self.tool_history.global_count <= self.config.history_threshold:
            return (
                self.config.cold_start_ttl_seconds,
                "cold_start",
                None,
                None,
            )

        if self.tool_history.tool_count(tool_name) <= self.config.history_threshold:
            source: TTLSource = "global"
            samples = self.tool_history.global_samples()
        else:
            source = "tool"
            samples = self.tool_history.tool_samples(tool_name)

        benefit_seconds = (
            self.queue_delay.value * self.memoryfulness.value
            + self.reconstruction_cost.estimate_seconds(num_context_tokens)
        )
        ttl_seconds, probability, utility = self._maximize_utility(
            samples, benefit_seconds
        )
        return ttl_seconds, source, probability, utility

    @staticmethod
    def _maximize_utility(
        samples: Sequence[float], benefit_seconds: float
    ) -> tuple[float, float, float]:
        ordered = sorted(max(0.0, sample) for sample in samples)
        if not ordered:
            return 0.0, 0.0, 0.0

        candidates = sorted({0.0, *ordered})
        best_ttl = 0.0
        best_probability = bisect_right(ordered, 0.0) / len(ordered)
        best_utility = best_probability * benefit_seconds
        for ttl_seconds in candidates[1:]:
            probability = bisect_right(ordered, ttl_seconds) / len(ordered)
            utility = probability * benefit_seconds - ttl_seconds
            if utility > best_utility:
                best_ttl = ttl_seconds
                best_probability = probability
                best_utility = utility
        return best_ttl, best_probability, best_utility

    def observe_queue_delay(self, queue_delay: float) -> None:
        self.queue_delay.observe(queue_delay)

    def observe_program(self, num_turns: int) -> None:
        self.memoryfulness.observe_program(num_turns)


def paper_ttl_policy_from_env(
    environ: Mapping[str, str] | None = None,
) -> PaperTTLPolicy:
    """Build the runtime policy from `CONTINUUM_*` environment variables."""
    values = process_environ if environ is None else environ
    config = PaperTTLConfig(
        history_threshold=_env_int(values, "CONTINUUM_HISTORY_THRESHOLD", 100),
        cold_start_ttl_seconds=_env_float(
            values, "CONTINUUM_COLD_START_TTL_SECONDS", DEFAULT_TTL_SECONDS
        ),
        max_history_samples=_env_int(values, "CONTINUUM_MAX_HISTORY_SAMPLES", 4096),
        queue_delay_window_size=_env_int(
            values, "CONTINUUM_QUEUE_DELAY_WINDOW_SIZE", 100
        ),
    )

    coefficient_names = (
        "CONTINUUM_PREFILL_QUADRATIC",
        "CONTINUUM_PREFILL_LINEAR",
        "CONTINUUM_PREFILL_CONSTANT",
    )
    coefficients = [values.get(name) for name in coefficient_names]
    if any(value is not None for value in coefficients):
        if not all(value is not None for value in coefficients):
            raise ValueError(
                "quadratic prefill profile requires quadratic, linear, and "
                "constant coefficients"
            )
        reconstruction_cost: ReconstructionCostEstimator = (
            QuadraticReconstructionCostEstimator(
                *(
                    _parse_float(name, value)
                    for name, value in zip(coefficient_names, coefficients, strict=True)
                )
            )
        )
    else:
        reconstruction_cost = ConstantReconstructionCostEstimator(
            _env_float(
                values,
                "CONTINUUM_RECONSTRUCTION_SECONDS",
                DEFAULT_TTL_SECONDS,
            )
        )
    return PaperTTLPolicy(config=config, reconstruction_cost=reconstruction_cost)


def _env_int(values: Mapping[str, str], name: str, default: int) -> int:
    raw = values.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, found {raw!r}") from exc


def _env_float(values: Mapping[str, str], name: str, default: float) -> float:
    raw = values.get(name)
    return default if raw is None else _parse_float(name, raw)


def _parse_float(name: str, value: str | None) -> float:
    assert value is not None
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, found {value!r}") from exc
