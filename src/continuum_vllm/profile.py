"""Offline prefill sweep for the Continuum CacheMissCost model.

Measures time-to-first-token against context length on a plain vLLM server
and fits the paper's quadratic profile. Uses only the standard library so it
runs inside an air-gapped image.

The reload half of CacheMissCost is NOT measured here. It is observed online
by the scheduler plugin from the KV connector's asynchronous load window.
"""

import argparse
import json
import random
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from typing import Any

DEFAULT_TOKEN_ID_MIN = 100
DEFAULT_TOKEN_ID_MAX = 5000
PREFIX_CACHE_SPEEDUP = 2.0


class ProfileError(RuntimeError):
    """Raised when the sweep cannot produce a trustworthy profile."""


def solve_3x3(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting."""
    rows = [row[:] + [vector[index]] for index, row in enumerate(matrix)]
    for column in range(3):
        pivot = max(range(column, 3), key=lambda r: abs(rows[r][column]))
        if abs(rows[pivot][column]) < 1e-18:
            raise ProfileError(
                "the sweep is degenerate; use more distinct context lengths"
            )
        rows[column], rows[pivot] = rows[pivot], rows[column]
        for target in range(column + 1, 3):
            factor = rows[target][column] / rows[column][column]
            for cell in range(column, 4):
                rows[target][cell] -= factor * rows[column][cell]
    solution = [0.0, 0.0, 0.0]
    for column in reversed(range(3)):
        total = rows[column][3] - sum(
            rows[column][c] * solution[c] for c in range(column + 1, 3)
        )
        solution[column] = total / rows[column][column]
    return solution


def fit_quadratic(points: Sequence[tuple[int, float]]) -> dict[str, float]:
    """Least-squares fit of seconds = a*n^2 + b*n + c, scaled for conditioning."""
    if len({tokens for tokens, _ in points}) < 3:
        raise ProfileError("need at least three distinct context lengths")
    scale = float(max(tokens for tokens, _ in points))
    sums = [0.0] * 5
    targets = [0.0] * 3
    for tokens, seconds in points:
        x = tokens / scale
        powers = [1.0, x, x * x, x**3, x**4]
        for index in range(5):
            sums[index] += powers[index]
        for index in range(3):
            targets[index] += seconds * powers[index]
    matrix = [
        [sums[4], sums[3], sums[2]],
        [sums[3], sums[2], sums[1]],
        [sums[2], sums[1], sums[0]],
    ]
    vector = [targets[2], targets[1], targets[0]]
    scaled_a, scaled_b, constant = solve_3x3(matrix, vector)
    return {
        "quadratic": scaled_a / (scale * scale),
        "linear": scaled_b / scale,
        "constant": constant,
    }


def r_squared(
    points: Sequence[tuple[int, float]], coefficients: dict[str, float]
) -> float:
    observed = [seconds for _, seconds in points]
    mean = statistics.fmean(observed)
    total = sum((value - mean) ** 2 for value in observed)
    if total == 0:
        return 1.0
    residual = sum(
        (seconds - evaluate(coefficients, tokens)) ** 2 for tokens, seconds in points
    )
    return 1.0 - residual / total


def evaluate(coefficients: dict[str, float], num_tokens: int) -> float:
    return (
        coefficients["quadratic"] * num_tokens**2
        + coefficients["linear"] * num_tokens
        + coefficients["constant"]
    )


def post_completion(
    base_url: str,
    model: str,
    token_ids: list[int],
    timeout: float,
    api_key: str | None,
) -> float:
    payload = json.dumps(
        {
            "model": model,
            "prompt": token_ids,
            "max_tokens": 1,
            "temperature": 0.0,
            "stream": False,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        response.read()
    return time.perf_counter() - started


def random_tokens(count: int, rng: random.Random, low: int, high: int) -> list[int]:
    return [rng.randrange(low, high) for _ in range(count)]


def check_prefix_cache_disabled(
    base_url: str,
    model: str,
    rng: random.Random,
    timeout: float,
    api_key: str | None,
    low: int,
    high: int,
) -> None:
    """Send one prompt twice; a large speedup means prefix caching is on."""
    token_ids = random_tokens(4096, rng, low, high)
    first = post_completion(base_url, model, token_ids, timeout, api_key)
    second = post_completion(base_url, model, token_ids, timeout, api_key)
    if second * PREFIX_CACHE_SPEEDUP < first:
        raise ProfileError(
            f"repeating the same prompt was {first / max(second, 1e-9):.1f}x "
            "faster, so prefix caching is serving the second request. Restart "
            "the server with --no-enable-prefix-caching before profiling."
        )


def sweep(arguments: argparse.Namespace) -> dict[str, Any]:
    rng = random.Random(arguments.seed)
    lengths = list(range(arguments.step, arguments.max_len + 1, arguments.step))
    if len(lengths) < 3:
        raise ProfileError("--max-len / --step must yield at least three lengths")

    if arguments.check_prefix_cache:
        check_prefix_cache_disabled(
            arguments.base_url,
            arguments.model,
            rng,
            arguments.timeout,
            arguments.api_key,
            arguments.token_id_min,
            arguments.token_id_max,
        )

    for _ in range(arguments.warmup):
        post_completion(
            arguments.base_url,
            arguments.model,
            random_tokens(
                lengths[0], rng, arguments.token_id_min, arguments.token_id_max
            ),
            arguments.timeout,
            arguments.api_key,
        )

    measurements: list[dict[str, Any]] = []
    points: list[tuple[int, float]] = []
    for num_tokens in lengths:
        durations = []
        for _ in range(arguments.repeat):
            token_ids = random_tokens(
                num_tokens, rng, arguments.token_id_min, arguments.token_id_max
            )
            durations.append(
                post_completion(
                    arguments.base_url,
                    arguments.model,
                    token_ids,
                    arguments.timeout,
                    arguments.api_key,
                )
            )
        median = statistics.median(durations)
        measurements.append(
            {
                "num_tokens": num_tokens,
                "median_seconds": median,
                "min_seconds": min(durations),
                "max_seconds": max(durations),
                "samples": durations,
            }
        )
        points.append((num_tokens, median))
        print(
            f"  {num_tokens:>7} tokens  median {median * 1000:8.1f} ms  "
            f"(min {min(durations) * 1000:7.1f}  max {max(durations) * 1000:7.1f})",
            file=sys.stderr,
            flush=True,
        )

    coefficients = fit_quadratic(points)
    return {
        "schema": "continuum-prefill-profile/1",
        "generated_at": time.time(),
        "model": arguments.model,
        "base_url": arguments.base_url,
        "repeat": arguments.repeat,
        "note": (
            "seconds = quadratic*n^2 + linear*n + constant; measured as "
            "end-to-end latency of a max_tokens=1 completion, so one decode "
            "step is absorbed into the constant term"
        ),
        "prefill_coefficients": coefficients,
        "r_squared": r_squared(points, coefficients),
        "measurements": measurements,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="continuum-vllm-profile",
        description=(
            "Measure the prefill half of Continuum's CacheMissCost against a "
            "plain vLLM server started WITHOUT a KV connector and WITHOUT "
            "prefix caching."
        ),
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-len", type=int, default=32768)
    parser.add_argument("--step", type=int, default=2048)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--token-id-min", type=int, default=DEFAULT_TOKEN_ID_MIN)
    parser.add_argument("--token-id-max", type=int, default=DEFAULT_TOKEN_ID_MAX)
    parser.add_argument(
        "--no-check-prefix-cache",
        dest="check_prefix_cache",
        action="store_false",
        help="skip the guard that detects an enabled prefix cache",
    )
    parser.add_argument("--out", default="continuum-prefill-profile.json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    print(
        f"Sweeping {arguments.model} at {arguments.base_url} "
        f"(step={arguments.step}, max_len={arguments.max_len}, "
        f"repeat={arguments.repeat}), concurrency 1.",
        file=sys.stderr,
    )
    try:
        document = sweep(arguments)
    except ProfileError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except urllib.error.URLError as exc:
        print(f"error: cannot reach {arguments.base_url}: {exc}", file=sys.stderr)
        return 2

    with open(arguments.out, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)

    coefficients = document["prefill_coefficients"]
    print(f"\nWrote {arguments.out}  (R^2 = {document['r_squared']:.4f})")
    print("\nPoint the plugin at it:")
    print(f"  export CONTINUUM_PREFILL_PROFILE={arguments.out}")
    print("\nOr pass the coefficients directly:")
    print(f"  export CONTINUUM_PREFILL_QUADRATIC={coefficients['quadratic']:.12g}")
    print(f"  export CONTINUUM_PREFILL_LINEAR={coefficients['linear']:.12g}")
    print(f"  export CONTINUUM_PREFILL_CONSTANT={coefficients['constant']:.12g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
