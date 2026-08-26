"""Render a Continuum stats dump as a readable report.

Reads the JSON written by CONTINUUM_STATS_DUMP_PATH and explains what the
collected data means for the TTL model. Standard library only.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Any

RULE = "-" * 72


def _seconds(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value < 1e-3:
        return f"{value * 1e6:.1f} us"
    if value < 1.0:
        return f"{value * 1e3:.2f} ms"
    return f"{value:.3f} s"


def _load(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        document = json.load(handle)
    schema = document.get("schema", "")
    if not schema.startswith("continuum-telemetry/"):
        raise ValueError(f"{path} is not a Continuum stats dump (schema={schema!r})")
    return document


def _print_header(document: dict[str, Any]) -> None:
    counters = document.get("counters", {})
    print(RULE)
    print("Continuum stats report")
    print(RULE)
    print(f"  uptime           {document.get('uptime_seconds', 0.0):.1f} s")
    print(f"  TTL decisions    {counters.get('decisions', 0)}")
    print(f"  pins created     {counters.get('pins', 0)}")


def _print_reload(document: dict[str, Any]) -> dict[str, Any]:
    reload_section = document.get("reload", {})
    mode = reload_section.get("mode", "unknown")
    print()
    print("Reload estimator (online, from the KV connector)")
    print(f"  mode             {mode}  [{reload_section.get('detail', '')}]")
    print(
        f"  samples in fit   {reload_section.get('sample_count', 0)}"
        "   (bounded window used by the estimator)"
    )
    print(
        f"  samples total    {reload_section.get('observed_total', 0)}"
        "   (cumulative since engine start)"
    )
    print(f"  warm             {reload_section.get('is_warm', False)}")
    if "microseconds_per_token" in reload_section:
        print(
            f"  fitted curve     {reload_section['microseconds_per_token']:.3f} "
            f"us/token + {_seconds(reload_section.get('intercept_seconds'))} base"
        )
    if mode == "disabled":
        print("  -> no KV tier attached; CacheMissCost is the prefill profile.")
    elif mode == "prefill_fallback":
        print("  -> tier loads synchronously; reload is not measurable here.")
    elif not reload_section.get("is_warm"):
        print("  -> still cold; CacheMissCost falls back to the prefill profile.")
    return reload_section


def _print_comparison(document: dict[str, Any], reload_section: dict[str, Any]) -> None:
    prefill = document.get("prefill", {})
    print()
    print(f"Prefill profile ({prefill.get('estimator', 'unknown')})")
    coefficients = prefill.get("coefficients")
    if coefficients:
        print(
            "  coefficients     "
            + "  ".join(f"{key}={value:.6g}" for key, value in coefficients.items())
        )

    prefill_estimates = prefill.get("estimates_seconds", {})
    reload_estimates = reload_section.get("estimates_seconds", {})
    if not prefill_estimates:
        return
    print()
    print("CacheMissCost by context length")
    print(f"  {'tokens':>8}  {'prefill':>12}  {'reload':>12}  {'ratio':>8}")
    for key in sorted(prefill_estimates, key=int):
        prefill_value = prefill_estimates[key]
        reload_value = reload_estimates.get(key)
        if reload_value is None:
            ratio = "-"
        else:
            ratio = f"{prefill_value / reload_value:.1f}x" if reload_value else "inf"
        print(
            f"  {int(key):>8}  {_seconds(prefill_value):>12}  "
            f"{_seconds(reload_value):>12}  {ratio:>8}"
        )
    if reload_estimates and reload_section.get("is_warm"):
        print()
        print(
            "  The 'ratio' column is how much Continuum would OVERESTIMATE the\n"
            "  benefit of pinning if it used the prefill profile while this tier\n"
            "  is attached. Higher ratio means the tier absorbs more of the cost,\n"
            "  so shorter TTLs are correct."
        )


def _print_sources(document: dict[str, Any]) -> None:
    ttl_sources = document.get("ttl_sources", {})
    recon_sources = document.get("reconstruction_sources", {})
    print()
    print("Decision sources")
    print(f"  TTL              {ttl_sources or '{}'}")
    print(f"  CacheMissCost    {recon_sources or '{}'}")
    if ttl_sources.get("cold_start"):
        print(
            f"  -> {ttl_sources['cold_start']} decisions still used the fixed "
            "cold-start TTL; the tool history has not passed the threshold."
        )
    if recon_sources.get("prefill_fallback"):
        print(
            f"  -> {recon_sources['prefill_fallback']} decisions had a tier "
            "attached but no usable reload estimate."
        )


def _print_counters(document: dict[str, Any]) -> None:
    counters = document.get("counters", {})
    pins = counters.get("pins", 0)
    print()
    print("Pin outcomes")
    for label, key in (
        ("ttl expired", "ttl_expirations"),
        ("handed off", "handoffs"),
        ("pressure", "pressure_unpins"),
        ("final", "final_releases"),
    ):
        value = counters.get(key, 0)
        share = f"{100.0 * value / pins:5.1f}%" if pins else "    -"
        print(f"  {label:<16} {value:>8}  {share}")
    if pins and counters.get("handoffs", 0) / pins < 0.2:
        print(
            "  -> fewer than 20% of pins were reclaimed by the next turn. TTLs\n"
            "     may be too short, or job_id is not stable across turns."
        )
    if pins and counters.get("pressure_unpins", 0) / pins > 0.3:
        print(
            "  -> more than 30% of pins were dropped under memory pressure.\n"
            "     TTLs are too long for the available KV cache."
        )


def _print_tools(document: dict[str, Any]) -> None:
    tools = document.get("tools", {})
    by_tool = tools.get("by_tool", {})
    print()
    print(
        f"Tool execution history (global {tools.get('global_count', 0)}, "
        f"threshold {tools.get('threshold', 0)})"
    )
    if not by_tool:
        print("  none recorded")
        return
    print(f"  {'tool':<20} {'n':>6} {'p50':>10} {'p90':>10} {'p99':>10}")
    for name, stats in sorted(by_tool.items(), key=lambda item: -item[1]["count"])[:20]:
        print(
            f"  {name[:20]:<20} {stats['count']:>6} "
            f"{_seconds(stats['p50_seconds']):>10} "
            f"{_seconds(stats['p90_seconds']):>10} "
            f"{_seconds(stats['p99_seconds']):>10}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="continuum-vllm-report",
        description="Explain a Continuum stats dump.",
    )
    parser.add_argument("path", help="JSON file written by CONTINUUM_STATS_DUMP_PATH")
    parser.add_argument(
        "--raw", action="store_true", help="print the parsed document instead"
    )
    arguments = parser.parse_args(argv)

    try:
        document = _load(arguments.path)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if arguments.raw:
        json.dump(document, sys.stdout, indent=2, sort_keys=True)
        print()
        return 0

    _print_header(document)
    reload_section = _print_reload(document)
    _print_comparison(document, reload_section)
    _print_sources(document)
    _print_counters(document)
    _print_tools(document)
    print()
    print(RULE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
