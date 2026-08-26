# Continuum TTL plugin for vLLM 0.25.1

This package adds Continuum's paper TTL policy to an existing vLLM 0.25.1
installation. It does not replace files under `site-packages/vllm` and does not
compile native code.

## What it changes

- Loads a custom scheduler through vLLM's `--scheduler-cls` option.
- Reads agent metadata from vLLM's native `vllm_xargs` request field.
- Keeps completed request KV blocks referenced until TTL expiry, handoff to the
  next turn, or memory pressure.
- Prioritizes preempted requests, then requests with a claimed pin, then
  program-level FCFS.
- Computes TTL using the empirical utility model described in the Continuum
  paper.
- Estimates the paper's `Prefill-Reload(r)` term from two separate curves: an
  offline prefill profile for recompute, and an online linear fit for tier
  reload, selected by whether a KV connector is attached.

The plugin targets vLLM `0.25.1`, including local wheel versions such as
`0.25.1+empty`. It fails fast on another vLLM release because Scheduler is an
internal API.

## Origin and license

This project is a standalone adaptation of
[`Hanchenli/vllm-continuum`](https://github.com/Hanchenli/vllm-continuum) for
vLLM 0.25.1. The integration was rewritten as an installable scheduler plugin
instead of distributing a modified vLLM source tree. It remains under the
Apache License 2.0; see `LICENSE`.

## Offline editable install

The target image must already contain vLLM 0.25.1 and its matching device
plugin. It must also contain `setuptools`, because build isolation is disabled
to prevent network access. Check the environment first:

```bash
python -m pip show vllm setuptools
```

Transfer this repository into the offline image, then install it from the
repository root:

```bash
python -m pip install --no-index --no-deps --no-build-isolation -e .
```

No dependency download or vLLM build is performed.

Validate the installed vLLM version, both custom Scheduler classes, and the
resolved cost model without loading a model:

```bash
continuum-vllm-check
```

This command verifies installation and compatibility; it does not prove that a
running server selected the plugin.

The package installs three commands:

| Command | Purpose |
| --- | --- |
| `continuum-vllm-check` | Verify the install and print the resolved cost model |
| `continuum-vllm-profile` | Measure the offline prefill curve against a plain server |
| `continuum-vllm-report` | Explain a runtime stats dump |

See [OPERATIONS.md](OPERATIONS.md) for the full deploy, collect, and read
workflow, including which parameters matter for each deployment shape.

## Start vLLM

Use the async scheduler when vLLM async scheduling is enabled:

```bash
vllm serve MODEL \
  --scheduler-cls continuum_vllm.scheduler.AsyncContinuumScheduler \
  --enable-prefix-caching
```

The engine process prints the following line when the custom Scheduler has
actually initialized:

```text
Continuum scheduler active: class=AsyncContinuumScheduler ttl_policy=PaperTTLPolicy allocate_slots_wrapper=enabled reload_estimator=online (OffloadingConnector)
```

The `reload_estimator` field reports how `CacheMissCost` is being estimated:
`online` when an asynchronous KV connector is attached and reload latency is
being measured, `prefill_fallback` when a tier exists but loads synchronously,
and `disabled` when no connector is configured.

For an end-to-end TTL check, start once with `VLLM_LOGGING_LEVEL=DEBUG`, send a
non-final request with `job_id` and `this_func_call`, and check for:

```text
Continuum pinned job=agent-001 tool=pytest ttl=2.000 ttl_source=cold_start recon=0.0623s recon_source=prefill tokens=1024
```

For a deployment that explicitly disables async scheduling:

```bash
vllm serve MODEL \
  --no-async-scheduling \
  --scheduler-cls continuum_vllm.scheduler.ContinuumScheduler \
  --enable-prefix-caching
```

## Request metadata

Send stable `job_id` values for all turns of one agent program. Boolean values
are encoded as `0` or `1` because `vllm_xargs` accepts scalar strings and
numbers.

```json
{
  "model": "MODEL",
  "messages": [{"role": "user", "content": "Run the tests"}],
  "vllm_xargs": {
    "job_id": "agent-001",
    "this_func_call": "pytest",
    "is_last_step": 0
  }
}
```

On the next turn, reuse `job_id` and pass the completed tool name:

```json
{
  "vllm_xargs": {
    "job_id": "agent-001",
    "last_func_call": "pytest",
    "is_last_step": 1
  }
}
```

If `this_func_call` is omitted, the default parser recognizes one fenced
`bash` block in the generated text. Explicit metadata is recommended for other
tool-call formats.

## Configuration

Configuration is read once when the Scheduler starts.

| Environment variable | Default | Meaning |
| --- | ---: | --- |
| `CONTINUUM_PREFILL_PROFILE` | — | Path to a `continuum-vllm-profile` result; highest priority |
| `CONTINUUM_PREFILL_QUADRATIC` | — | Manual quadratic coefficient; all three required |
| `CONTINUUM_PREFILL_LINEAR` | — | Manual linear coefficient |
| `CONTINUUM_PREFILL_CONSTANT` | — | Manual constant coefficient |
| `CONTINUUM_PREFILL_SECONDS` | `2.0` | Constant fallback when no profile is given |
| `CONTINUUM_RECONSTRUCTION_SECONDS` | — | Accepted legacy name for the line above |
| `CONTINUUM_RELOAD_MIN_SAMPLES` | `32` | Reload samples required before the online fit is used |
| `CONTINUUM_RELOAD_MAX_SAMPLES` | `1024` | Bound on the reload fitting window |
| `CONTINUUM_HISTORY_THRESHOLD` | `100` | Samples required before empirical TTL |
| `CONTINUUM_COLD_START_TTL_SECONDS` | `2.0` | TTL before enough samples exist |
| `CONTINUUM_MAX_HISTORY_SAMPLES` | `4096` | Global and per-tool history bound |
| `CONTINUUM_QUEUE_DELAY_WINDOW_SIZE` | `100` | Sliding window used for queue delay T |
| `CONTINUUM_PENDING_TOOL_MAX_ENTRIES` | `4096` | Bound on concurrently timed tool calls |
| `CONTINUUM_PENDING_TOOL_MAX_AGE_SECONDS` | `3600` | Tool-call timers older than this are dropped, not recorded |
| `CONTINUUM_STATS_INTERVAL_SECONDS` | `60` | Periodic stats log and dump refresh; `0` disables |
| `CONTINUUM_STATS_DUMP_PATH` | — | Stats dump path; the process id is appended |

### The two halves of CacheMissCost

The paper's `Prefill-Reload(r)` is the cost of getting a prefix back into
device memory. That is a recompute when no KV tier is attached and a transfer
when one is, so the plugin keeps them apart.

**Prefill** is a hardware and model property, independent of any offload tier.
Measure it once per `(model, device, parallelism, dtype)` against a server
started **without** a KV connector and **without** prefix caching:

```bash
continuum-vllm-profile --model MODEL --max-len 32768 --out prefill.json
export CONTINUUM_PREFILL_PROFILE=prefill.json
```

The estimator evaluates
`quadratic * context_tokens^2 + linear * context_tokens + constant` in seconds.

**Reload** is a transfer property and needs no offline run. The plugin times
the KV connector's asynchronous load window and fits a line against the number
of transferred tokens. It is warm after `CONTINUUM_RELOAD_MIN_SAMPLES` samples
and falls back to the prefill profile until then.

This only works for connectors that load asynchronously. `OffloadingConnector`,
`MooncakeStoreConnector`, `MooncakeConnector`, `NixlConnector`, and
`HF3FSKVConnector` do; `SimpleCPUOffloadConnector` does not, and setting
`load_async: false` in `kv_connector_extra_config` disables it for any
connector. The plugin detects both cases at startup, reports
`reload_estimator=prefill_fallback`, and warns again if a tier that claims to
load asynchronously produces no samples.

## Reading the collected data

```bash
continuum-vllm-report /path/to/stats.<pid>.json
```

The report shows cold-start progress, the fitted reload curve, the prefill
profile, both costs side by side across context lengths, the TTL and
CacheMissCost source histograms, pin outcomes, and per-tool execution
percentiles.

Every cold-start input has a fallback that is numerically indistinguishable
from a measured value, so the report prints sample counts next to each one:
the TTL stage, the queue delay T, the memoryfulness eta, and the reload fit.
Note that eta stays at exactly 1.0 until programs of at least two different
turn counts have finished, because within one program k and N-k are perfectly
anti-correlated. The dump is refreshed on the
`CONTINUUM_STATS_INTERVAL_SECONDS` cadence, so the engine does not need to be
stopped to read it.
