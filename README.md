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

Validate the installed vLLM version and both custom Scheduler classes without
loading a model:

```bash
continuum-vllm-check
```

This command verifies installation and compatibility; it does not prove that a
running server selected the plugin.

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
Continuum scheduler active: class=AsyncContinuumScheduler ttl_policy=PaperTTLPolicy allocate_slots_wrapper=enabled
```

For an end-to-end TTL check, start once with `VLLM_LOGGING_LEVEL=DEBUG`, send a
non-final request with `job_id` and `this_func_call`, and check for:

```text
Continuum pinned job=agent-001 tool=pytest ttl=2.000 source=cold_start
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

## TTL configuration

The defaults preserve the public prototype's two-second cold-start behavior.
Configuration is read once when the Scheduler starts.

| Environment variable | Default | Meaning |
| --- | ---: | --- |
| `CONTINUUM_HISTORY_THRESHOLD` | `100` | Samples required before empirical TTL |
| `CONTINUUM_COLD_START_TTL_SECONDS` | `2.0` | TTL before enough samples exist |
| `CONTINUUM_MAX_HISTORY_SAMPLES` | `4096` | Global and per-tool history bound |
| `CONTINUUM_QUEUE_DELAY_WINDOW_SIZE` | `100` | Sliding window used for queue delay T |
| `CONTINUUM_RECONSTRUCTION_SECONDS` | `2.0` | Constant Prefill-Reload fallback |

For a measured quadratic prefill profile, set all three coefficients. The
constant fallback is then ignored.

```bash
export CONTINUUM_PREFILL_QUADRATIC=0.000001
export CONTINUUM_PREFILL_LINEAR=0.0001
export CONTINUUM_PREFILL_CONSTANT=0.01
```

The estimator evaluates
`quadratic * context_tokens^2 + linear * context_tokens + constant` in seconds.
