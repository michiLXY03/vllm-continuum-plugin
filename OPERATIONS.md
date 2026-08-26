# Continuum 插件运维手册（v0.2.0）

面向在内网环境部署、采集、判读 Continuum TTL 插件的人或 agent。全流程离线，
不需要外网，不需要 numpy/requests 等第三方库。

---

## 0. TL;DR

```
一次性：  裸 vLLM 起服务 → continuum-vllm-profile → 得到 prefill 曲线 JSON
每次部署：--scheduler-cls + CONTINUUM_PREFILL_PROFILE + CONTINUUM_STATS_DUMP_PATH
看数据：  continuum-vllm-report <dump.json>
```

**不需要为了采集单独起三套服务。** 只有 prefill 曲线需要一次离线采集，reload
曲线由插件在生产服务里自己在线拟合。

---

## 1. 为什么只采一条曲线

论文的 `CacheMissCost(r)` 里有一项 `Prefill-Reload(r)`，语义是「如果不 pin 住
这份 KV，下一轮把它弄回显存要花多少秒」。这个值分两种情形：

| 部署形态 | 真实代价 | 谁来提供 |
| --- | --- | --- |
| 裸 vLLM，无 KV connector | `prefill(n)`，重新算一遍 | **离线采集一次**，见阶段 A |
| 挂了 CPU offload / Mooncake / Nixl 等 | `reload(n)`，从外部层传回来 | **插件在线拟合**，零操作 |

原因：

- **prefill 是硬件+模型属性**，与有没有 offload 层无关。同一个
  `(模型 × 卡型 × TP/PP × dtype)` 采一次，三种部署共用。
- **reload 是传输属性**，代价 ≈ 搬运的字节数，所以用线性模型。而 vLLM 0.25.1
  的调度器本身就暴露了完整的测量窗口：

  ```
  scheduler.py:739   connector.get_num_new_matched_tokens(...) -> (tokens, load_async)
  scheduler.py:752   num_external_computed_tokens                 ← 样本的 x（token 数）
  scheduler.py:950   status = WAITING_FOR_REMOTE_KVS              ← 计时起点
  scheduler.py:2494  finished_recving_kv_req_ids.add(req_id)      ← 计时终点
  ```

  插件包裹 `get_num_new_matched_tokens` 起表，在 `_update_from_kv_xfer_finished`
  停表，直接得到 `(token 数, 秒)` 样本对。默认攒够 32 个样本就开始用。

### 前提：connector 必须走异步加载

只有 `load_async=True` 的路径才经过 `WAITING_FOR_REMOTE_KVS`。同步加载发生在
worker 的 forward 里，调度器看不见。v0.25.1 实测：

| Connector | 异步? | 说明 |
| --- | --- | --- |
| `OffloadingConnector`（官方 CPU/disk offload） | ✅ | `return num_hit_tokens, bool(num_hit_tokens)`，命中即异步 |
| `MooncakeStoreConnector` | ✅ | `load_async` 默认 `True` |
| `MooncakeConnector`（PD 分离） | ✅ | 恒为 `True` |
| `NixlConnector` pull/push | ✅ | 恒为 `True` |
| `HF3FSKVConnector` | ✅ | 命中即异步 |
| `SimpleCPUOffloadConnector` | ❌ | 恒为 `False`，纯同步 |

插件启动时会自动探测并在日志里说明走的是哪条路（见 4.1）。

---

## 2. 阶段 A：离线采集 prefill 曲线（一次性）

### 2.1 起一个专门用来采集的服务

三个参数是硬要求：

```bash
vllm serve /path/to/MODEL \
  --served-model-name profile-target \
  --no-enable-prefix-caching \
  --max-model-len 32768 \
  --port 8000
```

| 要求 | 为什么 |
| --- | --- |
| `--no-enable-prefix-caching` | 否则重复前缀命中缓存，测出来的不是 prefill |
| **不要**加 `--kv-transfer-config` | 挂了 connector 测的就不是纯重算 |
| **不要**加 `--scheduler-cls` | 采集时不需要 Continuum |
| `--max-model-len` ≥ 你要扫的最大长度 | 否则长样本直接被拒 |

TP/PP、dtype、量化方式必须和**生产部署一致**，否则曲线不能用。

### 2.2 跑扫描

```bash
continuum-vllm-profile \
  --base-url http://127.0.0.1:8000 \
  --model profile-target \
  --max-len 32768 \
  --step 2048 \
  --repeat 5 \
  --out /data/continuum/prefill-<模型>-<卡型>-tp<N>.json
```

行为说明：

- 串行发送，**并发恒为 1**（并发会把排队时间混进 TTFT）
- 每个长度发 `--repeat` 次随机 token 的 prompt，取中位数
- 用随机 token id（默认 `[100, 5000)`）保证前缀不重复
- `max_tokens=1`，所以结果里含一个 decode step（约 10–30 ms），这部分被吸收进
  拟合的常数项，对秒级的 benefit 可忽略
- **自带前缀缓存检测**：开跑前发两次相同 prompt，如果第二次快 2 倍以上直接报错
  退出，提示你关掉 prefix caching。要跳过用 `--no-check-prefix-cache`

输出示例：

```
     2048 tokens  median    118.3 ms  (min   115.2  max   124.7)
     4096 tokens  median    241.9 ms  (min   238.1  max   250.3)
     ...
    32768 tokens  median   3412.5 ms  (min  3389.0  max  3455.1)

Wrote /data/continuum/prefill-qwen32b-910b-tp4.json  (R^2 = 0.9994)

Point the plugin at it:
  export CONTINUUM_PREFILL_PROFILE=/data/continuum/prefill-qwen32b-910b-tp4.json
```

**验收标准：`R^2 ≥ 0.99`。** 低于 0.99 说明测量被干扰了，检查：

- 是不是有别的请求打到这台机器
- prefix caching 是不是真关了
- `--repeat` 调大到 9，或 `--step` 调小增加采样点

采完这台服务就可以停了。

---

## 3. 阶段 B：部署

### 3.1 安装（离线）

镜像里要先有 vLLM 0.25.1 和 `setuptools`：

```bash
python -m pip show vllm setuptools
python -m pip install --no-index --no-deps --no-build-isolation -e /path/to/continuum-vllm
continuum-vllm-check
```

`continuum-vllm-check` 会打印 vLLM 版本、两个 Scheduler 类、以及**当前环境变量
解析出来的 prefill 估计值**，用它确认 `CONTINUUM_PREFILL_PROFILE` 真的被读到了：

```
vLLM: 0.25.1
sync scheduler: continuum_vllm.scheduler.ContinuumScheduler
async scheduler: continuum_vllm.scheduler.AsyncContinuumScheduler
prefill estimator: QuadraticPrefillCostEstimator
  prefill(1000) = 0.0623s
  prefill(8000) = 0.4815s
  prefill(32000) = 3.3901s
reload estimator: online, warm after 32 samples
```

如果这里显示 `ConstantPrefillCostEstimator` 且值恒为 2.0，说明 profile **没读到**。

### 3.2 通用启动参数

```bash
export CONTINUUM_PREFILL_PROFILE=/data/continuum/prefill-qwen32b-910b-tp4.json
export CONTINUUM_STATS_DUMP_PATH=/data/continuum/stats.json
export CONTINUUM_STATS_INTERVAL_SECONDS=60

vllm serve /path/to/MODEL \
  --scheduler-cls continuum_vllm.scheduler.AsyncContinuumScheduler \
  --enable-prefix-caching \
  ...
```

| 参数 | 必须 | 说明 |
| --- | --- | --- |
| `--scheduler-cls ...AsyncContinuumScheduler` | ✅ | 开着 async scheduling 时用这个 |
| `--scheduler-cls ...ContinuumScheduler` + `--no-async-scheduling` | 二选一 | 显式关闭异步调度时用这个 |
| `--enable-prefix-caching` | ✅ | pin 住的块要靠前缀缓存在下一轮命中，关掉整个机制无效 |
| `CONTINUUM_PREFILL_PROFILE` | ✅ | 不设就退化成恒定 2.0s，长上下文会被严重低估 |
| `CONTINUUM_STATS_DUMP_PATH` | 建议 | 不设就只有日志，没有可分析的文件 |

> `CONTINUUM_STATS_DUMP_PATH` 会**自动追加进程号**：`stats.json` →
> `stats.<pid>.json`。这样 DP / 多引擎进程不会互相覆盖。看数据时用通配符匹配。

### 3.3 三种形态的差异

**a) 裸 vLLM（无 offload 层）**

不加 `--kv-transfer-config`。启动日志应出现 `reload_estimator=disabled`。
CacheMissCost 全程用 prefill 曲线，这是正确行为。

**b) CPU offload**

保持你现有的 offload 配置不变，Continuum 不需要你为它改任何 connector 参数。
唯一禁忌：

```jsonc
// 不要这样配
"kv_connector_extra_config": { "load_async": false }
```

设成 false 会关掉异步窗口，插件测不到 reload，日志会降级为
`reload_estimator=prefill_fallback`。

**c) Mooncake**

同上。`MooncakeStoreConnector` 的 `load_async` 默认就是 `True`，不用动。

### 3.4 与 vllm-ascend 共存

- ascend 的 4 个调度器扩展（`batch_job_sched` / `recompute_scheduler` /
  `profiling_chunk` / `short_request_first`）以及 `enable_balance_scheduling`
  **全部保持默认关闭**，否则会和 `--scheduler-cls` 抢同一个槽位。
- ascend 有一个无条件的 monkey patch
  （`patch_balance_schedule.py:887` 的 `_sched_mod.Scheduler = BalanceScheduler`），
  会让 `ContinuumScheduler` 的实际基类变成 `BalanceScheduler`。
  `enable_balance_scheduling` 关闭时它的 `schedule()` 第一行就委托回真身，功能无害，
  但**启动时务必确认一次 MRO**，见 4.1。
- 版本配对：vLLM v0.25.1 → ascend `releases/v0.25.1rc`（不是 `main`，`main` 对的是
  v0.26.0）。

---

## 4. 阶段 C：怎么看数据

### 4.1 启动日志（确认插件真的生效）

grep `Continuum scheduler active`：

```
Continuum scheduler active: class=AsyncContinuumScheduler ttl_policy=PaperTTLPolicy
  allocate_slots_wrapper=enabled reload_estimator=online (OffloadingConnector)
```

`reload_estimator=` 的三种取值：

| 值 | 含义 | 要做什么 |
| --- | --- | --- |
| `online` | 挂了异步 connector，正在在线采 reload | 正常 |
| `prefill_fallback` | 有外部层但测不到（同步加载 / `load_async=false`） | 见 6.3 |
| `disabled` | 没挂 connector，纯 prefill | 正常（裸 vLLM 就该这样） |

**没有这行 = 插件根本没被加载**，检查 `--scheduler-cls` 拼写和 `pip install`。

顺便确认基类（和 ascend 共存时必查）：

```bash
python -c "
from continuum_vllm.scheduler import ContinuumScheduler
print([c.__name__ for c in ContinuumScheduler.__mro__])
"
```

### 4.2 周期日志（默认 60 秒一行）

```
Continuum stats: mode=online decisions=400 pins=259 ttl_expired=72 handoff=155
  pressure=32 reload_window=120 reload_total=418 reload_warm=True
  reload_us_per_token=1.101 reload_base_ms=4.00 ttl_src={'tool': 400}
  recon_src={'reload': 400} queue_delay_s=3.200 eta=0.811
```

| 字段 | 含义 |
| --- | --- |
| `decisions` | 做了多少次 TTL 决策 |
| `pins` | 其中多少次真的 pin 住了 KV |
| `ttl_expired` / `handoff` / `pressure` | pin 的三种结局 |
| `reload_window` | 当前拟合窗口里的样本数（有上限） |
| `reload_total` | 引擎启动至今累计观测到的样本数 |
| `reload_warm` | 样本是否够用（`false` 时退回 prefill 曲线） |
| `reload_us_per_token` | 拟合出的斜率，每 token 多少微秒 |
| `recon_src` | CacheMissCost 分别用了哪条曲线 |
| `eta` | 记忆性系数 η |

### 4.3 完整报告

dump 文件每 60 秒刷新一次，**不需要停服**。

```bash
continuum-vllm-report /data/continuum/stats.12345.json
```

输出：

```
------------------------------------------------------------------------
Continuum stats report
------------------------------------------------------------------------
  uptime           1843.0 s
  TTL decisions    400
  pins created     259

Reload estimator (online, from the KV connector)
  mode             online  [OffloadingConnector]
  samples in fit   120   (bounded window used by the estimator)
  samples total    418   (cumulative since engine start)
  warm             True
  fitted curve     1.101 us/token + 4.00 ms base

Prefill profile (QuadraticPrefillCostEstimator)
  coefficients     constant=0.015  linear=3e-05  quadratic=2.4e-09

CacheMissCost by context length
    tokens       prefill        reload     ratio
      1000      47.40 ms       5.10 ms      9.3x
      4000     173.40 ms       8.41 ms     20.6x
     16000       1.109 s      21.62 ms     51.3x
     32000       3.433 s      39.24 ms     87.5x
     65536      12.289 s      76.17 ms    161.3x

Decision sources
  TTL              {'tool': 400}
  CacheMissCost    {'reload': 400}

Pin outcomes
  ttl expired            72   27.8%
  handed off            155   59.8%
  pressure               32   12.4%
  final                   0    0.0%

Tool execution history (global 180, threshold 5)
  tool                      n        p50        p90        p99
  grep                     60  359.77 ms  593.12 ms  690.72 ms
  ls                       60   51.95 ms   82.91 ms   99.39 ms
  pytest                   60    4.287 s    5.968 s    6.887 s
------------------------------------------------------------------------
```

`--raw` 可以直接输出原始 JSON 给别的工具处理。

---

## 5. 判读速查

### CacheMissCost 那张表的 `ratio` 列

这一列是「如果继续用 prefill 曲线，会把 pin 的收益高估多少倍」。上面的例子里
32k 上下文是 **87.5 倍** —— 这正是 v0.1.0 的缺陷，也是这次改动要修的东西。
ratio 越大说明外部层吸收得越多，**更短的 TTL 才是对的**。

### Pin outcomes 怎么读

| 现象 | 含义 | 处理 |
| --- | --- | --- |
| `handed off` < 20% | pin 很少被下一轮取走 | TTL 太短，或 `job_id` 跨轮次不稳定 |
| `pressure` > 30% | 大量 pin 被显存压力踢掉 | TTL 太长，相对可用 KV cache 过于激进 |
| `handed off` 60%+ | 健康 | — |

报告会自动在越界时打印提示行。

### Decision sources 怎么读

| 现象 | 含义 | 处理 |
| --- | --- | --- |
| `ttl_src` 长期是 `cold_start` | 工具耗时历史没攒够 | 正常冷启动，或 `CONTINUUM_HISTORY_THRESHOLD` 设太高 |
| `recon_src` 出现 `prefill_fallback` | 有外部层但没有可用 reload 估计 | 看 4.1 的 mode，多半是同步 connector |
| `recon_src` 全是 `reload` | 在线估计已生效 | 健康 |

---

## 6. 排查

### 6.1 日志里没有 `Continuum scheduler active`

插件没被加载。按顺序查：`pip show continuum-vllm` → `continuum-vllm-check` →
`--scheduler-cls` 全路径拼写。

### 6.2 `pins created` 一直是 0

TTL 决策没产出有效 pin。查：

1. 请求里有没有带 `vllm_xargs.job_id`？没有 `job_id` 的请求整个机制不启动。
2. 有没有带 `this_func_call`？没带的话默认解析器只认输出里**恰好一个**
   ` ```bash ` 代码块，其他 agent 格式一律解析失败。
3. `is_last_step` 是不是每轮都传了 1？最后一轮不 pin 是正确行为。

### 6.3 `reload_estimator=prefill_fallback`

有外部层但测不到，两个原因：

- `kv_connector_extra_config.load_async` 被显式设成了 false → 改回 true
- 用的是 `SimpleCPUOffloadConnector` → 该 connector 恒为同步，调度器无法测量。
  只能换成 `OffloadingConnector`，或接受保守估计（TTL 会偏长）

### 6.4 `reload_estimator=online` 但 `reload_total` 一直是 0

插件会在 64 次决策后打一条 WARNING：

```
Continuum made 64 TTL decisions with KV tier X attached but observed no
asynchronous reload samples. The connector is loading synchronously, ...
```

这是兜底探测，能抓住类名不在已知同步名单里、但实际同步加载的 connector。
处理方式同 6.3。

### 6.5 `continuum-vllm-check` 显示 `ConstantPrefillCostEstimator` 且恒为 2.0

`CONTINUUM_PREFILL_PROFILE` 没生效。查路径是否存在、进程是否能读、JSON 里是否有
`prefill_coefficients` 对象。

### 6.6 找不到 dump 文件

文件名带进程号。用通配符：

```bash
ls -la /data/continuum/stats.*.json
```

如果一个都没有：`CONTINUUM_STATS_DUMP_PATH` 没设，或
`CONTINUUM_STATS_INTERVAL_SECONDS=0`（关掉了周期刷新，只有 shutdown 时才写）。

---

## 7. 环境变量全表

| 变量 | 默认 | 作用 |
| --- | ---: | --- |
| `CONTINUUM_PREFILL_PROFILE` | — | 阶段 A 产出的 JSON 路径。**优先级最高** |
| `CONTINUUM_PREFILL_QUADRATIC` | — | 手工指定二次项，三个必须同时给 |
| `CONTINUUM_PREFILL_LINEAR` | — | 手工指定一次项 |
| `CONTINUUM_PREFILL_CONSTANT` | — | 手工指定常数项 |
| `CONTINUUM_PREFILL_SECONDS` | `2.0` | 上面都没给时的恒定回退值 |
| `CONTINUUM_RECONSTRUCTION_SECONDS` | — | v0.1.0 的旧名，仍兼容，等价于上一行 |
| `CONTINUUM_RELOAD_MIN_SAMPLES` | `32` | 攒够多少 reload 样本才启用在线估计 |
| `CONTINUUM_RELOAD_MAX_SAMPLES` | `1024` | 拟合窗口上限，超出丢弃最旧的 |
| `CONTINUUM_HISTORY_THRESHOLD` | `100` | 工具耗时样本阈值 K，冷启动三级策略的分界 |
| `CONTINUUM_COLD_START_TTL_SECONDS` | `2.0` | 样本不足时的固定 TTL |
| `CONTINUUM_MAX_HISTORY_SAMPLES` | `4096` | 全局/单工具耗时历史上限 |
| `CONTINUUM_QUEUE_DELAY_WINDOW_SIZE` | `100` | 排队延迟 T 的滑动窗口大小 |
| `CONTINUUM_STATS_INTERVAL_SECONDS` | `60` | 周期日志 + dump 刷新间隔，`0` 关闭 |
| `CONTINUUM_STATS_DUMP_PATH` | — | dump 路径，会自动追加 `.<pid>` |

优先级：`PROFILE` 文件 > 三个系数 > `RECONSTRUCTION_SECONDS` > `PREFILL_SECONDS`。

---

## 8. 请求侧约定

每一轮都要带 `vllm_xargs`，同一个 agent 程序的所有轮次共用一个 `job_id`：

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

下一轮把上一轮实际执行的工具名回传，用于统计工具耗时：

```json
{
  "vllm_xargs": {
    "job_id": "agent-001",
    "last_func_call": "pytest",
    "this_func_call": "git",
    "is_last_step": 0
  }
}
```

最后一轮必须 `"is_last_step": 1`，否则程序结束后 pin 不会被立即释放，只能等 TTL。

布尔值用 `0` / `1`，因为 `vllm_xargs` 只接受标量字符串和数字。
