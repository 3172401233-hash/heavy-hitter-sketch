# heavy-hitter-sketch

Python 高频键（heavy hitter）近似统计库：每个分片用有明确误差上界的
Count-Min Sketch 统计访问量，序列化后跨进程合并成全局 Top-K，
避免全量 Counter 的内存开销，也避免普通本地 Top-K 在分片交界处漏掉真正高频的 key。

## 快速开始

```python
from heavy_hitters import HeavyHitterSketch

# 每个分片一个 sketch，按内存预算自动定参
shard = HeavyHitterSketch.from_budget(budget_bytes=64_000, delta=0.01)
shard.add("user:123", weight=1)
shard.add("user:123", weight=2)

# 序列化 -> 跨进程传输 -> 合并
payload = shard.dumps()                 # bytes，可走 RPC / 文件 / 消息队列
remote = HeavyHitterSketch.loads(payload)
global_sketch = HeavyHitterSketch.merged([remote, other_shard, ...])

# 全局 Top-K：排名 + 估计计数 + 误差范围 + 确认/可能标记
for row in global_sketch.topk(10):
    print(row.key, row.estimate, row.lower, row.upper, row.status)

# 或者按阈值比例捞高频键（计数 >= phi * 总权重）
hitters = global_sketch.heavy_hitters(phi=0.01)
```

## 参数

| 参数 | 含义 | 影响 |
| --- | --- | --- |
| `width` | 每行计数器个数 | 越大误差越小：`epsilon = e / width` |
| `depth` | 哈希行数 | 越大失败概率越小：`delta = exp(-depth)` |
| `capacity` | 候选 key 表上限 | 决定 Top-K 最多能跟踪多少个不同 key |
| `budget_bytes` | `from_budget` 的总内存预算 | 75% 分给计数器、25% 分给候选表，自动推算 width/depth/capacity |
| `delta` | `from_budget` 的失败概率 | 反推 `depth = ceil(ln(1/delta))` |
| `weight` | `add` 的权重 | 非负数，计数器按权重累加 |
| `k` / `threshold` | `topk` 的排名长度与判定阈值 | 阈值默认取 `error_bound` |
| `phi` | `heavy_hitters` 的比例阈值 | 返回估计计数 `>= phi * total` 的 key |

内存占用约为 `width * depth * 8 + capacity * 64` 字节（`memory_bytes` 属性可查）。
预算极小时会钳制到最小形状（width=2, depth=1, capacity=1），此时实际占用可能略超预算，
但误差上界依然成立（只是 `epsilon` 很大）。

## 误差模型

设 `N = total`（所有权重之和），`epsilon = e / width`，`delta = exp(-depth)`。
对任意 key 的真实计数 `c`：

- `estimate >= c`（Count-Min 只多不少，永不低估）；
- 以概率 `>= 1 - delta` 有 `estimate <= c + epsilon * N`。

因此每个结果行给出区间 `[lower, upper] = [estimate - epsilon*N, estimate]`，
真实计数以 `>= 1 - delta` 的概率落在其中。合并不改变该上界：
合并后的 sketch 等价于对全量流直接建立的 sketch（计数器逐元素相加）。

## “确认高频” vs “可能高频”

对阈值 `T`（`topk` 默认 `T = error_bound`，`heavy_hitters` 中 `T = phi * N`）：

- `confirmed`：`lower = estimate - epsilon*N >= T`，即使按最坏误差扣除仍达标，确认高频；
- `possible`：`estimate >= T` 但 `lower < T`，估计值达标但误差带无法排除虚高，仅为可能高频；
- `estimate < T` 的 key 不会出现在 `heavy_hitters` 结果中。

## 确定性与可合并性

- 哈希使用 blake2b（固定域分离种子），不依赖进程随机盐，跨进程结果一致；
- 计数器状态是更新的纯和，插入顺序不影响最终 sketch；
  相同权重分布以任意顺序插入，序列化结果与 Top-K 输出完全一致；
- 候选表裁剪与排名均以 `(-estimate, key)` 排序，平局按 key 字典序，保证输出稳定；
- 合并要求各分片 `width`/`depth` 相同（建议统一用 `from_budget` 定参），否则抛 `ValueError`。

注意：当不同 key 数超过 `capacity` 时，候选表会发生裁剪，极端情况下排名尾部
可能与全量统计略有出入；把 `capacity` 调到不小于不同 key 数即可完全避免。

## 测试

```bash
python -m unittest
```

覆盖：均匀流量、单 key 偏斜、重复 key 与权重、分片合并（含序列化传输与合并顺序无关）、
空输入、极小预算、插入顺序稳定性、序列化往返、确认/可能高频区分。
