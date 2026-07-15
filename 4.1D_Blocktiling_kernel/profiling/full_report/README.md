# 1D Blocktiling Full Report

把完整 Nsight Compute 报告截图放到 `figures/` 目录。

当前保留这些图：

- `01_summary.png`
- `02_speed_of_light.png`
- `03_memory_chart.png`
- `04_scheduler_statistics.png`
- `05_warp_state.png`

这一版当前没有单独保留 `Memory Workload Analysis`，因为已经有一张更直观的 `Memory Chart`，对 1D block tiling 的数据路径更有帮助。

## 当前解读

### `01_summary.png`

这张图先把这版 kernel 的整体状态摆出来了：

- kernel time 大约 `678.69 us`
- compute throughput `78.74%`
- memory throughput `78.74%`
- `Registers/thread = 56`

下面的优化提示也很有代表性：

- `Mio Throttle Stalls`
- `Theoretical Occupancy`

这说明现在的瓶颈已经不再像 naive 那样集中在 global-memory access pattern，而是开始落到：

- shared-memory / MIO pipeline 压力
- 寄存器带来的 occupancy 限制

### `02_speed_of_light.png`

这张图最值得记的是：

- compute 和 memory 都在 `~78.74%`

对学习项目来说，这个状态很说明问题：

- 它已经不是 naive 那种一边高一边低的失衡状态
- 也不是单纯靠 coalescing 就能解释的阶段
- 说明这版已经进入“计算与数据复用都比较重”的更成熟区间

### `03_memory_chart.png`

这一页很适合拿来理解这版 kernel 的数据路径：

- `L1/TEX Hit Rate` 很低，大约 `1.46%`
- `L2 Hit Rate` 很高，大约 `93.82%`
- shared-memory traffic 很重

这和 kernel 结构是对得上的：

- A/B tile 先经过 global + L2
- block 内再大量通过 shared memory 复用

所以这张图最支持的不是“global memory 完全没问题”，而是：

- 这版的核心收益已经越来越依赖 shared-memory 复用和片上数据流

### `04_scheduler_statistics.png`

这一页主要用来观察：

- active warps
- eligible warps
- issued warps

对 1D block tiling 来说，它的价值在于帮助判断：

- 当前 throughput 是不是被发射条件卡住
- occupancy 限制有没有明显传导到调度层

这一页建议后面和 `2D block tiling` 做重点对照。

### `05_warp_state.png`

这一页已经把 stall 主因说得比较清楚：

- `Stall MIO Throttle` 第一
- `Stall Long Scoreboard` 第二
- `Stall Barrier` 也有存在感

这组组合很像一个典型的 tiling / shared-memory 阶段画像：

- shared-memory 与 MIO pipeline 很忙
- 同步和数据等待都已经进入主要成本

所以这版 full report 最适合写成这样的总结：

- 1D block tiling 已经把性能推到一个明显更高的档位
- 但新的瓶颈已经转成 shared-memory / pipeline / register occupancy 这一侧
