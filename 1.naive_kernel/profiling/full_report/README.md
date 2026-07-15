# Naive Full Report

把完整 Nsight Compute 报告截图放到 `figures/` 目录。

当前保留这些图：

- `01_summary.png`
- `02_speed_of_light.png`
- `03_memory_workload.png`
- `04_scheduler_statistics.png`
- `05_warp_state.png`

`Source` 截图现在是可选项，不再作为必需项。

## 当前解读

### `01_summary.png`

这张图把 naive 版最关键的几件事直接摆出来了：

- kernel time 大约 `20.49 ms`
- compute throughput 只有 `11.91%`
- memory throughput 却高到 `98.22%`

下面的 optimization opportunities 也非常直接：

- `L1TEX Global Load Access Pattern`
- `L1TEX Global Store Access Pattern`
- `Uncoalesced Global Accesses`

也就是说，Nsight Compute 几乎是在明说这版 kernel 的主要问题就是：

- global load/store 访问模式差
- warp 内线程访问不连续
- 大量 transaction 被浪费在 non-coalesced 访存上

### `02_speed_of_light.png`

这张图最重要的结论非常清楚：

- `Memory Throughput = 98.22%`
- `Compute (SM) Throughput = 11.91%`

这不是“算得不够多”的问题，而是典型的 memory-bound baseline：

- 内存通路已经很忙
- 计算单元却远没有被压满

对这个项目来说，这张图的价值很高，因为它把“naive 为什么慢”从代码直觉变成了可量化证据。

### `03_memory_workload.png`

这一页给出了最有说服力的访存证据：

- global load access pattern 明确被标成不理想
- global store access pattern 也不理想
- 每个 sector 平均只有 `4.0 / 32 bytes` 被有效利用

这正是 non-coalesced baseline 的典型症状：

- 数据确实被搬了
- 但线程没有把一整个 transaction 用满
- 带宽被浪费在低利用率访问上

这张图也解释了为什么 `Speed Of Light` 里 memory 已经接近打满，但实际 GFLOPS 仍然不高。

### `04_scheduler_statistics.png`

这张图说明问题已经进一步传导到了调度层：

- `Active Warps Per Scheduler` 大约 `7.87`
- `Eligible Warps Per Scheduler` 只有 `0.13`
- `Issued Warp Per Scheduler` 只有 `0.03`
- `No Eligible` 高达 `96.50%`

换句话说，不是 warp 数量完全不够，而是大多数时刻 warp 都在等，真正随时可发射的 warp 很少。  
这也是为什么 naive 版看起来“线程很多”，但发射效率却很差。

### `05_warp_state.png`

这一页把 stall 原因说得更具体了：

- `Stall LG Throttle` 约 `210.21`
- 占总 stall 的主导地位
- 远高于其他 stall 项

这和前面的几张图能完整闭环：

1. `Instruction Statistics` 里 `LDG` 很重
2. `Memory Workload` 里 global access pattern 很差
3. `Scheduler Statistics` 里 eligible warp 很少
4. `Warp State` 里最终体现为 `LG Throttle` 主导

所以这套 naive profiling 的总判断可以写成一句话：

- naive 版的主要瓶颈不是算术单元，而是低效的 global-memory 访问把调度和发射都拖住了
