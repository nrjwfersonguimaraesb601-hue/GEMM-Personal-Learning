# Coalesced Full Report

把完整报告截图放到 `figures/`。

当前保留：

- `01_summary.png`
- `02_speed_of_light.png`
- `03_memory_workload.png`
- `04_scheduler_statistics.png`
- `05_warp_state.png`

如果这一版的 `Memory Chart` 很清楚，再额外保留：

- `06_memory_chart.png`

## 当前解读

### `01_summary.png`

这张图已经能说明 coalesced 版为什么是一个明显的中间 checkpoint：

- kernel time 下降到大约 `2.49 ms`
- compute throughput 提升到 `97.82%`
- memory throughput 也在 `97.82%`

和 naive 相比，这个变化非常大。  
它说明简单调整线程映射以后，kernel 已经不再是那种“只会把内存打满、SM 却闲着”的状态。

下方的 opportunities 也变了味道：

- `L1TEX Global Load Access Pattern` 还在
- `Lg Throttle Stalls` 还在
- 还出现了 `Theoretical Occupancy`

也就是说，这一版虽然显著变快了，但瓶颈已经从“纯粹的严重 non-coalesced”开始往更细的层面移动。

### `02_speed_of_light.png`

这张图是和 naive 对比时最直观的一张：

- `Compute (SM) Throughput = 97.82%`
- `Memory Throughput = 97.82%`

和 naive 的 `11.91% compute / 98.22% memory` 比起来，这里的变化几乎可以直接写进项目总结：

- coalescing 之后，SM 不再被明显饿住
- 计算和内存都进入高利用区间
- kernel 不再是“明显只受单一坏访问模式拖累”的状态

### `03_memory_workload.png`

这一页最关键的数字是：

- 每个 sector 的平均有效利用从 naive 的 `4.0 / 32 bytes`
- 提升到这里的 `26.4 / 32 bytes`

虽然它还不是完美的 `32 / 32`，但已经是非常实在的改善。  
这正是为什么：

- 指令表面上还是有很多 `LDG`
- 但最终吞吐却能从 naive 的低效状态跃迁到接近满载

这一步优化的本质就是：

- 没有减少“要读多少数据”
- 但显著减少了“为这些读取浪费掉的 transaction”

### `04_scheduler_statistics.png`

调度层面的改善也很明显：

- `Eligible Warps Per Scheduler` 提升到 `1.16`
- `Issued Warp Per Scheduler` 提升到 `0.29`
- `No Eligible` 降到 `71.20%`

它仍然不是一个完全理想的值，但和 naive 的：

- `Eligible = 0.13`
- `Issued = 0.03`
- `No Eligible = 96.50%`

相比，已经好很多。  
这说明访存模式改善以后，warp 更容易进入“能真正发射”的状态。

### `05_warp_state.png`

这里的主 stall 仍然是 `LG Throttle`，但数值已经从 naive 的大约 `210`
下降到这里的 `19.40` 左右。

这很重要，因为它说明：

- 问题没有完全消失
- 但已经从“压倒性的 global load/store 节流”缓和成了“仍有负担，但可以继续优化”的状态

这一版 full report 最适合得出的总结是：

- coalescing 没有改变 GEMM 的基本算法
- 但它几乎单靠访问模式整理，就把 kernel 从严重 memory-bound baseline 推到了接近高吞吐区间
