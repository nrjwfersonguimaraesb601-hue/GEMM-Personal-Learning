# Naive Kernel Visual Checklist

这一版最值得放的图，不是“多”，而是要能说明为什么它慢。

## 指令报告值得放的图

- `Instruction Statistics` 整页合并图
  一张图同时保留统计表和 `Executed Instruction Mix` 柱状图即可。
- `Source`
  这一版改为可选，不再强制要求。

## 完整报告值得放的图

- `Summary`
  用来交代 kernel 名、grid/block、总时间。
- `Speed Of Light`
  看是更像 memory-bound 还是 compute-bound。
- `Memory Workload Analysis`
  看 global load/store 压力、sector/request 是否难看。
- `Scheduler Statistics` 或 `Warp State`
  看 stall 是否偏向 memory dependency。

## 这一版特别值得讲的观察

- `LDG` 很重，说明线程重复从 global memory 取数。
- `A` 和 `C` 的访问模式不友好，适合和 coalesced 版做一组对照图。
- 如果 `dram` 很忙但 `compute` 不高，这正好支持“baseline 受访存限制”的结论。
