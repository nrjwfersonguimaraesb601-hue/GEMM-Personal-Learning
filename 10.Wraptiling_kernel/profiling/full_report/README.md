# Warp Tiling Full Report

Full report 对应 `1024³` workload。主要结论：

- 168 registers/thread 和 16.38 KiB static shared memory 将理论 occupancy 限制
  在 25%。
- achieved occupancy 为 22.33%，平均只有 1.00 eligible warp/scheduler。
- scheduler 45.84% 的周期没有 eligible warp。
- shared store 存在约 4-way bank conflict，共 786432 次 conflict。
- Compute throughput 52.46%，Memory throughput 33.03%，瓶颈不是单一 DRAM
  bandwidth 问题。

报告由 38 次 replay pass 采集；其中显示的 350.24 us 只用于 profiler 上下文，
不进入 CUDA Event benchmark 表。

`figures/` 预留给从 `ncu-ui` 人工导出的 Summary、Speed of Light、Memory、
Scheduler 和 Warp State 截图。当前没有图片证据，因此不对图表细节做额外推断。
