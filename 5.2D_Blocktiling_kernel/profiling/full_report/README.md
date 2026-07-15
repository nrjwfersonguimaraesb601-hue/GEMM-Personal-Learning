# 2D Blocktiling Full Report

## Figures

- `01_summary.png`: kernel 配置、throughput 与主要优化提示
- `02_speed_of_light.png`: compute、memory 与 cache throughput
- `03_memory_workload.png`: cache 数据流、global store 与 shared load 问题
- `04_scheduler_statistics.png`: active、eligible 与 issued warps
- `05_warp_state.png`: 各类 warp stall 分布

## Interpretation

Summary 显示 `42.63%` compute throughput、`84.73%` memory throughput 和 `116 registers/thread`。Memory Workload 进一步指出 global store sector 利用率低，以及约 `4.8-way` 的 shared load bank conflict。

Scheduler 平均只有 `0.77` 个 eligible warps、`0.37` 个 issued warps；Warp State 中 `MIO Throttle` 约为 `2.98 cycles/instruction`，是最明显的 stall。当前性能限制主要位于 shared-memory/MIO 路径、写回访问模式与寄存器压力，而不是 DRAM 带宽。
