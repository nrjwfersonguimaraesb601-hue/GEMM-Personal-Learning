# Double Buffering Full Report

## Figures

- `01_summary.png`：launch、duration、throughput、register 和优化提示。
- `02_speed_of_light.png`：compute、memory、cache 和 DRAM throughput。
- `03_memory_workload.png`：global-store sector 与 shared-load conflict。
- `04_scheduler_statistics.png`：active、eligible、issued warps。
- `05_warp_state.png`：warp stall state 分布。

## Interpretation

Memory throughput 74.69% 高于 compute throughput 56.79%，L1/TEX 达 84.58%。
global store 仍只使用每个 32-byte sector 的 16 bytes，shared load 有约 5-way
bank conflict。128 registers/thread 将理论 occupancy 限制为 33.33%。

Scheduler 平均有 3.47 active warps，但只有 1.63 eligible，发射 0.64 warp/cycle；
35.51% 周期没有 eligible warp。double buffering 改善了部分 pipeline 利用率，
但 shared-memory replay、C store pattern 和 occupancy 仍限制最终收益。
