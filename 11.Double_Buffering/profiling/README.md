# Double Buffering Profiling

采集对象为 `sgemm_double_buffering<128,128,8,8,8>`，workload `1024³`，grid
`(8,8,1)`，block `(256,1,1)`。

## Reading Order

1. [Summary](./full_report/figures/01_summary.png)
2. [Speed of Light](./full_report/figures/02_speed_of_light.png)
3. [Instruction Statistics](./instr_report/figures/01_instruction_statistics_and_mix.png)
4. [Memory Workload](./full_report/figures/03_memory_workload.png)
5. [Scheduler Statistics](./full_report/figures/04_scheduler_statistics.png)
6. [Warp State](./full_report/figures/05_warp_state.png)

原始报告：

- [Full report](./raw/Double_Buffering_full.ncu-rep)
- [InstructionStats](./raw/Double_Buffering_instr.ncu-rep)

## Key Findings

| Metric | Value |
| --- | ---: |
| Duration under profiler | 301.92 us |
| Compute / memory throughput | 56.79% / 74.69% |
| L1/TEX / L2 / DRAM throughput | 84.58% / 15.51% / 10.24% |
| Registers/thread | 128 |
| Static shared memory | 16.90 KiB |
| Theoretical / achieved occupancy | 33.33% / 28.93% |
| Active / eligible / issued warps/scheduler | 3.47 / 1.63 / 0.64 |
| No Eligible | 35.51% |
| Shared-load bank conflicts | 4194304, about 5-way |
| Executed instructions | 37746688 |

profiler duration 由 replay 采集，只用于理解报告，不能进入正式 benchmark 表。
