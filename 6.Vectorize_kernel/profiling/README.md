# Vectorized GEMM Profiling

这组 Nsight Compute 截图记录 `BM=64, BN=64, BK=8, TM=8, TN=8` 的
Vectorized GEMM kernel。采集尺寸为 `1024^3`，grid 为 `16 x 16`，block 为
`64 x 1`。

## Reading Order

1. [Summary](./full_report/figures/01_summary.png)
2. [Speed of Light](./full_report/figures/02_speed_of_light.png)
3. [Instruction Statistics](./instr_report/figures/01_instruction_statistics_and_mix.png)
4. [Memory Workload](./full_report/figures/03_memory_workload.png)
5. [Scheduler Statistics](./full_report/figures/04_scheduler_statistics.png)
6. [Warp State](./full_report/figures/05_warp_state.png)

## Key Findings

- Compute throughput `54.20%`，memory throughput `85.03%`
- L1/TEX throughput `87.50%`，L2 throughput `20.81%`，DRAM throughput `9.68%`
- `117 registers/thread`，相比 2D 版本的 `116` 基本不变
- global store 每个 32-byte sector 平均使用 `16 bytes`；比 2D 版本的 `4 bytes` 明显改善，但仍未完全利用 sector
- shared load 平均约 `5.0-way` bank conflict，占 shared-load wavefronts 的 `40.0%`
- shared store 平均约 `2.4-way` bank conflict，占 shared-store wavefronts 的 `33.33%`
- 调度器平均每 `1.7 cycles` 发射一次指令，主要 stall 包括 `MIO Throttle`、Long/Short Scoreboard 和 Barrier

## Compared with 2D Register Tiling

| Metric | 2D Tiling | Vectorized |
| ------ | --------: | ---------: |
| Compute Throughput | `42.63%` | `54.20%` |
| Memory Throughput | `84.73%` | `85.03%` |
| Registers/thread | `116` | `117` |
| Global-store bytes used per 32-byte sector | `4` | `16` |
| Eligible warps/scheduler | `0.77` | `1.42` |
| Issued warps/scheduler | `0.37` | `0.57` |
| Issue interval | `2.7 cycles` | `1.7 cycles` |
| MIO Throttle | `2.98 cycles/instruction` | `0.86 cycles/instruction` |

Vectorized load/store 和 C 写回方式确实改善了 global-memory access 与调度效率，
但 shared-memory bank conflict 并没有消失。下一步应优先处理 A 的 cooperative
load/store 映射、shared-memory layout，以及只使用半个 global-store sector 的问题。

## Files

- [Full report notes](./full_report/README.md)
- [Instruction report notes](./instr_report/README.md)
- [Capture commands](./capture_commands.md)
- [Visual checklist](./visual_checklist.md)
- [Full `.ncu-rep`](./raw/Vectorize_full.ncu-rep)
- [Instruction `.ncu-rep`](./raw/Vectorize_instr.ncu-rep)
