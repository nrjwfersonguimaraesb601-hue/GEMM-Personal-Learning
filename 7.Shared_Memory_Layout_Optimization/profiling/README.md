# Shared-memory Padding GEMM Profiling

这组 Nsight Compute 截图记录 Stage 7 的 As/Bs padding kernel。采集对象为
`sgemm_shared_memory_layout_padding<64, 64, 8, 8, 8>`，输入尺寸为 `1024³`，
grid 为 `16 x 16`，block 为 `64 x 1`。

## Reading Order

1. [Summary](./full_report/figures/01_summary.png)
2. [Speed of Light](./full_report/figures/02_speed_of_light.png)
3. [Instruction Statistics](./instr_report/figures/01_instruction_statistics_and_mix.png)
4. [Memory Workload](./full_report/figures/03_memory_workload.png)
5. [Scheduler Statistics](./full_report/figures/04_scheduler_statistics.png)
6. [Warp State](./full_report/figures/05_warp_state.png)

## Key Findings

- kernel duration `282.98 us`
- Compute throughput `63.73%`，memory throughput `65.72%`
- L1/TEX throughput `68.23%`，L2 throughput `23.13%`，DRAM throughput `10.97%`
- `119 registers/thread`，static shared memory `4352 bytes`
- shared load bank conflicts：`0`
- shared store bank conflicts：`0`
- global load 每个 32-byte sector 使用 `32 bytes`，global store 仍只使用
  `16 bytes`
- 平均每个 scheduler 有 `1.85` 个 eligible warps，发射 `0.68` 个 warp/cycle
- warp cycles per issued instruction 为 `4.79`
- 主要 stall 为 Not Selected `1.73`、Dispatch Stall `0.56`、Long Scoreboard
  `0.50`、Barrier `0.31` cycles/instruction

本阶段的直接目标已经完成：Stage 6 中 shared load/store 的 bank conflict 在这次
报告中都降为 0。剩余更明显的问题是 C 的 global store 仍只利用半个 sector，
以及 global-memory access、barrier 和 scoreboard 带来的等待。

## Compared with Stage 6 Vectorized

| Metric | Vectorized | Padding |
| ------ | ---------: | ------: |
| Duration | `319.46 us` | `282.98 us` |
| Compute Throughput | `54.20%` | `63.73%` |
| Memory Throughput | `85.03%` | `65.72%` |
| Registers/thread | `117` | `119` |
| Shared-load bank conflicts | `4,194,304` | `0` |
| Shared-store bank conflicts | `524,288` | `0` |
| Global-store bytes/sector | `16/32` | `16/32` |
| Eligible warps/scheduler | `1.42` | `1.85` |
| Issued warps/scheduler | `0.57` | `0.68` |
| No-eligible cycles | `42.58%` | `32.32%` |
| Warp cycles/instruction | `5.72` | `4.79` |
| MIO Throttle | `0.86` | `0.26` cycles/instruction |

Memory Throughput 百分比下降不等于实际性能下降。Stage 6 的较高数值主要表示
memory pipeline 更接近当时的瓶颈；padding 消除 excessive shared wavefront 后，
compute 与 memory throughput 更均衡，报告中的 kernel duration 也更短。

## Files

- [Full report notes](./full_report/README.md)
- [Instruction report notes](./instr_report/README.md)
- [Capture commands](./capture_commands.md)
- [Visual checklist](./visual_checklist.md)
- [Full `.ncu-rep`](./raw/Shared_Memory_Layout_Padding_full.ncu-rep)
- [Instruction `.ncu-rep`](./raw/Shared_Memory_Layout_Padding_instr.ncu-rep)
