# 2D Blocktiling Profiling

这组 Nsight Compute 截图记录 `BM=64, BN=64, BK=8, TM=8, TN=8` 的基础 2D register tiling kernel。

## Reading Order

1. [Summary](./full_report/figures/01_summary.png)
2. [Speed of Light](./full_report/figures/02_speed_of_light.png)
3. [Instruction Statistics](./instr_report/figures/01_instruction_statistics_and_mix.png)
4. [Memory Workload](./full_report/figures/03_memory_workload.png)
5. [Scheduler Statistics](./full_report/figures/04_scheduler_statistics.png)
6. [Warp State](./full_report/figures/05_warp_state.png)

## Key Findings

- Compute throughput `42.63%`，memory throughput `84.73%`
- L1/TEX throughput `91.95%`，但 DRAM throughput 仅 `5.56%`，压力集中在片上数据路径
- `116 registers/thread` 带来较高寄存器压力
- shared load 平均约 `4.8-way` bank conflict
- global store 每个 32-byte sector 平均仅使用 `4 bytes`
- 调度器平均每 `2.7 cycles` 发射一次指令，主要 stall 为 `MIO Throttle`

这些指标说明下一步应优先优化 shared-memory layout、C 写回模式和寄存器占用。

## Files

- [Full report notes](./full_report/README.md)
- [Instruction report notes](./instr_report/README.md)
- [Capture commands](./capture_commands.md)
- [Visual checklist](./visual_checklist.md)
