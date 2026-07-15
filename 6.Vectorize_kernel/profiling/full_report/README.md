# Vectorized GEMM Full Report

## Figures

- `01_summary.png`: kernel 配置、throughput 与主要优化提示
- `02_speed_of_light.png`: compute、memory 与 cache throughput
- `03_memory_workload.png`: cache 数据流、global store 和 shared-memory conflict
- `04_scheduler_statistics.png`: active、eligible 与 issued warps
- `05_warp_state.png`: warp stall 分布

## Summary

| Metric | Value |
| ------ | ----: |
| Duration | `319.46 us` |
| Compute Throughput | `54.20%` |
| Memory Throughput | `85.03%` |
| L1/TEX Throughput | `87.50%` |
| L2 Throughput | `20.81%` |
| DRAM Throughput | `9.68%` |
| Registers/thread | `117` |

Summary 同时提示了 tail effect、global-store access pattern 和 uncoalesced shared
access。`1024^3` 产生 `256` 个 blocks，最后一个不完整 block wave 会降低部分
执行阶段的利用率；这个提示应在更大 grid 上复测，而不能直接视为 kernel 本身
一半时间都可以消除。

## Memory Workload

- L1/TEX hit rate `16.79%`，L2 hit rate `92.30%`
- global store 每个 32-byte sector 平均只使用 `16 bytes`
- shared load 平均 `5.0-way` conflict，共记录 `4,194,304` 次 bank conflicts
- shared store 平均 `2.4-way` conflict，共记录 `524,288` 次 bank conflicts

`float4` 让 C store 相比 2D 版本明显改善，但线程到输出位置的映射仍只使用一半
sector。A 的转置写入改善了后续读取方式，同时也带来了需要继续处理的 shared
store conflict。

## Scheduler and Warp State

- Active warps/scheduler: `3.28`
- Eligible warps/scheduler: `1.42`
- Issued warps/scheduler: `0.57`
- No eligible warp cycles: `42.58%`
- Warp cycles per issued instruction: `5.72`
- 主要 stall：Not Selected `1.47`、MIO Throttle `0.86`、Long Scoreboard `0.60`、Short Scoreboard `0.53` cycles/instruction

Vectorized 版本相对 2D 版本提高了 eligible/issued warps，并明显降低 MIO
Throttle，但 issue slot 仍没有每 cycle 充分利用。shared-memory conflict、数据
依赖和 barrier 都还有继续优化的空间。
