# Shared-memory Padding GEMM Full Report

## Figures

- [01_summary.png](./figures/01_summary.png)：launch 配置、duration、throughput、
  register 数量和 Nsight 优化提示
- [02_speed_of_light.png](./figures/02_speed_of_light.png)：compute、memory、cache
  和 DRAM throughput
- [03_memory_workload.png](./figures/03_memory_workload.png)：cache 数据流、global
  sector 利用率和 shared-memory 指标
- [04_scheduler_statistics.png](./figures/04_scheduler_statistics.png)：active、eligible
  和 issued warps
- [05_warp_state.png](./figures/05_warp_state.png)：warp stall 分布

## Summary

| Metric | Value |
| ------ | ----: |
| Duration | `282.98 us` |
| Compute Throughput | `63.73%` |
| Memory Throughput | `65.72%` |
| L1/TEX Throughput | `68.23%` |
| L2 Throughput | `23.13%` |
| DRAM Throughput | `10.97%` |
| Registers/thread | `119` |
| Static shared memory | `4352 bytes` |

Nsight 的 Summary 对 `1024³` launch 给出 Tail Effect 提示：`256` 个 blocks 在
24 个 SM 上形成一个完整 wave 和一个包含 63 个 blocks 的 partial wave。界面中的
`50%` estimated speedup 是假设每个 block 时间完全一致时的模型上界，不代表只改
一行代码就能获得 2 倍性能；需要用更大 grid 或不同尺寸复测 tail effect。

## Memory Workload

| Metric | Value |
| ------ | ----: |
| Memory throughput | `29.82 GB/s` |
| L1/TEX hit rate | `16.51%` |
| L2 hit rate | `93.48%` |
| Average global-load bytes/sector | `32/32` |
| Average global-store bytes/sector | `16/32` |
| Shared-load bank conflicts | `0` |
| Shared-store bank conflicts | `0` |

Shared-memory 表中四个 bank-conflict 计数器——总计、load、store 和 atomic——
全部为 0，说明本阶段 As/Bs padding 的目标已经在这次采集中实现。Stage 6 对应
的 shared load/store conflict 分别为 `4,194,304` 和 `524,288`。

global store 仍只使用每个 32-byte sector 的 16 bytes。Summary 还记录了
`131,072` 个 excessive global sectors，约占 `4,456,448` 个 sectors 的 `3%`。
因此下一步 memory-layout 工作应转向 C 的线程映射和 global-store 合并访问，
而不是继续为当前 As/Bs 添加 padding。

## Scheduler Statistics

- Active warps/scheduler：`3.24`
- Eligible warps/scheduler：`1.85`
- Issued warps/scheduler：`0.68`
- One or more eligible：`67.68%`
- No eligible：`32.32%`

Stage 6 的 eligible/issued 分别为 `1.42/0.57`，No Eligible 为 `42.58%`。
padding 后 scheduler 更经常能找到可发射的 warp，这与 shared-memory excessive
wavefront 消失的结果一致。

## Warp State

- Warp cycles per issued instruction：`4.79`
- Not Selected：`1.73` cycles/instruction
- Dispatch Stall：`0.56`
- Long Scoreboard：`0.50`
- Barrier：`0.31`
- MIO Throttle：`0.26`
- Short Scoreboard：`0.23`
- Wait：`0.13`

MIO Throttle 从 Stage 6 的 `0.86` 降到 `0.26`，Short Scoreboard 从 `0.53`
降到 `0.23`。Not Selected 上升并不等同于坏事：它表示存在其他 eligible warp
被 scheduler 选中。当前更值得继续观察的是 Dispatch Stall、Long Scoreboard
和同步 barrier。
