# Shared-memory Padding GEMM Visual Checklist

- Summary：确认 grid/block、`119 registers/thread`、duration 和 tail-effect 提示
- Speed of Light：记录 compute、memory、L1/TEX、L2、DRAM throughput
- Instruction Statistics：检查 `FFMA`、`LDS`、`IADD3`、`STS` 的指令构成
- Memory Workload：确认 shared load/store bank-conflict 计数器均为 0
- Memory Workload：检查 global load/store 的 bytes per sector
- Scheduler Statistics：对比 active、eligible、issued warps 和 No Eligible 比例
- Warp State：检查 Not Selected、Dispatch Stall、Scoreboard、Barrier、MIO Throttle
- 与 Stage 6 对照：确认 padding 消除了什么，以及瓶颈转移到了哪里
