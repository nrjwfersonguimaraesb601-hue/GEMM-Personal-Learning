# Vectorized GEMM Visual Checklist

- Summary：记录 grid/block、`117 registers/thread`、tail effect 与优化提示
- Speed of Light：对比 compute、memory、L1/TEX、L2 与 DRAM throughput
- Instruction Statistics：确认 `FFMA`、`LDS`、`STS` 的主导关系
- Memory Workload：检查 global-store sector 利用率和 shared load/store bank conflict
- Scheduler Statistics：检查 active、eligible、issued warps 与 issue interval
- Warp State：检查 `MIO Throttle`、Long/Short Scoreboard 和 Barrier
- 与 2D 报告对照：确认向量化改善了什么，以及瓶颈是否转移
