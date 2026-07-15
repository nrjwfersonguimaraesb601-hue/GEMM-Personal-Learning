# 2D Blocktiling Visual Checklist

- Summary：记录 kernel、block size、寄存器数与优化提示
- Speed of Light：对比 compute、memory、L1/TEX、L2 与 DRAM throughput
- Instruction Statistics：确认 `FFMA` 与 `LDS` 的主导关系
- Memory Workload：检查 global store pattern 与 shared load bank conflict
- Scheduler Statistics：检查 eligible/issued warps 与 issue slot utilization
- Warp State：检查 `MIO Throttle`、scoreboard 与 barrier stalls
