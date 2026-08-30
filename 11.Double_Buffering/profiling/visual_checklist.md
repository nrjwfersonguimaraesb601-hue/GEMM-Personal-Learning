# Double Buffering Visual Checklist

- Summary：确认 block 256、128 registers/thread 和 16.90 KiB static SMEM。
- Speed of Light：记录 compute 56.79%、memory 74.69%、L1/TEX 84.58%。
- Memory Workload：检查 global store 16/32 bytes per sector 和 5-way shared-load conflict。
- Scheduler：检查 active/eligible/issued `3.47/1.63/0.64`。
- Warp State：关注 Not Selected、MIO Throttle、Short Scoreboard、Dispatch、Barrier。
- Instruction Statistics：确认 FFMA 主导，并以 SASS 验证向量指令宽度。
