# Stage 8 Visual Checklist

- 确认截图对应 C00 还是 C08，不混合模板实例。
- Speed of Light：记录 SM、memory、DRAM throughput。
- Memory Workload：检查 global store 利用率和 shared-store conflict。
- Warp State：检查 dispatch、scoreboard 和 Not Selected。
- Launch Statistics：记录 block、register/thread、shared memory 和 occupancy。
- profiler latency 只作上下文，正式排名读取 `results/autotune_full.csv`。
