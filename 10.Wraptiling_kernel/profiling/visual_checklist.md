# Warp Tiling Visual Checklist

- Summary：确认 grid `(8,8,1)`、block 128、168 registers/thread。
- Speed of Light：记录 compute/memory/L1/L2/DRAM throughput。
- Memory Workload：检查 shared-store 4-way bank conflict 和 excessive wavefronts。
- Scheduler：检查 active/eligible/issued warps 与 No Eligible 45.84%。
- Warp State：记录主要 stall reason，不从源码猜测。
- Occupancy：确认 register 和 shared memory 对 25% 理论 occupancy 的限制。
- Instruction Statistics：导出 executed instruction mix 后再判断指令宽度。
