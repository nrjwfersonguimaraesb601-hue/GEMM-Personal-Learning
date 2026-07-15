# SMEM Kernel Capture Commands

## 1. instruction-only 报告

```bash
ncu \
  --set full \
  --section InstructionStats \
  --kernel-name calculate_Matrix \
  -o smem_instr \
  ./My_SMEM_kernel_benchmark 4096 4096 4096 --iters 50 --no-check
```

## 2. 完整报告

```bash
ncu \
  --set full \
  --kernel-name calculate_Matrix \
  -o smem_full \
  ./My_SMEM_kernel_benchmark 4096 4096 4096 --iters 50 --no-check
```

## 3. 建议重点看的 section

- Summary
- Speed Of Light
- Instruction Statistics
- Memory Workload Analysis
- Shared Memory
- Occupancy
- Scheduler Statistics
- Source

## 4. 记录模板

- Profile date:
- GPU:
- Matrix size:
- Tile size:
- Block size:
- Warmup / iters:
- Compare target: coalesced + naive
