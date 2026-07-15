# Coalesced Kernel Capture Commands

## 1. instruction-only 报告

```bash
ncu \
  --set full \
  --section InstructionStats \
  --kernel-name calculate_Matrix \
  -o coalesced_instr \
  ./My_Global_Memory_Coalescing_kernel_benchmarker 4096 4096 4096 --iters 50 --no-check
```

## 2. 完整报告

```bash
ncu \
  --set full \
  --kernel-name calculate_Matrix \
  -o coalesced_full \
  ./My_Global_Memory_Coalescing_kernel_benchmarker 4096 4096 4096 --iters 50 --no-check
```

## 3. 建议重点看的 section

- Summary
- Speed Of Light
- Memory Workload Analysis
- Instruction Statistics
- Memory Chart
- Source

## 4. 记录模板

- Profile date:
- GPU:
- Matrix size:
- Block size:
- Warmup / iters:
- Compare target: naive_kernel
