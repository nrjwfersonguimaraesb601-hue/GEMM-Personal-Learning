# Naive Kernel Capture Commands

下面是建议保留在仓库里的 profiling 命令模板，你后面可以按实际可执行文件路径微调。

## 1. 采 instruction-only 报告

```bash
ncu \
  --set full \
  --section InstructionStats \
  --kernel-name calculate_Matrix \
  -o naive_instr \
  ./My_naive_kernel_benchmark 4096 4096 4096 --iters 50 --no-check
```

## 2. 采完整报告

```bash
ncu \
  --set full \
  --kernel-name calculate_Matrix \
  -o naive_full \
  ./My_naive_kernel_benchmark 4096 4096 4096 --iters 50 --no-check
```

## 3. 如果你想固定单次 case，建议记在这里

```bash
# example
./My_naive_kernel_benchmark 1024 1024 1024 --iters 100 --no-check
```

## 4. 这一版建议重点看的 section

- Summary
- Speed Of Light
- Instruction Statistics
- Memory Workload Analysis
- Scheduler Statistics
- Source

## 5. 记录模板

- Profile date:
- GPU:
- Matrix size:
- Block size:
- Warmup / iters:
- Binary commit or note:
