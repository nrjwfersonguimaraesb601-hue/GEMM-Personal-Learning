# 1D Blocktiling Capture Commands

下面这组命令按当前目录里的 benchmark 口径整理：

- block: `512 x 1`
- warmup: `10`
- iterations: `50`
- correctness check: `disabled`

## 1. 采 instruction-only 报告

```bash
ncu \
  --set full \
  --section InstructionStats \
  --kernel-name calculate_Matrix \
  -o 1D_Blocktiling_instr \
  ./1D_Blocktiling_kernel_benchmark 4096 4096 4096 --iters 50 --no-check
```

## 2. 采完整报告

```bash
ncu \
  --set full \
  --kernel-name calculate_Matrix \
  -o 1D_Blocktiling_full \
  ./1D_Blocktiling_kernel_benchmark 4096 4096 4096 --iters 50 --no-check
```

## 3. 记录模板

- Profile date:
- GPU:
- Matrix size:
- Block size:
- Warmup / iters:
- Binary note:
