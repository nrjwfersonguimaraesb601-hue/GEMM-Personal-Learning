# 2D Blocktiling Capture Commands

```bash
# Full report
ncu -f --set full \
  --kernel-name-base demangled \
  --kernel-name regex:.*sgemm_2d.* \
  --launch-skip 1 --launch-count 1 \
  -o 2D_Blocktiling_full \
  ./2D_Blocktiling_bench 1024 1024 1024 \
  --warmup 1 --iters 3 --bx 64 --by 1 --no-check

# Instruction statistics
ncu -f --section InstructionStats \
  --kernel-name-base demangled \
  --kernel-name regex:.*sgemm_2d.* \
  --launch-skip 1 --launch-count 1 \
  -o 2D_Blocktiling_InstructionStats \
  ./2D_Blocktiling_bench 1024 1024 1024 \
  --warmup 1 --iters 3 --bx 64 --by 1 --no-check
```

如果 kernel 使用 `calculate_Matrix`，将 regex 改为 `regex:.*calculate_Matrix.*`。
