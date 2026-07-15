# Vectorized GEMM Capture Commands

当前 kernel 参数为 `BM=64`、`BN=64`、`BK=8`、`TM=8`、`TN=8`，
因此 launch 必须使用 `block=(64, 1)`。kernel 没有边界分支，输入尺寸必须满足：

```text
M % 64 == 0, N % 64 == 0, K % 8 == 0
```

## Build

RTX 4060 Laptop GPU 的 Compute Capability 是 8.9：

```bash
cd /home/fish/GEMM_For_Myself/6.Vectorize_kernel

nvcc -O3 -lineinfo -arch=sm_89 \
  Vectorize_kernel_benchmark.cu \
  -o Vectorize_bench
```

不要为性能测试添加 `-G`。如果要查看 ptxas 输出的寄存器和 shared memory
用量，可以临时添加 `-Xptxas=-v`。

## Correctness Check

先单独验证正确性。这里不能添加 `--no-check`：

```bash
./Vectorize_bench 1024 1024 1024 \
  --warmup 2 --iters 5 --bx 64 --by 1
```

成功时 `check` 列会显示 `PASS`；任何 mismatch 都会显示首个错误位置，
并让程序返回非零退出码。校验使用 `abs_tol=1e-3`、`rel_tol=1e-6`，同时
输出 `max err`。默认 `max_check_dim=4096`，所以默认 suite 中的所有尺寸都会
实际校验，而不会对 4096 尺寸显示 `SKIP`。

## Instruction Statistics Report

```bash
ncu -f \
  --section InstructionStats \
  --kernel-name-base demangled \
  --kernel-name 'regex:.*sgemm_vectorize_GEMM_SMEM.*' \
  --launch-skip 1 \
  --launch-count 1 \
  -o Vectorize_instr \
  ./Vectorize_bench 1024 1024 1024 \
  --warmup 1 --iters 3 --bx 64 --by 1 --no-check
```

生成 `Vectorize_instr.ncu-rep`。

## Full Report

```bash
ncu -f \
  --set full \
  --kernel-name-base demangled \
  --kernel-name 'regex:.*sgemm_vectorize_GEMM_SMEM.*' \
  --launch-skip 1 \
  --launch-count 1 \
  -o Vectorize_full \
  ./Vectorize_bench 1024 1024 1024 \
  --warmup 1 --iters 3 --bx 64 --by 1 --no-check
```

生成 `Vectorize_full.ncu-rep`。

`--warmup 1` 与 `--launch-skip 1` 配套：跳过一次预热 launch，只采集第一
次正式计时 launch。Nsight Compute 会 replay 被采集的 kernel，因此运行在
`ncu` 下时 benchmark 打印的 `avg(ms)` 会被严重放大；性能对比应使用不带
`ncu` 的正常运行结果。

`1024^3` 会产生 `16 x 16 = 256` 个 block，足以覆盖这张 24-SM GPU，数据
分配约为 12 MiB，同时 full-set 的 replay 时间仍然较短，适合与前几个
kernel 的 `1024^3` 报告直接比较。

## Open Reports

```bash
ncu-ui Vectorize_instr.ncu-rep
ncu-ui Vectorize_full.ncu-rep
```
