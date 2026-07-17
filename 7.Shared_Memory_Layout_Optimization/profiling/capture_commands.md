# Shared-memory Padding GEMM Capture Commands

当前 kernel 参数为 `BM=64`、`BN=64`、`BK=8`、`TM=8`、`TN=8`，固定使用
`block=(64, 1)`。输入尺寸必须满足：

```text
M % 64 == 0, N % 64 == 0, K % 8 == 0
```

## Build

```bash
cd /home/fish/GEMM_For_Myself/7.Shared_Memory_Layout_Optimization

nvcc -O3 -lineinfo -arch=sm_89 \
  Shared_Memory_Layout_Padding_benchmark.cu \
  -o Shared_Memory_Layout_Padding_bench
```

## Correctness Check

```bash
./Shared_Memory_Layout_Padding_bench 1024 1024 1024 \
  --warmup 2 --iters 5
```

成功时 `check` 列应显示 `PASS`。

## Instruction Statistics Report

```bash
ncu -f \
  --section InstructionStats \
  --kernel-name-base demangled \
  --kernel-name 'regex:.*sgemm_shared_memory_layout_padding.*' \
  --launch-skip 1 \
  --launch-count 1 \
  -o Shared_Memory_Layout_Padding_instr \
  ./Shared_Memory_Layout_Padding_bench 1024 1024 1024 \
  --warmup 1 --iters 3 --no-check
```

生成 `Shared_Memory_Layout_Padding_instr.ncu-rep`。

## Full Report

```bash
ncu -f \
  --set full \
  --kernel-name-base demangled \
  --kernel-name 'regex:.*sgemm_shared_memory_layout_padding.*' \
  --launch-skip 1 \
  --launch-count 1 \
  -o Shared_Memory_Layout_Padding_full \
  ./Shared_Memory_Layout_Padding_bench 1024 1024 1024 \
  --warmup 1 --iters 3 --no-check
```

生成 `Shared_Memory_Layout_Padding_full.ncu-rep`。

## Open Reports

```bash
ncu-ui Shared_Memory_Layout_Padding_instr.ncu-rep
ncu-ui Shared_Memory_Layout_Padding_full.ncu-rep
```

`ncu` 会 replay kernel，因此 profiler 下 benchmark 打印的 latency 不用于性能
对比。`--warmup 1` 与 `--launch-skip 1` 配套，只采集第一次正式测量 launch。
