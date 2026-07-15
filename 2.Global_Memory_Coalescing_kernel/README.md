# Global Memory Coalescing Kernel

## Idea

这一版是在 naive baseline 上做的第一步优化。

核心变化是调整 thread mapping，让 warp 内线程更自然地访问连续的 global memory 地址。算法本身仍然是一个 thread 计算一个输出元素，不使用 shared memory，也不做 register tiling。

这一阶段主要用来观察：仅仅把 global memory access pattern 调整好，可以带来多大的提升。

## Build

```bash
nvcc My_Global_Memory_Coalescing_kernel.cu -O3 -o gmemc
nvcc My_Global_Memory_Coalescing_kernel_benchmarker.cu -O3 -lineinfo -o gmemc_bench
```

## Run Benchmark

```bash
./gmemc_bench \
  --warmup 10 --iters 50 --bx 32 --by 32 --no-check
```

当前性能表使用 `--no-check`，主要用于 benchmark 性能记录。正确性需要单独去掉 `--no-check` 开启 CPU check。

## Benchmark Result

| M | N | K | Block | Avg (ms) | Avg GFLOPS | Best GFLOPS |
|---|---|---|---|---:|---:|---:|
| 256 | 256 | 256 | 32x32 | 0.0544 | 616.2947 | 642.5098 |
| 512 | 512 | 512 | 32x32 | 0.3233 | 830.3416 | 856.6797 |
| 1024 | 1024 | 1024 | 32x32 | 2.4238 | 886.0027 | 892.0255 |
| 2048 | 2048 | 2048 | 32x32 | 16.7439 | 1026.0384 | 1031.0482 |
| 4096 | 4096 | 4096 | 32x32 | 157.8177 | 870.8718 | 891.6107 |
| 1023 | 1023 | 1023 | 32x32 | 2.2740 | 941.6104 | 1011.1287 |
| 4096 | 256 | 4096 | 32x32 | 8.4779 | 1013.2102 | 1021.0437 |
| 256 | 4096 | 4096 | 32x32 | 9.7675 | 879.4382 | 884.9676 |

## Compared with Naive

| Case | Naive Avg GFLOPS | Coalesced Avg GFLOPS | Change |
|---|---:|---:|---:|
| 256^3 | 92.2776 | 616.2947 | +567.9% |
| 512^3 | 109.5825 | 830.3416 | +657.7% |
| 1024^3 | 123.4315 | 886.0027 | +617.8% |
| 2048^3 | 123.5758 | 1026.0384 | +730.3% |
| 4096^3 | 123.0080 | 870.8718 | +608.0% |

## What This Stage Shows

- global memory coalescing 是第一波最明显的收益来源
- square case 从 naive 的 `~123 GFLOPS` 提升到 `~616-1026 GFLOPS`
- `2048^3` 已经超过 `1 TFLOPS`
- 这一步说明“先把访存方向改对”非常重要

## Known Issues / Notes

- 仍然没有 shared memory tile reuse
- 每个 thread 仍然只计算一个输出
- 后续性能继续提升需要 shared memory tiling 和 register blocking

## Next Step

下一阶段是 `Shared Memory Tiling`：把 `A` / `B` 的 tile 搬到 shared memory 中，让 block 内线程复用数据。
