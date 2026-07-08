# Shared Memory Tiling Kernel

## Idea

这一版在 global memory coalescing 的基础上加入 shared memory tiling。

核心思路：

- 一个 block 负责 C 的一个 tile
- 把对应的 A tile 和 B tile 先加载到 shared memory
- block 内线程重复使用 shared memory 中的数据
- 减少重复 global memory load

这一阶段的重点不是改变每个 thread 的计算数量，而是观察 shared memory reuse 对 GEMM 的影响。

## Build

```bash
nvcc My_SMEM_kernel.cu -O3 -o My_SMEM_kernel
nvcc My_SMEM_kernel_benchmark.cu -O3 -lineinfo -o smem_bench
```

## Run Benchmark

```bash
./smem_bench \
  --warmup 10 --iters 50 --bx 32 --by 32 --no-check
```

当前性能表使用 `--no-check`，主要用于 benchmark 性能记录。正确性需要单独去掉 `--no-check` 开启 CPU check。

## Benchmark Result

| M | N | K | Block | Avg (ms) | Avg GFLOPS | Best GFLOPS |
|---|---|---|---|---:|---:|---:|
| 256 | 256 | 256 | 32x32 | 0.0524 | 640.7902 | 669.5887 |
| 512 | 512 | 512 | 32x32 | 0.3209 | 836.6070 | 876.7358 |
| 1024 | 1024 | 1024 | 32x32 | 2.3737 | 904.6960 | 911.4089 |
| 2048 | 2048 | 2048 | 32x32 | 16.5285 | 1039.4091 | 1043.5539 |
| 4096 | 4096 | 4096 | 32x32 | 143.4114 | 958.3544 | 974.4422 |
| 1023 | 1023 | 1023 | 32x32 | 2.2281 | 961.0163 | 1033.1097 |
| 4096 | 256 | 4096 | 32x32 | 9.1191 | 941.9762 | 978.2633 |
| 256 | 4096 | 4096 | 32x32 | 9.1756 | 936.1697 | 965.5396 |

## Compared with Global Memory Coalescing

| Case | Coalesced Avg GFLOPS | SMEM Avg GFLOPS | Change |
|---|---:|---:|---:|
| 256^3 | 616.2947 | 640.7902 | +4.0% |
| 512^3 | 830.3416 | 836.6070 | +0.8% |
| 1024^3 | 886.0027 | 904.6960 | +2.1% |
| 2048^3 | 1026.0384 | 1039.4091 | +1.3% |
| 4096^3 | 870.8718 | 958.3544 | +10.0% |

## What This Stage Shows

- shared memory tiling 在当前实现中带来稳定但不夸张的提升
- 多数 square case 相对 coalesced 有小幅领先
- `4096^3` 从 `870.8718 GFLOPS` 提升到 `958.3544 GFLOPS`
- shared memory reuse 是后续 block tiling / register tiling 的基础

## Known Issues / Notes

- 每个 thread 仍然只计算一个输出元素
- shared memory tile 已经复用，但寄存器级复用还不够
- 相比 coalesced 的提升没有跨数量级，说明后续需要更深的 register blocking

## Next Step

下一阶段是 `1D Register Tiling`：让每个 thread 在寄存器里累加多个输出，进一步提高数据复用。
