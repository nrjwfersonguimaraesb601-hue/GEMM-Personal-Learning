# 1D Register Tiling Kernel

## Idea

这一版在 shared memory tiling 的基础上加入 1D register tiling。

核心思路：

- 一个 block 负责 `BM x BN` 的 C tile
- shared memory 缓存 `A` / `B` 的 tile
- 每个 thread 不再只计算一个输出
- 每个 thread 负责同一列方向上的 `TM` 个输出
- `threadResult[TM]` 在寄存器中累加

当前配置：

| Parameter  |     Value |
| ---------- | --------: |
| `BM`       |        64 |
| `BN`       |        64 |
| `BK`       |         8 |
| `TM`       |         8 |
| `blockDim` | `512 x 1` |

相比 SMEM 版本，这一版第一次明显利用寄存器来扩大每个 thread 的计算量。

## Build

```bash
nvcc 1D_Blocktiling_kernel.cu -O3 -o 1D_Blocktiling_kernel
nvcc 1D_Blocktiling_kernel_benchmark.cu -O3 -lineinfo -o 1D_Blocktiling_bench
```

## Run Benchmark

```bash
./1D_Blocktiling_bench \
  --warmup 10 --iters 50 --bx 512 --by 1 --no-check
```

当前性能表使用 `--no-check`，主要用于 benchmark 性能记录。正确性需要单独去掉 `--no-check` 开启 CPU check。

## Benchmark Result

| M    | N    | K    | Block | Avg (ms) | Avg GFLOPS | Best GFLOPS |
| ---- | ---- | ---- | ----- | -------: | ---------: | ----------: |
| 256  | 256  | 256  | 512x1 |   0.0267 |  1258.8250 |   1489.4545 |
| 512  | 512  | 512  | 512x1 |   0.1003 |  2676.5946 |   2880.7032 |
| 1024 | 1024 | 1024 | 512x1 |   0.6321 |  3397.1641 |   3426.7189 |
| 2048 | 2048 | 2048 | 512x1 |   4.6454 |  3698.2388 |   4033.1968 |
| 4096 | 4096 | 4096 | 512x1 |  37.5543 |  3659.7348 |   3708.4916 |
| 1023 | 1023 | 1023 | 512x1 |   0.5778 |  3705.5630 |   3788.0688 |
| 4096 | 256  | 4096 | 512x1 |   2.3885 |  3596.3530 |   3628.2908 |
| 256  | 4096 | 4096 | 512x1 |   2.4800 |  3463.6959 |   3486.5371 |

## Compared with Shared Memory Tiling

| Case   | SMEM Avg GFLOPS | 1D Avg GFLOPS |  Change |
| ------ | --------------: | ------------: | ------: |
| 256^3  |        640.7902 |     1258.8250 |  +96.4% |
| 512^3  |        836.6070 |     2676.5946 | +219.9% |
| 1024^3 |        904.6960 |     3397.1641 | +275.5% |
| 2048^3 |       1039.4091 |     3698.2388 | +255.8% |
| 4096^3 |        958.3544 |     3659.7348 | +281.9% |

## What This Stage Shows

- register tiling 是当前路线里非常关键的一步
- square case 从 SMEM 的 `~0.9-1.0 TFLOPS` 推到 `~1.26-3.70 TFLOPS`
- 从 `1024^3` 开始，性能基本稳定进入 `3 TFLOPS+`
- 每个 thread 计算多个输出，明显提高了 shared memory tile 的复用效率

## Known Issues / Notes

- 1D 版本只在一个方向上增加每个 thread 的输出数量
- 每个 thread 仍然只复用一列方向上的结果
- 后续 2D register tiling 会尝试让每个 thread 计算 `TM x TN` 的结果小块

## Next Step

下一阶段是 `2D Register Tiling`：每个 thread 计算一个二维 micro tile，通过 `regM[TM]` 和 `regN[TN]` 做外积。
