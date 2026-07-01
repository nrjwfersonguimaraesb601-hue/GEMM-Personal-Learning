# 1D Blocktiling Kernel

这个目录记录的是我在 SMEM 之后继续往前走的一步：

- 一个 thread 不再只算一个输出
- 一个 block 负责一个更大的输出 tile
- 每个 thread 用寄存器数组累加多个结果

这版对我来说，已经不只是“思路走通”，而是第一次真正把性能又往上推了一大截。

## 文件说明

- `1D_Blocktiling_kernel.cu`: correctness-first 版本
- `1D_Blocktiling_kernel_benchmark.cu`: benchmark 版本
- `main_gpu_kernel.cu`: 保留下来的核心思路草稿
- `profiling/`: Nsight Compute 截图、命令模板和逐图解读

## 当前配置

- `BM = 64`
- `BN = 64`
- `BK = 8`
- `TM = 8`
- `blockDim = (512, 1)`

简单理解就是：

- 一个 block 负责 `64 x 64` 的输出 tile
- 只用 `512` 个线程
- 每个线程负责同一列上的 `8` 个输出

## 编译

```bash
nvcc -O3 1D_Blocktiling_kernel.cu -o 1D_Blocktiling_kernel
nvcc -O3 1D_Blocktiling_kernel_benchmark.cu -o 1D_Blocktiling_kernel_benchmark
```

## 使用方式

```bash
./1D_Blocktiling_kernel
./1D_Blocktiling_kernel_benchmark
./1D_Blocktiling_kernel_benchmark 1024 1024 1024
./1D_Blocktiling_kernel_benchmark 4096 4096 4096 --no-check
```

## 这次 benchmark 设置

- GPU: `NVIDIA GeForce RTX 4060 Laptop GPU`
- Warmup: `10`
- Iterations: `50`
- Block: `512 x 1`
- CPU check: `disabled`

## 实测结果

| M | N | K | Block | Avg (ms) | Avg GFLOPS | Best GFLOPS |
|---|---|---|---|---:|---:|---:|
| 256 | 256 | 256 | 512x1 | 0.0267 | 1258.8250 | 1489.4545 |
| 512 | 512 | 512 | 512x1 | 0.1003 | 2676.5946 | 2880.7032 |
| 1024 | 1024 | 1024 | 512x1 | 0.6321 | 3397.1641 | 3426.7189 |
| 2048 | 2048 | 2048 | 512x1 | 4.6454 | 3698.2388 | 4033.1968 |
| 4096 | 4096 | 4096 | 512x1 | 37.5543 | 3659.7348 | 3708.4916 |
| 1023 | 1023 | 1023 | 512x1 | 0.5778 | 3705.5630 | 3788.0688 |
| 4096 | 256 | 4096 | 512x1 | 2.3885 | 3596.3530 | 3628.2908 |
| 256 | 4096 | 4096 | 512x1 | 2.4800 | 3463.6959 | 3486.5371 |

## 相对前三版

| Case | Naive | Coalesced | SMEM | 1D Blocktiling |
|---|---:|---:|---:|---:|
| 256 x 256 x 256 | 92.2776 | 616.2947 | 640.7902 | 1258.8250 |
| 512 x 512 x 512 | 109.5825 | 830.3416 | 836.6070 | 2676.5946 |
| 1024 x 1024 x 1024 | 123.4315 | 886.0027 | 904.6960 | 3397.1641 |
| 2048 x 2048 x 2048 | 123.5758 | 1026.0384 | 1039.4091 | 3698.2388 |
| 4096 x 4096 x 4096 | 123.0080 | 870.8718 | 958.3544 | 3659.7348 |

## 这组结果怎么看

- 这次和前面三版已经明显拉开了
- square case 基本站在 `~1.26-3.70 TFLOPS`
- 从 `1024` 开始，已经稳定进入 `3 TFLOPS+`

这说明这版的关键收益是真实的：

- 一个 thread 计算多个输出值是值得的
- register blocking 确实把数据复用进一步放大了
- block tiling 不只是“结构更复杂”，而是真的把吞吐往上推了

经验上看，这版最该记住的一句话是：

- shared memory 把路铺好以后，继续让每个 thread 多算几个结果，性能提升会非常明显

## 当前结论

这版现在已经是当前项目里最强的一版：

- 比 SMEM 再快一个明显台阶
- 比 naive 快得非常多
- 已经可以作为后续继续做 `2D block tiling`、`vectorized load`、`warp tiling` 的新起点
