# Shared Memory Kernel

这一版是在 coalesced kernel 的基础上继续往前走的一步：

- 先把 `A` 和 `B` 的 tile 搬进 shared memory
- 再让 block 内线程重复使用这份 tile

对我来说，这版最大的意义不是“概念上用了 shared memory”，而是它已经在当前机器上带来了比较稳定的额外收益。

## 文件说明

- `My_SMEM_kernel.cu`: correctness-first 版本
- `My_SMEM_kernel_benchmark.cu`: benchmark 版本
- `teacher.md`: 参考代码解释

## 编译

```bash
nvcc -O3 My_SMEM_kernel.cu -o My_SMEM_kernel
nvcc -O3 My_SMEM_kernel_benchmark.cu -o My_SMEM_kernel_benchmark
```

## 使用方式

```bash
./My_SMEM_kernel
./My_SMEM_kernel_benchmark
./My_SMEM_kernel_benchmark 1024 1024 1024
./My_SMEM_kernel_benchmark 4096 4096 4096 --no-check
```

## 这次 benchmark 设置

- GPU: `NVIDIA GeForce RTX 4060 Laptop GPU`
- Warmup: `10`
- Iterations: `50`
- Block: `32 x 32`
- CPU check: `disabled`

## 实测结果

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

## 相对前两版

| Case | Naive | Coalesced | SMEM |
|---|---:|---:|---:|
| 256 x 256 x 256 | 92.2776 | 616.2947 | 640.7902 |
| 512 x 512 x 512 | 109.5825 | 830.3416 | 836.6070 |
| 1024 x 1024 x 1024 | 123.4315 | 886.0027 | 904.6960 |
| 2048 x 2048 x 2048 | 123.5758 | 1026.0384 | 1039.4091 |
| 4096 x 4096 x 4096 | 123.0080 | 870.8718 | 958.3544 |

## 这组结果怎么看

- 相对 naive，提升仍然非常大
- 相对 coalesced，这次已经不是“基本打平”，而是多数 square case 都有小幅领先
- `4096` 这档从 `870.8718` 提到 `958.3544 GFLOPS`，这次 shared memory 的收益就更明确了

但经验上也不用把这版想得太神：

- 它不是一下子跳到一个完全不同的数量级
- 它更像是在 coalesced 基础上继续往上推了一截
- 说明 shared memory 这一步已经值得保留，但后面还需要更深的 tiling / register reuse 才能继续拉开差距

## 当前结论

这版现在的定位我会这样记：

- 它已经是一个比较成熟的 `~0.9-1.0 TFLOPS` 档学习版本
- 在这台 RTX 4060 Laptop GPU 上，已经整体强于当前 coalesced 版
- 它很适合作为后续 block tiling 和 register blocking 的起点
