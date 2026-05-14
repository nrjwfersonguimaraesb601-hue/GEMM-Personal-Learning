# Naive CUDA GEMM

这个目录记录了我最基础的一版 CUDA GEMM 实现：

`C = A x B`

这一版的目标不是马上追 cuBLAS，而是先建立一个清晰、可验证、可复现的 baseline。它主要用来：

- 验证手写 CUDA GEMM kernel 的正确性
- 建立一套可重复的 benchmark 流程
- 观察 naive kernel 的性能上限和瓶颈
- 为后续的 tiling、shared memory、vectorized load 等优化提供对照组

## 文件说明

- `My_naive_kernel.cu`: 偏 correctness-first 的版本，先保证实现正确
- `My_naive_kernel_benchmark.cu`: 偏 benchmark-oriented 的版本，支持命令行参数和统计输出
- `My_naive_kernel`: 由 correctness 版本编译得到的可执行文件
- `My_naive_kernel_benchmark`: 由 benchmark 版本编译得到的可执行文件

## Kernel 概述

这个目录里的 kernel 是一个非常典型的 naive GEMM kernel：

- 一个 thread 负责计算一个输出元素 `C[row][col]`
- 每个 thread 都会完整遍历一次 `K` 维
- accumulation 过程直接从 global memory 读取 `A` 和 `B`
- 没有使用 shared memory tiling
- 没有使用 register blocking
- 没有使用 vectorized load/store
- 没有做 warp-level optimization

这样的实现优点是代码直观、容易验证；缺点也很明显：`A` 和 `B` 中的数据会被大量重复从 global memory 加载，数据复用很差，因此整体会非常受 memory traffic 限制，无法充分发挥 RTX 4060 的真实算力。

## Benchmark 版本说明

`My_naive_kernel_benchmark.cu` 目前已经升级成更适合测速的版本，支持：

- 单组 case 运行，例如 `M N K`
- 默认 benchmark suite，覆盖 square case、boundary case、rectangular case
- 可配置 `warmup`、`iters`、`blockDim.x`、`blockDim.y`
- 输出 `min / avg / max` kernel time
- 输出 `avg GFLOPS` 和 `best GFLOPS`
- 可选 CPU correctness check
- 对大矩阵自动跳过 CPU 全量校验
- 支持 CSV 输出，方便后续画图或导入表格

## 编译

```bash
nvcc -O3 My_naive_kernel.cu -o My_naive_kernel
nvcc -O3 My_naive_kernel_benchmark.cu -o My_naive_kernel_benchmark
```

## 使用方式

运行默认 benchmark suite：

```bash
./My_naive_kernel_benchmark
```

运行单个指定 case：

```bash
./My_naive_kernel_benchmark 1024 1024 1024
```

调整 timing 或 launch configuration：

```bash
./My_naive_kernel_benchmark 4096 4096 4096 --iters 50 --no-check
./My_naive_kernel_benchmark 2048 2048 2048 --bx 32 --by 8
./My_naive_kernel_benchmark --csv
```

## 测试环境

下面这组 benchmark 结果采集自：

- GPU: `NVIDIA GeForce RTX 4060 Laptop GPU`
- Compute capability: `8.9`
- Global memory: `8.00 GiB`
- SM count: `24`
- 本次测试使用的 block size: `16 x 16`

## 实测结果

下面的数据来自当前 benchmark 版本：

| M | N | K | Block | Check | Min (ms) | Avg (ms) | Max (ms) | Avg GFLOPS | Best GFLOPS | Max Error | Note |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| 256 | 256 | 256 | 16x16 | PASS | 0.3123 | 0.3234 | 0.3850 | 103.7455 | 107.4361 | 0.0000 | |
| 512 | 512 | 512 | 16x16 | PASS | 2.2241 | 2.4632 | 2.8109 | 108.9805 | 120.6924 | 0.0000 | |
| 1024 | 1024 | 1024 | 16x16 | PASS | 19.0136 | 20.4585 | 22.2915 | 104.9677 | 112.9444 | 0.0000 | |
| 2048 | 2048 | 2048 | 16x16 | PASS | 158.8869 | 161.3996 | 163.8134 | 106.4431 | 108.1264 | 0.0000 | |
| 4096 | 4096 | 4096 | 16x16 | SKIP | 1719.2428 | 1759.4509 | 1906.3910 | 78.1147 | 79.9416 | 0.0000 | CPU check skipped |

## 性能分析

### 1. Correctness

对于 `256`、`512`、`1024` 和 `2048` 这几组 case，benchmark 输出都是 `PASS`，并且 `max error = 0.0000`。这说明这版 naive kernel 在当前测试范围内功能上是正确的，至少对这些单精度 float 的 square case 没有出现明显误差。

`4096` 这组被标成 `SKIP`，不是因为 GPU 算错了，而是 benchmark 逻辑主动跳过了 CPU full-check。原因很简单：矩阵太大时，CPU 参考实现的代价会非常高，已经不适合继续作为常规 benchmark 的一部分。

### 2. Scaling Behavior

从 `256` 到 `2048`，整体吞吐基本稳定在 `104-109 GFLOPS` 左右。这个现象很关键，因为它说明了几件事：

- 运行时间基本符合 `O(MNK)` 的增长趋势
- 随着矩阵变大，launch overhead 不再是主导因素
- kernel 性能进入了一个比较稳定的平台期

也就是说，这个 naive kernel 虽然不快，但它的表现是稳定、可解释的，不是“偶然跑快”或者“随机波动很大”的状态。

### 3. 为什么 4096 会掉速

在 `4096 x 4096 x 4096` 这个 case 上，吞吐下降到了大约 `78 GFLOPS`，明显低于前面的平台期。这个掉速非常符合 naive GEMM 的预期，主要原因包括：

- kernel 会重复从 global memory 读取 `A` 和 `B`
- 每个 thread 都要执行很长的标量累加循环
- 没有通过 shared memory 做数据复用
- 工作集变大后，cache reuse 会进一步变差
- kernel 运行时间变长后，memory system 的压力会更明显

换句话说，矩阵一旦大到一定程度，global memory access pattern 的低效就会越来越突出，算术运算本身反而不是最主要的问题。

### 4. 这个结果说明了什么

这版实现已经是一个合格的学习型 baseline：

- 实现简单
- 正确性清楚
- 性能趋势稳定
- 很适合作为后续优化前的对照组

但它也非常典型地暴露了 naive GEMM 的上限：

- arithmetic intensity 不高
- global memory fetch 重复很多
- 数据复用差
- 没有利用现代 GPU memory hierarchy

对 RTX 4060 Laptop GPU 来说，`~100 GFLOPS` 左右作为 naive baseline 是合理的，但和真正经过优化的 GEMM kernel 相比，差距会非常大。

### 5. 当前测速质量如何

这份 benchmark 测的是 pure kernel time，而不是端到端时间，这一点很重要：

- `Host to Device` 拷贝不计入计时
- `Device to Host` 拷贝不计入计时
- 正式测速前有 warmup
- 计时不是单次，而是多次迭代
- 输出的不只是平均值，还有 `min / avg / max`

因此这组结果已经比“单次跑一下然后打印一个时间”要可靠很多，适合后续拿来和优化版 kernel 做横向比较。

## 当前局限

虽然这版已经能做比较规范的 benchmark，但它仍然只是 baseline，还有不少局限：

- 目前只测了一种 kernel 实现
- 还没有加入 cuBLAS baseline
- 还没有 shared memory tiling
- 还没有系统比较不同 block size
- 还没有 occupancy 或 memory bandwidth 分析
- 大尺寸正确性目前是跳过，而不是改成 cuBLAS 对比或抽样校验

## 下一步建议

如果继续往下做，这几个方向最值得优先推进：

1. 加一个 tiled shared-memory GEMM kernel
2. 系统比较 `16x16`、`32x8`、`32x16` 等 block 配置
3. 加入 cuBLAS baseline
4. 补充 rectangular matrix 的系统测试
5. 画出 `matrix size vs GFLOPS` 和 `matrix size vs time` 曲线
6. 对大矩阵改成 cuBLAS 校验或者 sampled verification

## 总结

这个目录现在已经具备了一个 CUDA GEMM 学习项目应有的基本形态：

- 一个 correctness-first 的 naive kernel
- 一个 benchmark-oriented 的测速程序
- 一组在 RTX 4060 Laptop GPU 上得到的真实 baseline 数据

从当前结果看，结论很明确：

- kernel 是正确的
- 在中等规模上性能比较稳定
- 性能瓶颈非常符合 naive global-memory GEMM 的典型特征

后续如果引入 shared memory tiling、better block mapping、cuBLAS baseline，这个项目的分析深度和展示效果都会再上一个台阶。
