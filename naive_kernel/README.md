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
- 当前线程映射刻意做成了 non-coalesced baseline
- 没有使用 shared memory tiling
- 没有使用 register blocking
- 没有使用 vectorized load/store
- 没有做 warp-level optimization

这里当前保留的是一个“故意不做 coalescing”的对照版本：

- warp 内通常是 `row` 连续、`col` 固定
- `A[row * K + k]` 是跨步访问
- `C[row * N + col]` 写回时也是跨步访问
- `B[k * N + col]` 则更接近同地址重复读取

这样的实现优点是代码直观、容易验证，而且非常适合作为后续 coalesced / tiled 版本的对照组；缺点也很明显：global memory 访存模式不友好，`A` 和 `B` 中的数据也会被大量重复读取，数据复用很差，因此整体仍然无法充分发挥 RTX 4060 的真实算力。

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
- Max threads per block: `1024`
- Warmup: `10`
- Iterations: `100`
- 本次测试使用的 block size: `16 x 16`
- CPU check: `enabled`, `max check dim = 2048`

## 实测结果

下面的数据来自当前 benchmark 版本：

| M | N | K | Block | Check | Min (ms) | Avg (ms) | Max (ms) | Avg GFLOPS | Best GFLOPS | Max Error | Note |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| 256 | 256 | 256 | 16x16 | PASS | 0.1751 | 0.1830 | 0.2099 | 183.3508 | 191.6257 | 0.0000 | |
| 512 | 512 | 512 | 16x16 | PASS | 1.1346 | 1.3963 | 2.0060 | 192.2429 | 236.5920 | 0.0000 | |
| 1024 | 1024 | 1024 | 16x16 | PASS | 8.8422 | 9.3365 | 10.2922 | 230.0084 | 242.8665 | 0.0001 | |
| 2048 | 2048 | 2048 | 16x16 | PASS | 70.4645 | 71.1545 | 77.6356 | 241.4446 | 243.8088 | 0.0002 | |
| 4096 | 4096 | 4096 | 16x16 | SKIP | 560.2662 | 575.8431 | 699.9061 | 238.6743 | 245.3101 | 0.0000 | CPU check skipped |
| 1023 | 1023 | 1023 | 16x16 | PASS | 3.0863 | 3.2389 | 3.7100 | 661.0822 | 693.7671 | 0.0001 | boundary case |
| 4096 | 256 | 4096 | 16x16 | SKIP | 2.2906 | 2.5852 | 3.1436 | 3322.7325 | 3750.0415 | 0.0000 | CPU check skipped |
| 256 | 4096 | 4096 | 16x16 | SKIP | 2.4812 | 2.5929 | 3.1119 | 3312.8442 | 3462.0750 | 0.0000 | CPU check skipped |

## 性能分析

### 1. Correctness

对于 `256`、`512`、`1024`、`1023` 和 `2048` 这几组 case，benchmark 输出都是 `PASS`，`max error` 也都控制在 `1e-4` 量级。这说明这版 non-coalesced naive kernel 在当前测试范围内功能上是正确的，至少对这些单精度 float case 没有出现明显误差。

`4096` 和两组 rectangular case 被标成 `SKIP`，不是因为 GPU 算错了，而是 benchmark 逻辑主动跳过了 CPU full-check。原因很简单：矩阵太大时，CPU 参考实现的代价会非常高，已经不适合继续作为常规 benchmark 的一部分。

### 2. Scaling Behavior

从 `256` 到 `2048`，square case 的吞吐基本稳定在 `183-241 GFLOPS` 左右，而且矩阵越大越接近一个比较稳定的平台期。这个现象说明了几件事：

- 运行时间基本符合 `O(MNK)` 的增长趋势
- 随着矩阵变大，launch overhead 不再是主导因素
- 当前 non-coalesced baseline 的 steady-state 表现大约落在 `230-240 GFLOPS`

也就是说，这个 naive kernel 虽然仍然很朴素，但它的表现已经是稳定、可解释的，不是“偶然跑快”或者“随机波动很大”的状态。

### 3. 为什么这次 4096 没有像旧结果那样明显掉速

在这次结果里，`4096 x 4096 x 4096` 的平均吞吐仍然维持在大约 `239 GFLOPS`，和 `1024`、`2048` 非常接近，没有出现旧版 README 里那种明显掉到 `~78 GFLOPS` 的情况。这意味着原先 README 记录的那组数据已经不再代表当前代码。

从当前结果看，更合理的结论是：

- 这版 kernel 的 square-case 平台期比原先记录的更高
- 当前 non-coalesced 映射虽然访存不友好，但并没有在 `4096` 这一档立刻出现断崖式掉速
- 因此后续做 coalesced kernel 时，应该拿这组 `~230-240 GFLOPS` 的数据作为新的 naive baseline，而不是继续沿用旧结果

### 4. Boundary case 和 rectangular case 要怎么理解

`1023 x 1023 x 1023` 这组结果的平均吞吐达到了 `661 GFLOPS`，两组 rectangular case 甚至都超过了 `3.3 TFLOPS`。这些结果说明当前 kernel 对 shape 非常敏感，因此：

- square case 更适合作为后续优化前后的主 baseline
- boundary case 和 rectangular case 更适合作为补充观察，帮助看 shape sensitivity
- 这几组高吞吐结果不应该简单拿来和 square case 直接横向比较

换句话说，这份 benchmark 现在不仅是在测一个“慢的 naive kernel”，也开始暴露这个 kernel 在不同 shape 下的行为差异。

### 5. 这个结果说明了什么

这版实现已经是一个合格的学习型 baseline：

- 实现简单
- 正确性清楚
- 性能趋势稳定
- 很适合作为后续优化前的对照组

但它也非常典型地暴露了 naive GEMM 的上限：

- 当前版本仍然没有 coalesced global-memory access
- global memory fetch 重复很多
- 数据复用差
- 没有利用 shared memory 和更深层的 memory hierarchy

对 RTX 4060 Laptop GPU 来说，当前这版 non-coalesced naive baseline 更接近 `~230-240 GFLOPS` 的 square-case 平台，而不是旧 README 里写的 `~100 GFLOPS`。后续如果引入 coalescing、shared memory tiling、vectorized access，这个 baseline 仍然有很大的提升空间。

### 6. 当前测速质量如何

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
- 一组在 RTX 4060 Laptop GPU 上得到的真实 non-coalesced baseline 数据

从当前结果看，结论很明确：

- kernel 是正确的
- square case 在 `~230-240 GFLOPS` 附近比较稳定
- 当前代码和旧 README 已经不是同一版 baseline，后续对比应以这组新数据为准

后续如果引入 shared memory tiling、better block mapping、cuBLAS baseline，这个项目的分析深度和展示效果都会再上一个台阶。
