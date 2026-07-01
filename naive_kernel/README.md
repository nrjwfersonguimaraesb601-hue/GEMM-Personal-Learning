# Naive CUDA GEMM

这个目录保留的是最原始的 baseline。

它的定位很明确：

- 一个 thread 计算一个输出元素
- 不做 shared memory
- 不做 register blocking
- 保留当前这版 non-coalesced 的线程映射

这版最大的价值不是快，而是给后面的 coalescing、SMEM、block tiling 提供一个稳定对照组。

## 文件说明

- `My_naive_kernel.cu`: correctness-first 版本
- `My_naive_kernel_benchmark.cu`: benchmark 版本

## 编译

```bash
nvcc -O3 My_naive_kernel.cu -o My_naive_kernel
nvcc -O3 My_naive_kernel_benchmark.cu -o My_naive_kernel_benchmark
```

## 使用方式

```bash
./My_naive_kernel
./My_naive_kernel_benchmark
./My_naive_kernel_benchmark 1024 1024 1024
./My_naive_kernel_benchmark 4096 4096 4096 --no-check
```

## 这次 benchmark 设置

- GPU: `NVIDIA GeForce RTX 4060 Laptop GPU`
- Compute capability: `8.9`
- Global memory: `8.00 GiB`
- SM count: `24`
- Max threads per block: `1024`
- Warmup: `10`
- Iterations: `50`
- Block: `32 x 32`
- CPU check: `disabled`

说明：

- 下面这组数据是纯测速结果，所以 `check` 都是 `SKIP`
- correctness 还是由单独的 correctness 版本负责确认

## 实测结果

| M | N | K | Block | Avg (ms) | Avg GFLOPS | Best GFLOPS |
|---|---|---|---|---:|---:|---:|
| 256 | 256 | 256 | 32x32 | 0.3636 | 92.2776 | 95.2558 |
| 512 | 512 | 512 | 32x32 | 2.4496 | 109.5825 | 120.4706 |
| 1024 | 1024 | 1024 | 32x32 | 17.3982 | 123.4315 | 123.9250 |
| 2048 | 2048 | 2048 | 32x32 | 139.0230 | 123.5758 | 124.6061 |
| 4096 | 4096 | 4096 | 32x32 | 1117.3169 | 123.0080 | 124.7219 |
| 1023 | 1023 | 1023 | 32x32 | 5.2488 | 407.9408 | 415.2957 |
| 4096 | 256 | 4096 | 32x32 | 73.7055 | 116.5439 | 123.7789 |
| 256 | 4096 | 4096 | 32x32 | 69.4507 | 123.6838 | 124.1653 |

## 这组结果怎么看

- 常见 square case 已经很稳定，基本落在 `~92-123 GFLOPS`
- 从 `1024` 往后看，square case 平台期大致就在 `~123 GFLOPS`
- 这说明这版 baseline 已经够稳定，适合拿来做后续 speedup 对照

经验上看，这组数据也很符合 naive kernel 的定位：

- 实现简单
- 正确性容易验证
- 性能不高，但趋势很清楚

`1023` 和两组 rectangular case 看起来更快，不要把它们当作 square case 的主结论。对我来说，这几组更适合当补充观察，用来提醒自己 shape 会影响吞吐表现。

## 当前结论

这版 naive kernel 现在可以作为项目里的 baseline：

- square baseline 大约是 `~123 GFLOPS`
- 后面所有优化版都应该和它做对比
- 它不是项目重点，但它是后面所有结论的起点
