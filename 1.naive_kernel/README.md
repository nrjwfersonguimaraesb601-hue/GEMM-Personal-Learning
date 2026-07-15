# Naive Kernel

## Idea

这是整个项目的 baseline。

这一版保持最基础的 SGEMM 写法：

- 一个 CUDA thread 计算一个 `C[row][col]`
- 每个 thread 直接从 global memory 读取 `A[row][k]` 和 `B[k][col]`
- 不使用 shared memory
- 不做 register tiling
- 保留 non-coalesced 的访问特征，方便后续版本对比

它的目标不是快，而是提供一个稳定、容易验证的起点。

## Build

```bash
nvcc My_naive_kernel.cu -O3 -o My_naive_kernel
nvcc My_naive_kernel_benchmark.cu -O3 -lineinfo -o naive_bench
```

## Run Benchmark

```bash
./naive_bench \
  --warmup 10 --iters 50 --bx 32 --by 32 --no-check
```

当前性能表使用 `--no-check`，主要用于 benchmark 性能记录。正确性需要单独去掉 `--no-check` 开启 CPU check，或运行 correctness-first 版本。

## Benchmark Result

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

## What This Stage Shows

- `1024^3` 之后的 square case 基本稳定在 `~123 GFLOPS`
- 这说明 baseline 足够稳定，可以作为后续优化的对照
- `1023^3` 的吞吐明显更高，更适合作为 shape-sensitive 观察，不作为主 baseline 结论

## Known Issues / Notes

- global memory 访问没有优化，warp 内访存不够合并
- 没有 shared memory reuse
- 没有 register blocking
- 这是故意保留的低性能 baseline，不应该继续复杂化

## Next Step

下一阶段是 `Global Memory Coalescing`：先不改变算法结构，只调整 thread mapping，让 global memory access 更连续。
