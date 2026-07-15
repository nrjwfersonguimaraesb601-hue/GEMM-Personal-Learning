# Performance Summary

这份文件统一记录 CUDA GEMM 学习项目中各阶段 kernel 的 benchmark 结果。

## Hardware

- GPU: `NVIDIA GeForce RTX 4060 Laptop GPU`
- Compute Capability: `8.9`
- Global Memory: `8.00 GiB`
- SM Count: `24`
- Max Threads Per Block: `1024`

## Benchmark Notes

- Warmup: `10`
- Iterations: `50`
- 使用 CUDA event 统计 kernel latency，不包含 H2D / D2H 和 CPU reference 时间
- Stage 1–5 的历史性能表来自 `--no-check` 运行
- Stage 6 Vectorized 的本次运行启用了 CPU check，7 个 case 全部 `PASS`
- CPU check 位于计时区间之外，因此 Stage 6 的 kernel 数据仍可与历史结果比较

Laptop GPU 会受功耗、温度和频率变化影响，表格主要用于观察优化趋势，而不是
作为跨机器的绝对性能结论。

## Benchmark Commands

```bash
# Stage 1: Naive
./naive_bench --warmup 10 --iters 50 --bx 32 --by 32 --no-check

# Stage 2: Global memory coalescing
./gmemc_bench --warmup 10 --iters 50 --bx 32 --by 32 --no-check

# Stage 3: Shared memory tiling
./smem_bench --warmup 10 --iters 50 --bx 32 --by 32 --no-check

# Stage 4: 1D register tiling
./1D_Blocktiling_bench --warmup 10 --iters 50 --bx 512 --by 1 --no-check

# Stage 5: 2D register tiling
./2D_Blocktiling_bench --warmup 10 --iters 50 --bx 64 --by 1 --no-check

# Stage 6: Vectorized memory access, correctness enabled
./Vectorize_bench --warmup 10 --iters 50 --bx 64 --by 1
```

这些命令应分别在对应的 kernel 目录中运行。

## Kernel Performance Table

统一比较 Avg GFLOPS：

| Case              |    Naive | Coalesced |      SMEM | 1D Register | 2D Register | Vectorized |
| ----------------- | -------: | --------: | --------: | ----------: | ----------: | ---------: |
| `256^3`           |  92.2776 |  616.2947 |  640.7902 |   1258.8250 |    801.0145 |  1335.4937 |
| `512^3`           | 109.5825 |  830.3416 |  836.6070 |   2676.5946 |   2431.5510 |  4243.2712 |
| `1024^3`          | 123.4315 |  886.0027 |  904.6960 |   3397.1641 |   3395.6509 |  6412.4137 |
| `2048^3`          | 123.5758 | 1026.0384 | 1039.4091 |   3698.2388 |   4054.4146 |  7290.9726 |
| `4096^3`          | 123.0080 |  870.8718 |  958.3544 |   3659.7348 |   4505.6055 |  7802.2829 |
| `1023^3`          | 407.9408 |  941.6104 |  961.0163 |   3705.5630 |   3768.9978 |          — |
| `4096x256x4096`   | 116.5439 | 1013.2102 |  941.9762 |   3596.3530 |   4121.9961 |  6726.1882 |
| `256x4096x4096`   | 123.6838 |  879.4382 |  936.1697 |   3463.6959 |   3904.3388 |  6447.6574 |

Vectorized kernel 不支持 `1023^3`：当前实现没有边界分支，并要求
`M % 64 == 0`、`N % 64 == 0`、`K % 8 == 0`。

## Relative to Previous Kernel

相对上一阶段的 Avg GFLOPS 变化：

| Case              | Coalesced vs Naive | SMEM vs Coalesced | 1D vs SMEM | 2D vs 1D | Vectorized vs 2D |
| ----------------- | -----------------: | ----------------: | ---------: | -------: | ---------------: |
| `256^3`           |            +567.9% |             +4.0% |     +96.4% |   -36.4% |           +66.7% |
| `512^3`           |            +657.7% |             +0.8% |    +219.9% |    -9.2% |           +74.5% |
| `1024^3`          |            +617.8% |             +2.1% |    +275.5% |   -0.04% |           +88.8% |
| `2048^3`          |            +730.3% |             +1.3% |    +255.8% |    +9.6% |           +79.8% |
| `4096^3`          |            +608.0% |            +10.0% |    +281.9% |   +23.1% |           +73.2% |
| `1023^3`          |            +130.8% |             +2.1% |    +285.6% |    +1.7% |                — |
| `4096x256x4096`   |            +769.4% |             -7.0% |    +281.8% |   +14.6% |           +63.2% |
| `256x4096x4096`   |            +611.0% |             +6.5% |    +270.0% |   +12.7% |           +65.1% |

## Vectorized Detailed Result

| Case              | Check | Min (ms) | Avg (ms) | Max (ms) | Avg GFLOPS | Best GFLOPS | Max Error |
| ----------------- | :---: | -------: | -------: | -------: | ----------: | -----------: | --------: |
| `256^3`           | PASS  |   0.0225 |   0.0251 |   0.0307 |   1335.4937 |    1489.4545 |    0.0000 |
| `512^3`           | PASS  |   0.0594 |   0.0633 |   0.0819 |   4243.2712 |    4519.7240 |    0.0000 |
| `1024^3`          | PASS  |   0.3154 |   0.3349 |   0.4536 |   6412.4137 |    6808.9353 |    0.0001 |
| `2048^3`          | PASS  |   2.3040 |   2.3563 |   2.7545 |   7290.9726 |    7456.5408 |    0.0002 |
| `4096^3`          | PASS  |  16.4045 |  17.6152 |  20.0703 |   7802.2829 |    8378.1353 |    0.0004 |
| `4096x256x4096`   | PASS  |   1.2493 |   1.2771 |   1.4950 |   6726.1882 |    6875.9083 |    0.0004 |
| `256x4096x4096`   | PASS  |   1.3015 |   1.3323 |   1.5400 |   6447.6574 |    6600.0062 |    0.0004 |

## Stage Summary

| Stage              | Main Change                                 | `4096^3` Avg Result |
| ------------------ | ------------------------------------------- | ------------------: |
| Naive              | direct global-memory implementation         |    `0.123 TFLOPS` |
| Coalesced          | coalesced global-memory access              |    `0.871 TFLOPS` |
| SMEM               | shared-memory tile reuse                    |    `0.958 TFLOPS` |
| 1D Register Tiling | each thread computes `TM` results           |    `3.660 TFLOPS` |
| 2D Register Tiling | each thread computes a `TM x TN` micro tile |    `4.506 TFLOPS` |
| Vectorized         | `float4` load/store and transposed A tile   |    `7.802 TFLOPS` |

## Vectorized vs 2D Notes

- 所有共同合法尺寸均有提升，范围为 `+63.2%` 到 `+88.8%`
- `1024^3` 提升最大：`3395.65 -> 6412.41 GFLOPS`
- `4096^3` 提升 `73.2%`，Avg latency 从 `30.5040 ms` 降到 `17.6152 ms`
- `4096^3` 相对 Naive 已达到约 `63.4x` 的 Avg GFLOPS
- 最大矩阵误差为 `0.0004`，全部 correctness check 为 `PASS`

当前结果说明向量化访存与 A tile 的 shared-memory 转置非常有效。Nsight
Compute 进一步显示 global-store sector 利用率已经从 2D 版本的 `4/32 B`
提升到 `16/32 B`，Compute Throughput 从 `42.63%` 提升到 `54.20%`；但
shared load/store bank conflict 仍是下一步的主要优化方向。

## Notes

- 数据用于学习和趋势观察，不代表工业级最终性能
- 当前项目不声明达到或替代 cuBLAS
- Vectorized 结果与旧阶段来自不同测试轮次，Laptop GPU 状态会带来一定波动
- 对 Vectorized kernel，不应使用不满足 tile/对齐约束的尺寸做 correctness 结论

## Source READMEs

- [Stage 1: Naive](./1.naive_kernel/README.md)
- [Stage 2: Global memory coalescing](./2.Global_Memory_Coalescing_kernel/README.md)
- [Stage 3: Shared memory tiling](./3.SMEM_kernel/README.md)
- [Stage 4: 1D register tiling](./4.1D_Blocktiling_kernel/README.md)
- [Stage 5: 2D register tiling](./5.2D_Blocktiling_kernel/README.md)
- [Stage 6: Vectorized memory access](./6.Vectorize_kernel/README.md)
