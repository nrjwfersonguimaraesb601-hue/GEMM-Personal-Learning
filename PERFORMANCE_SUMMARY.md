# Performance Summary

这份文件统一记录当前 CUDA GEMM 学习项目中各阶段 kernel 的 benchmark 结果。

## Hardware

- GPU: `NVIDIA GeForce RTX 4060 Laptop GPU`
- Compute Capability: `8.9`
- Global Memory: `8.00 GiB`
- SM Count: `24`
- Max Threads Per Block: `1024`

## Benchmark Notes

性能表主要用于观察优化趋势：

- Warmup: `10`
- Iterations: `50`
- CPU check: `disabled`
- benchmark 使用 `--no-check`
- 表格记录的是 kernel benchmark 时间，不包含 H2D / D2H 拷贝

正确性需要单独开启 CPU check，或运行各目录中的 correctness-first 版本。

## Benchmark Commands

Naive:

```bash
./naive_bench --warmup 10 --iters 50 --bx 32 --by 32 --no-check
```

Global Memory Coalescing:

```bash
./gmemc_bench --warmup 10 --iters 50 --bx 32 --by 32 --no-check
```

Shared Memory Tiling:

```bash
./smem_bench --warmup 10 --iters 50 --bx 32 --by 32 --no-check
```

1D Register Tiling:

```bash
./1D_Blocktiling_bench --warmup 10 --iters 50 --bx 512 --by 1 --no-check
```

2D Register Tiling:

```bash
./2D_Blocktiling_bench --warmup 10 --iters 50 --bx 64 --by 1 --no-check
```

## Kernel Performance Table

Avg GFLOPS:

| Case | Naive | Coalesced | SMEM | 1D Register Tiling | 2D Register Tiling |
|---|---:|---:|---:|---:|---:|
| 256^3 | 92.2776 | 616.2947 | 640.7902 | 1258.8250 | 801.0145 |
| 512^3 | 109.5825 | 830.3416 | 836.6070 | 2676.5946 | 2431.5510 |
| 1024^3 | 123.4315 | 886.0027 | 904.6960 | 3397.1641 | 3395.6509 |
| 2048^3 | 123.5758 | 1026.0384 | 1039.4091 | 3698.2388 | 4054.4146 |
| 4096^3 | 123.0080 | 870.8718 | 958.3544 | 3659.7348 | 4505.6055 |
| 1023^3 | 407.9408 | 941.6104 | 961.0163 | 3705.5630 | 3768.9978 |
| 4096 x 256 x 4096 | 116.5439 | 1013.2102 | 941.9762 | 3596.3530 | 4121.9961 |
| 256 x 4096 x 4096 | 123.6838 | 879.4382 | 936.1697 | 3463.6959 | 3904.3388 |

## Relative to Previous Kernel

相对上一阶段的 Avg GFLOPS 变化：

| Case | Coalesced vs Naive | SMEM vs Coalesced | 1D vs SMEM | 2D vs 1D |
|---|---:|---:|---:|---:|
| 256^3 | +567.9% | +4.0% | +96.4% | -36.4% |
| 512^3 | +657.7% | +0.8% | +219.9% | -9.2% |
| 1024^3 | +617.8% | +2.1% | +275.5% | -0.04% |
| 2048^3 | +730.3% | +1.3% | +255.8% | +9.6% |
| 4096^3 | +608.0% | +10.0% | +281.9% | +23.1% |
| 1023^3 | +130.8% | +2.1% | +285.6% | +1.7% |
| 4096 x 256 x 4096 | +769.4% | -7.0% | +281.8% | +14.6% |
| 256 x 4096 x 4096 | +611.0% | +6.5% | +270.0% | +12.7% |

## Stage Summary

| Stage | Main Change | Representative Result |
|---|---|---|
| Naive | non-coalesced baseline | square baseline about `~123 GFLOPS` |
| Coalesced | better global memory access pattern | square cases reach `~616-1026 GFLOPS` |
| SMEM | shared memory tile reuse | modest but stable improvement over coalesced |
| 1D Register Tiling | each thread computes `TM` results | `4096^3`: `3659.7348 GFLOPS` |
| 2D Register Tiling | each thread computes `TM x TN` results | `4096^3`: `4505.6055 GFLOPS` |

## 2D vs 1D Notes

2D Register Tiling 的结果比较有分界感：

- `256^3`: `-36.4%`
- `512^3`: `-9.2%`
- `1024^3`: approximately flat, `-0.04%`
- `2048^3`: `+9.6%`
- `4096^3`: `+23.1%`
- `4096 x 256 x 4096`: `+14.6%`
- `256 x 4096 x 4096`: `+12.7%`

这说明当前 2D 基础实现已经在大矩阵上体现出收益，但小矩阵不一定更快。结合 Nsight Compute 结果看，当前实现仍受 shared memory bank conflict、global store access pattern 和 register pressure 影响。

## Notes

- 这些数据用于学习和趋势观察，不代表最终优化结果
- 当前项目不是工业级 GEMM 库，也不声明达到 cuBLAS 水平
- 非整除尺寸可以帮助观察 shape 行为，但 correctness 仍应单独开启 CPU check 验证

## Source READMEs

- [`naive_kernel/README.md`](./naive_kernel/README.md)
- [`Global_Memory_Coalescing_kernel/README.md`](./Global_Memory_Coalescing_kernel/README.md)
- [`SMEM_kernel/README.md`](./SMEM_kernel/README.md)
- [`1D_Blocktiling_kernel/README.md`](./1D_Blocktiling_kernel/README.md)
- [`2D_Blocktiling_kernel/README.md`](./2D_Blocktiling_kernel/README.md)
