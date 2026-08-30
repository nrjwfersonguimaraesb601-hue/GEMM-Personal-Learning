# Stage 11 — Double Buffering Experiment

## Goal

本阶段尝试通过两个 shared-memory stage 隐藏 tile 加载延迟。它从
vectorized/padded register-tiled 路线发展而来，不包含 Stage 10 的 warp-tiling
模板参数，因此不是 Stage 10 的严格后继。

## Kernel Organization

| Parameter | Value |
| --- | ---: |
| `BM/BN/BK` | `128/128/8` |
| `TM/TN` | `8/8` |
| Block | 256 threads |
| Static shared memory | 16896 bytes |
| Registers/thread | 128 |

每个 thread 计算一个 `8×8` C micro tile。A 使用转置并 padding 的
`As[2][BK][BM+4]`，B 使用 `Bs[2][BK][BN+4]`。

## Data Movement and Pipeline

```text
current stage: SMEM[readStage] -> registers -> FFMA
next stage:    GMEM -> per-thread float4 registers -> SMEM[writeStage]
```

prologue 先填充 stage 0；主循环预取下一 tile、计算当前 tile，再把预取寄存器写入
另一个 SMEM stage 并交换 `readStage/writeStage`。这是 software-pipelined、
register-prefetched ping-pong double buffering，不是 `cp.async`。

## Files and Authority

- `DoubleBuffering_main_kernel.cu`：权威 double-buffering kernel。
- `Double_Buffering_benchmark.cu`：直接 include 权威 kernel。
- `results/`：正式 CUDA Event 记录。
- `profiling/`：原始 Nsight report、截图与分析。

## Build

```bash
cd /home/fish/GEMM_For_Myself
make double-buffering
```

## Correctness

```bash
./build/double_buffering_bench 256 256 256 \
  --warmup 2 --iters 5 --max-check-dim 256
```

该配置已单独执行 CPU reference 并得到 `PASS`。正式 `--no-check` 运行显示的
`SKIP` 不代表已在该次运行验证正确性。

## Performance

2026-08-30 的同轮 4096³ FP32 comparison 得到 Avg `8768.70`、Best `9224.07`
GFLOPS，处于同轮 C08 与 cuBLAS FP32 的相近区间。统一结果见
[`comparison_4096.csv`](../results/rtx4060_laptop/comparison_4096.csv)。

历史完整 suite 使用 warmup 10、iterations 50：

| Case | Avg GFLOPS | Best GFLOPS |
| --- | ---: | ---: |
| `1024³` | 7083.9711 | 7307.1498 |
| `2048³` | 7618.0570 | 8380.2273 |
| `4096³` | 8722.6411 | 9129.2158 |

另一次独立 `4096³` 得到 Avg `8854.6274`、Best `9012.7404` GFLOPS。根汇总
采用完整 suite 的 `8.723 TFLOPS`。它比 Stage 7 历史 full-suite 记录略高，但
没有超过 Stage 8 C08 的历史 `9.665 TFLOPS`，且不同轮次的小差距不构成严格排名。

## Nsight Compute Findings

`1024³` Full report 显示：

- Compute throughput 56.79%，Memory throughput 74.69%，L1/TEX 84.58%。
- 128 registers/thread，16.90 KiB static shared memory。
- 理论/实际 occupancy 为 33.33%/28.93%，register 是主要 occupancy 限制。
- Active/eligible/issued warps per scheduler 为 `3.47/1.63/0.64`，No Eligible
  为 35.51%。
- global store 每个 32-byte sector 平均只利用 16 bytes。
- shared load 平均约 5-way bank conflict，共 4194304 conflicts。
- Warp cycles per issued instruction 为 5.39。

double buffering 提高了调度和计算利用率，但没有消除 shared-load conflict、C
store sector 利用不足和 register-limited occupancy，因此收益有限。

## Limitations

- FP32 only；没有 Tensor Core、自定义 TF32 或 `cp.async`。
- 要求 `M%128==0`、`N%128==0`、`K%8==0`。
- 没有 arbitrary-shape tail path。
- 软件预取不保证编译器能完全重叠所有 GMEM latency。
- 参数和性能结论只对应当前 RTX 4060 Laptop 测试环境。

## What I Learned

double buffering 需要与加载映射、shared-memory layout、occupancy 和同步成本共同
评估。增加 pipeline stage 可以改善部分 latency hiding，但不会自动消除其他瓶颈。
