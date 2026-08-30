# Stage 10 — Warp Tiling Experiment

## Goal

本阶段在 vectorized/register-tiled SGEMM 上显式加入 warp tile，让一个 block 内的
warp 分别负责固定的 `WM × WN` 区域。它是 Stage 8 之后的高级实验分支，不是
Stage 11 的直接父实现。

## Kernel Organization

| Parameter | Value |
| --- | ---: |
| `BM/BN/BK` | `128/128/16` |
| `WM/WN` | `64/64` |
| `TM/TN` | `8/4` |
| `WMITER/WNITER` | `2/2` |
| Block | 128 threads / 4 warps |
| Static shared memory | 16384 bytes |

每个 warp 负责一个 `64×64` tile，并通过两个 M/N subtile 迭代更新寄存器结果。
A 使用转置后的 `As[BK][BM]`，B 使用 `Bs[BK][BN]`，A/B/C 均采用 `float4`
global-memory transaction。

## Files and Authority

- `main_Wraptiling_kernel.cu`：权威 kernel。
- `Wraptiling_kernel_benchmark.cu`：直接 include 权威 kernel，负责 correctness、
  CUDA Event benchmark 和 CSV 输出。
- `profiling/raw/`：Nsight Compute 原始报告。

历史目录名 `Wraptiling` 保留兼容性，正文统一使用正确名称 Warp Tiling。

## Build

```bash
cd /home/fish/GEMM_For_Myself
make warp
```

## Correctness

```bash
./build/warp_tiling_bench 256 256 256 \
  --warmup 2 --iters 5 --max-check-dim 256
```

该配置已单独运行 CPU reference 并得到 `PASS`。正式性能运行使用
`--no-check`，其中的 `SKIP` 仅表示没有在该次运行执行 CPU reference。

## Benchmark

2026-08-30 的同轮 4096³ FP32 comparison 得到 Avg `6657.02`、Best `6931.66`
GFLOPS，与下面的历史完整 suite 记录接近。统一结果见
[`comparison_4096.csv`](../results/rtx4060_laptop/comparison_4096.csv)。

历史完整 suite 使用 warmup 10、iterations 50：

```bash
./build/warp_tiling_bench --warmup 10 --iters 50 --no-check
```

| Case | Avg GFLOPS | Best GFLOPS |
| --- | ---: | ---: |
| `1024³` | 6313.0621 | 6553.6002 |
| `2048³` | 6578.1271 | 7760.0446 |
| `4096³` | 6670.5097 | 6873.8835 |

另一次独立 `4096³` 得到 Avg `6748.1435`、Best `6990.1429` GFLOPS；根汇总
保留统一 suite 的 `6.671 TFLOPS` 作为主记录。

## Nsight Compute Findings

`1024³` Full report 显示：

- 168 registers/thread，无 spill；static shared memory 16.38 KiB。
- 理论 occupancy 25%，register 和 shared memory 都限制 block residency。
- Compute throughput 52.46%，memory throughput 33.03%。
- Active/eligible/issued warps per scheduler 为 `2.68/1.00/0.54`。
- No Eligible cycles 45.84%，warp cycles per issued instruction 4.94。
- shared store 平均约 4-way bank conflict，786432 conflicts，占 shared-store
  wavefront 的约 60%。

详细数据与复现命令见 [profiling/README.md](./profiling/README.md)。profiler
duration 是 replay 结果，不属于正式性能表。

## Why It Did Not Help

Warp Tiling 本身不是必然负优化；本实验的当前线程映射和 tile 参数同时带来了较高
register/shared-memory 资源占用、较低 occupancy、较少 eligible warps 和 shared
store conflict。结果是 `4096³` 从成熟的 8–9+ TFLOPS 路线回退到约 6.66
TFLOPS。这个阶段作为“复杂优化不保证单调变快”的负优化证据被完整保留。

## Limitations

- 只支持 FP32。
- 要求 `M%128==0`、`N%128==0`、`K%16==0`。
- 没有 arbitrary-shape tail path。
- 参数针对 RTX 4060 Laptop，结论不泛化到其他 GPU。

## What I Learned

warp-level 分工只有在资源占用、shared-memory 布局和调度效率共同合理时才可能带来
收益。更复杂的层次结构不会自动转化成更高吞吐。
