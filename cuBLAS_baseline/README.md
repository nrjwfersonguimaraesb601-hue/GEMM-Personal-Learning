# cuBLAS Baseline

## Goal

使用 cuBLAS 为本项目提供外部性能参考。自定义 kernel 的公平主比较是 cuBLAS
FP32；TF32 Tensor Core 模式单独列出。

## Layout and API

benchmark 使用 row-major `A[M,K]`、`B[K,N]`、`C[M,N]`。传统 cuBLAS 接口按
column-major 解释数据，因此调用 `cublasGemmEx` 计算等价的：

```text
Cᵀ = Bᵀ × Aᵀ
```

无需真实转置矩阵，也没有把转置成本隐藏在计时外。

- FP32：`CUBLAS_COMPUTE_32F`
- TF32：`CUBLAS_COMPUTE_32F_FAST_TF32`

CUDA Event 只包围 GEMM，不包含 allocation、H2D/D2H、CPU reference 或 handle
初始化。

## Build and Run

```bash
cd /home/fish/GEMM_For_Myself
make cublas

./build/cublas_bench --math fp32 --warmup 10 --iters 50
./build/cublas_bench --math tf32 --warmup 10 --iters 50
```

单尺寸 correctness：

```bash
./build/cublas_bench 512 512 512 \
  --math fp32 --warmup 2 --iters 5 --max-check-dim 512
```

## Controlled 4096³ Result

2026-08-30 的统一 FP32 comparison（warmup 10、iterations 50）记录为：

| Avg (ms) | Min (ms) | Max (ms) | Avg GFLOPS | Best GFLOPS |
| ---: | ---: | ---: | ---: | ---: |
| 15.7804 | 14.8511 | 16.9605 | 8709.47 | 9254.48 |

同一脚本中的 Stage 8 C08 Avg 为 8914.28 GFLOPS，高约 2.35%；这是未锁定 laptop
GPU 状态的一次顺序运行，只支持“处于相近性能区间”，不证明稳定胜出。原始行见
[`comparison_4096.csv`](../results/rtx4060_laptop/comparison_4096.csv)。

## Historical 4096³ Reference

| Mode | Avg GFLOPS | Best GFLOPS | Meaning |
| --- | ---: | ---: | --- |
| cuBLAS FP32 | 9429.4001 | 9553.0118 | FP32 baseline |
| cuBLAS TF32 | 14844.7540 | 15925.2171 | separate Tensor Core reference |

独立 FP32 验证得到 Avg `9357.0652` GFLOPS；设置
`NVIDIA_TF32_OVERRIDE=0` 后得到 Avg `9308.8622` GFLOPS，处于同一波动区间。
因此 README 使用“约 9.3–9.4 TFLOPS”。

Stage 8 C08 的历史 9.665 TFLOPS 与这些 cuBLAS 数据来自不同测试轮次。合理结论是：
在本机 `4096³` workload 下，最佳自定义 FP32 kernel 达到与 cuBLAS FP32 相同的
性能区间；不能声称稳定超过 cuBLAS。

## Limitations

- TF32 与自定义 FP32 不属于同精度公平比较。
- Laptop GPU 的温度、TGP 和 boost clock 会造成波动。
- 本项目只记录单 GPU、单精度、单机 workload，不代表跨硬件结论。
