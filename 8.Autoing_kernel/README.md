# Stage 8: Compile-time GEMM Autotuning

本阶段在 Stage 7 的 shared-memory padding kernel 上做编译期参数搜索。计算、
`float4` 访存和 As/Bs padding 布局保持不变，只改变 `BM`、`BN`、`BK`、`TM`
和 `TN`，比较不同 tile 形状在多种矩阵上的实际吞吐。

## Kernel 结构

`Shared_Memory_Layout_Padding_kernel.cu` 是 Stage 7 kernel 的模板化版本：

- A、B 使用 `float4` 从 global memory 加载，C 使用 `float4` 写回
- A 在写入 shared memory 时转置为 `As[BK][BM + 4]`
- B 使用中间插入 4 个 float 的 padding，物理布局为 `Bs[BK][BN + 4]`
- 每个线程计算一个 `TM x TN` 的 register tile
- block 线程数为 `BM * BN / (TM * TN)`，所有配置均要求整 warp

`autotune_padding_benchmark.cu` 为每组模板参数实例化独立 kernel，先在
`256^3` 上做 CPU reference check，再使用 CUDA event 测量 latency，并输出
Avg/Best GFLOPS、寄存器数、shared memory 用量和几何平均加速比。

这不是运行时 autotuner：每次运行都会把候选配置全部编译进可执行文件，
搜索结果需要再手动选出配置并固定到后续 kernel。

## 文件

| File | Purpose |
| --- | --- |
| `Shared_Memory_Layout_Padding_kernel.cu` | Stage 7 padding kernel 的模板化实现 |
| `autotune_padding_benchmark.cu` | 正确性、批量 benchmark、CSV 和排名 |
| `run_autotune.sh` | 编译并运行 quick/full suite |
| `autotune_full.csv` | 7 个矩阵、14 个配置的完整原始结果 |
| `ncu_C00_4096.ncu-rep` / `ncu_C08_4096.ncu-rep` | 基准与候选配置的 Nsight Compute 报告 |
| [`profiling/`](./profiling/README.md) | 报告说明和分类截图 |

## 运行

```bash
cd /home/fish/GEMM_For_Myself/8.Autoing_kernel

# 快速筛选：1024^3、4096^3，warmup=5，iters=20
./run_autotune.sh quick

# 完整比较：5 个方阵 + 2 个长宽比矩阵，warmup=10，iters=50
./run_autotune.sh full
```

RTX 4060 Laptop 使用 `sm_89`。其他 GPU 可以覆盖架构参数：

```bash
CUDA_ARCH=sm_89 ./run_autotune.sh quick
```

程序会先验证每个配置的 `256^3` CPU reference，默认容差为 `1e-2`。只有验证
通过且矩阵尺寸满足 tile 整除条件的配置才会进入测速和排名。

## 搜索空间

`C00` 是 Stage 7 基准配置；其余配置只改变编译期 tile 参数。

| Group | Values explored |
| --- | --- |
| `BM/BN` | `32/64/128` 的若干组合 |
| `BK` | `8`, `16` |
| `TM/TN` | `8x8`, `4x8`, `8x4` |

当前实现没有边界分支，输入必须满足所选配置的 `M % BM == 0`、`N % BN == 0`、
`K % BK == 0`，并满足 `float4` 加载/存储所需的静态约束。搜索器通过
`static_assert` 拒绝当前 cooperative-load 映射无法安全表示的组合。

## 完整 suite 结果

测试环境为 NVIDIA GeForce RTX 4060 Laptop GPU，warmup 10 次，正式迭代 50 次。
下表使用同一轮 `autotune_full.csv` 的 Avg GFLOPS；百分比相对同一轮的 C00，
不是与 Stage 7 历史测试轮次直接混合。

| Case | C00 `64x64x8, 8x8` | C08 `128x64x16, 8x8` | Change |
| --- | ---: | ---: | ---: |
| `256^3` | 1161.39 | 1323.16 | +13.9% |
| `512^3` | 4793.33 | 4403.70 | -8.1% |
| `1024^3` | 7194.59 | 8764.70 | +21.8% |
| `2048^3` | 9776.41 | 9614.12 | -1.7% |
| `4096^3` | 9103.69 | 9664.98 | +6.2% |
| `4096x256x4096` | 7308.30 | 8865.61 | +21.3% |
| `256x4096x4096` | 8396.84 | 8942.19 | +6.5% |

完整 suite 的综合排名：

1. `C08_128x64x16_8x8`: `1.0805x`
2. `C13_64x64x16_8x4`: `1.0631x`
3. `C11_64x64x8_8x4`: `1.0312x`

C08 是本轮 7 个 case 的综合候选，在 `4096^3` 达到 `9664.98 GFLOPS`
(`9.665 TFLOPS`)；它不是所有矩阵尺寸上的单 case 最优配置。Laptop GPU
会受温度、功耗和频率影响，候选配置在正式采用前应重复运行。

## 相关记录

- [完整原始 CSV](./autotune_full.csv)
- [Nsight Compute 报告与截图](./profiling/README.md)
- [根目录性能汇总](../PERFORMANCE_SUMMARY.md)
