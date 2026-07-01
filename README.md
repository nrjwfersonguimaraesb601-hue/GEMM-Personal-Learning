# GEMM_For_Myself

这是一个我自己用来学习 CUDA GEMM 的仓库。

这个项目的目标不是一下子写出一个最终高性能库，而是把优化路径一版一版走清楚：

- 先有 baseline
- 再做 coalescing
- 再做 shared memory
- 再做 block tiling / register blocking

我更在意的是每一步都能解释清楚、验证清楚、测量清楚。

## 当前进度

现在仓库里已经有四个主要阶段：

- [naive_kernel](./naive_kernel/README.md): 最基础的 non-coalesced baseline
- [Global_Memory_Coalescing_kernel](./Global_Memory_Coalescing_kernel/README.md): 第一版 global memory 合并访问优化
- [SMEM_kernel](./SMEM_kernel/README.md): 第一版 shared-memory tiling
- [1D_Blocktiling_kernel](./1D_Blocktiling_kernel/README.md): 第一版 1D block tiling / register blocking

这四版现在都已经开始有各自的 profiling 资料：

- `naive / coalesced / SMEM / 1D blocktiling` 都保留了 Nsight Compute 的截图和说明
- 可以直接沿着代码、benchmark、profiling 三条线一起看每一步优化到底改变了什么

## 这次统一 benchmark 口径

下面这些 README 和总结文件，当前都统一按这套测速口径记录：

- GPU: `NVIDIA GeForce RTX 4060 Laptop GPU`
- Warmup: `10`
- Iterations: `50`
- `CPU check: disabled`
- 表格中的结果是 pure kernel benchmark，不含 H2D / D2H 拷贝

correctness 仍然由各目录里的 correctness-first 可执行版本单独确认。

## 现在可以怎么理解这条优化路线

从这次最新结果看，路线已经很清楚了：

1. naive baseline 先把起点固定在 `~123 GFLOPS`
2. coalescing 直接把 square case 推到 `~616-1026 GFLOPS`
3. shared memory 再把这条线往上推到 `~641-1039 GFLOPS`
4. 1D block tiling / register blocking 则第一次把主力区间推进到 `~1.26-3.70 TFLOPS`

换句话说，这个项目现在已经不只是“有几个学习版 kernel”，而是已经形成了一条比较完整、数据也比较连贯的优化路径。

## 仓库结构

- `naive_kernel/`: naive baseline
- `Global_Memory_Coalescing_kernel/`: coalesced global-memory 版本
- `SMEM_kernel/`: shared-memory 版本
- `1D_Blocktiling_kernel/`: 1D block tiling 版本
- `*/profiling/`: 对应版本的 Nsight Compute 截图、命令模板和解读
- `PERFORMANCE_SUMMARY.md`: 各版本横向对比
- `TODO.md`: 后续推进路线
- `NSIGHT_COMPUTE_PROFILING_GUIDE.md`: Nsight Compute 使用记录

## 当前结论

按这次最新数据，我会把这四版的定位记成这样：

- `naive`: 稳定 baseline
- `coalesced`: 第一波大提升
- `SMEM`: 在 coalesced 基础上继续抬高
- `1D blocktiling`: 当前项目里最强的一版，第一次明显进入 `TFLOPS` 级别并站稳 `3 TFLOPS+`

这也说明后面的路线已经比较明确了：

- 继续做更深的 tiling
- 继续提高寄存器和 shared memory 的利用
- 最后再和 cuBLAS 做系统对比
