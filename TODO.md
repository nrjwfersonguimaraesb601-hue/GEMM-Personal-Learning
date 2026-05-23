# TODO

这个文件按 Si Boehm 的 CUDA GEMM worklog 路线来维护当前项目的推进顺序：

参考：
- Si Boehm, "How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog"
- https://siboehm.com/articles/22/CUDA-MMM

## 当前状态

- [x] `naive_kernel` 基线实现
- [x] `naive_kernel` benchmark 固化
- [x] `naive_kernel` README 与实测数据整理
- [x] `naive_kernel` Nsight Compute 截图归档与解读
- [x] `Global_Memory_Coalescing_kernel` 实现
- [x] `Global_Memory_Coalescing_kernel` benchmark 与 README 整理
- [x] `Global_Memory_Coalescing_kernel` Nsight Compute 截图归档与解读
- [x] `SMEM_kernel` shared-memory caching / tiling 实现
- [x] `SMEM_kernel` benchmark 与 README 整理
- [x] `SMEM_kernel` teacher snippet 逐行讲解整理
- [x] `SMEM_kernel` Nsight Compute 截图归档与解读
- [x] 三个 kernel 的 profiling 目录规范统一

## 下一步

- [ ] 系统比较 block size
  - [ ] 跑 `16x16`、`32x8`、`32x16`、`8x32`
  - [ ] 把 `32x32` 也纳入 shared-memory 版本的对比
  - [ ] 记录 square case 和 rectangular case 的差异
  - [ ] 固化一版默认配置

- [ ] `1D_BlockTiling_kernel`
  - [ ] 让一个 thread 计算多个输出值
  - [ ] 在 shared-memory 版本基础上继续放大数据复用
  - [ ] 观察寄存器使用和吞吐变化
  - [ ] 和 shared memory 版本对比

- [ ] `2D_BlockTiling_kernel`
  - [ ] 扩展到二维 block tiling
  - [ ] 比较和 1D block tiling 的收益差异

## 后续优化

- [ ] `Vectorized_Memory_Access_kernel`
  - [ ] 尝试 `float4` / vectorized load-store
  - [ ] 检查对齐要求

- [ ] `WarpTiling_kernel`
  - [ ] 引入 warp-level 分工
  - [ ] 观察是否进一步接近文章里的高性能阶段

- [ ] `Autotuning`
  - [ ] 自动搜索 block/tile 配置
  - [ ] 固化一份适合 RTX 4060 Laptop GPU 的参数组合

- [ ] `cuBLAS` 对比
  - [ ] 增加 cuBLAS baseline
  - [ ] 统一输出 speedup 和 relative performance

## 每一阶段的完成标准

每做完一个 kernel 版本，至少补齐下面这些内容：

- [x] correctness check 通过
- [x] benchmark 可复现
- [x] README 记录核心改动
- [x] README 记录相对上一版的性能提升
- [x] 至少保留一组 RTX 4060 Laptop GPU 的实测数据

下一版开始时，重新按该 checklist 检查新的 kernel 目录。

## 可选补充

- [ ] 加 occupancy 分析
- [ ] 加 memory bandwidth 分析
- [ ] 给 `SMEM_kernel` 补 `occupancy` 和 `shared_memory` 专门截图
- [ ] 做一份三版 kernel 的 profiling 对照总表
- [ ] 画 `GFLOPS vs size` 曲线
- [ ] 画 `runtime vs size` 曲线
- [ ] 补更多非方阵 case
