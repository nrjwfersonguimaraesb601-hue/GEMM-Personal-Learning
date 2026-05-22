# GEMM_For_Myself

这是一个我自己用来学习 CUDA GEMM 的小项目仓库。

我现在还在学习中，这个项目不是一个已经完成的高性能库，也不是一个成熟的工程产品。它更像是我的一份学习记录、实验记录和优化过程记录。我希望通过自己手写、自己测试、自己分析的方式，逐步理解一个 GEMM kernel 是怎么从最朴素的实现，一步一步走向更高性能的。

## 项目目标

这个项目主要有几个目标：

- 从零开始实现一个最基础的 CUDA GEMM
- 先完成 correctness，再逐步关注 performance
- 给每个阶段都保留 benchmark 结果和分析
- 理解 CUDA GEMM 中常见的优化手段
- 建立一条属于我自己的学习路线，而不是只停留在“看懂代码”

对我来说，这个项目最重要的不是一上来就追求极致性能，而是把每一步为什么这么写、为什么会更快、为什么会变慢都尽量搞清楚。

## 参考资料

这个项目目前主要参考了 Si Boehm 的文章：

- [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM)

这篇文章对我的帮助很大，因为它不是直接丢出一个最终版本，而是按照 worklog 的方式，从 naive kernel 开始，逐步引入优化，包括：

- global memory coalescing
- shared memory caching
- block tiling
- vectorized memory access
- warptiling
- autotuning

我很喜欢这种“从基础实现一路推到优化实现”的思路，所以我也想按类似的方式，把自己的 GEMM 学习过程整理下来。

## 当前进度

目前这个仓库还在 very early stage，已经完成和正在做的内容包括：

- [naive_kernel](./naive_kernel/README.md)：实现并验证了最基础的 naive CUDA GEMM kernel
- 为 naive kernel 写了一个 benchmark 版本
- 在 RTX 4060 Laptop GPU 上跑出了一组刻意保留 non-coalesced 访问方式的 baseline 数据
- [Global_Memory_Coalescing_kernel](./Global_Memory_Coalescing_kernel/README.md)：完成了第一步 global memory coalescing 优化，并记录了相对 naive 的提升
- [SMEM_kernel](./SMEM_kernel/README.md)：完成了第一版 shared-memory caching / tiling，实现了 block 内 tile 复用，并整理了相对前两版的 benchmark 结果
- 用 [TODO.md](./TODO.md) 维护后续优化路线

换句话说，我现在还处在“先把 baseline 做扎实”的阶段。

## 计划路线

接下来我希望这个项目大致按下面的路线推进：

1. 完成 naive kernel 的整理和 benchmark 固化
2. 分析 block size 对性能的影响
3. 引入 global memory coalescing 优化
4. 引入 shared memory tiling
5. 在 shared memory 基础上尝试 register blocking / block tiling
6. 补充 vectorized load/store
7. 加入 cuBLAS baseline 做对比
8. 对不同版本做系统化性能分析

如果后面我学到更多内容，也可能继续往这些方向扩展：

- warp-level optimization
- occupancy 分析
- memory bandwidth 分析
- autotuning
- 更大矩阵和更多 shape 的测试

## 仓库结构

当前仓库结构还比较简单：

- `naive_kernel/`: 最基础的一版 CUDA GEMM 实现，以及对应的 benchmark 和性能分析文档
- `Global_Memory_Coalescing_kernel/`: 第一版 coalesced global-memory 访问优化，以及对应 benchmark
- `SMEM_kernel/`: 第一版 shared-memory caching / tiling 实现，以及对应 benchmark、teacher notes 和性能分析文档
- `TODO.md`: 按 worklog 路线维护后续优化任务

后面如果我继续推进优化版本，预计会逐渐补充更多子目录，比如：

- `coalesced_kernel/`
- `shared_memory_kernel/`
- `tiled_kernel/`
- `cublas_compare/`

当然，这部分还在学习和规划中，最终结构可能会边做边调整。

## 我希望这个项目呈现什么样子

我希望这个仓库最后不是一个“只放代码、不解释过程”的仓库，而是一个能看出学习路径的仓库。理想状态下，它应该能回答这些问题：

- 最基础的 CUDA GEMM 是怎么写的
- naive kernel 为什么慢
- 每一种优化到底解决了什么问题
- 优化后性能提升了多少
- 和 cuBLAS 还有多少差距

如果以后我回头看这个仓库，我希望我能看到的不只是代码本身，而是自己当时是怎么理解、怎么试、怎么改、怎么验证的。

## 当前状态说明

这个项目目前仍然处在学习阶段，所以这里的代码和文档都可能继续变化。你如果看到了比较基础、比较朴素、甚至有些“笨”的实现，那大概率不是因为我要故意写差，而是因为我正在按学习顺序一点点把这些东西吃透。

也正因为如此，这个仓库里会保留一些非常基础的版本。对我来说，这些版本很重要，因为它们记录了性能优化真正开始之前的起点。

## 致自己

这个项目是我给自己搭的一条 CUDA GEMM 学习路径。

我还在学习中，但我希望把这个过程认真留下来。哪怕一开始只是一个 naive kernel，只要每一步都能解释清楚、验证清楚、测量清楚，它就已经是一个很好的开始。
