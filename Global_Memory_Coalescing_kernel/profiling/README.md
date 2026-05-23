# Global Memory Coalescing Kernel Profiling

如果说 naive 版是在回答“为什么它会慢”，那这一版 profiling 更像是在回答：

- 只改线程映射和访问方向，不改算法结构，到底能不能真的变快

从结果看，答案是肯定的，而且比我一开始预期得还要直接。

## 这一版想回答什么

这一版最重要的问题不是绝对性能有多高，而是：

1. 和 naive 相比，访存模式是不是真的变干净了
2. 这种改善能不能在 Nsight Compute 的图里被看见
3. 提升来自“算法变了”，还是主要来自“global-memory 访问终于更像样了”

这也是我为什么特别想保留这一版 profiling。  
因为它很像一个标准的中间 checkpoint：代码改动不算大，但结果和原因之间的因果关系非常清楚。

## 建议怎么看这组图

建议先和 naive 对照着看，而不是单看这一版：

1. [full_report/figures/01_summary.png](/home/fish/GEMM_For_Myself/Global_Memory_Coalescing_kernel/profiling/full_report/figures/01_summary.png)
2. [full_report/figures/02_speed_of_light.png](/home/fish/GEMM_For_Myself/Global_Memory_Coalescing_kernel/profiling/full_report/figures/02_speed_of_light.png)
3. [full_report/figures/03_memory_workload.png](/home/fish/GEMM_For_Myself/Global_Memory_Coalescing_kernel/profiling/full_report/figures/03_memory_workload.png)
4. [full_report/figures/04_scheduler_statistics.png](/home/fish/GEMM_For_Myself/Global_Memory_Coalescing_kernel/profiling/full_report/figures/04_scheduler_statistics.png)
5. [full_report/figures/05_warp_state.png](/home/fish/GEMM_For_Myself/Global_Memory_Coalescing_kernel/profiling/full_report/figures/05_warp_state.png)
6. [instr_report/figures/01_instruction_statistics_and_mix.png](/home/fish/GEMM_For_Myself/Global_Memory_Coalescing_kernel/profiling/instr_report/figures/01_instruction_statistics_and_mix.png)

只要把 `naive` 和 `gmemc` 的 `summary / speed_of_light / memory_workload` 三张图并排看，很多结论其实就已经出来了。

## 这一版最重要的观察

这一版最有意思的地方在于：

- 指令类型没有发生“天翻地覆”的变化
- 但性能状态和硬件利用率却已经明显不一样了

这说明一个非常重要的学习点：

- 有时候 kernel 变快，不是因为指令种类完全换了
- 而是因为同样在做 load/store，这些访问终于更符合 GPU 喜欢的方式了

也正因为如此，这一版最值得看的不是单独的 opcode 表，而是 memory 相关页面：

- `Memory Workload Analysis`
- `Speed Of Light`
- `Scheduler Statistics`
- `Warp State`

它们合起来在说明一件事：

- warp 内线程访问更连续以后，global memory transaction 的浪费下降了
- SM 不再像 naive 那样明显被饿住
- 调度层面和 stall 层面的压力也跟着缓和了

## 这一版在整个项目里的位置

我把这版看成“第一个真正把原因和收益连起来的优化版本”。

原因是：

- 它还没有引入 shared memory
- 也没有引入更深层的 tiling
- 但已经能靠更合理的 global-memory access pattern 把 kernel 从 naive baseline 拉到完全不同的档位

这一步非常适合作为后面 `SMEM_kernel` 的前置对照，因为 shared memory 再往前走时，我们就能问：

- 如果 coalescing 已经把事情做到了这里，那 shared memory 到底还能额外带来什么

## 这个目录里放了什么

- [capture_commands.md](/home/fish/GEMM_For_Myself/Global_Memory_Coalescing_kernel/profiling/capture_commands.md)
- [visual_checklist.md](/home/fish/GEMM_For_Myself/Global_Memory_Coalescing_kernel/profiling/visual_checklist.md)
- [instr_report/README.md](/home/fish/GEMM_For_Myself/Global_Memory_Coalescing_kernel/profiling/instr_report/README.md)
- [full_report/README.md](/home/fish/GEMM_For_Myself/Global_Memory_Coalescing_kernel/profiling/full_report/README.md)

如果后面我还想让这版更完整，最值得额外补的一张还是 `Memory Chart`，因为它最适合拿来和 naive 做“transaction 利用率改善”的直观对照。
