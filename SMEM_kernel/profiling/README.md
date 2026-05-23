# SMEM Kernel Profiling

这一组图对我来说特别有意思，因为它终于把 shared memory 引进来了，但结果却没有简单粗暴地变成“理所当然更快很多”。

这正是我很想保留它的原因。  
它不像 coalesced 版那样是一个非常干净的正收益故事，而更像是一个真实的学习节点：

- 共享内存复用已经做起来了
- 但是同步、bank conflict、occupancy、MIO pipeline 压力也跟着出现了

换句话说，这一版开始真正告诉我：

- shared memory 不是魔法，它是一种交换

## 这一版想回答什么

这一版 profiling 最想回答的是：

1. shared memory 到底有没有改变指令和数据流结构
2. 如果变了，为什么吞吐没有稳定碾压 coalesced 版
3. 现在新的瓶颈是不是已经不是 naive/coalesced 那种 global-memory 模式问题了

这些问题在这套图里其实都能找到答案。

## 建议怎么读这组图

这一版最好按“先看结构变化，再看代价”的顺序读：

1. [instr_report/figures/01_instruction_statistics_and_mix.png](/home/fish/GEMM_For_Myself/SMEM_kernel/profiling/instr_report/figures/01_instruction_statistics_and_mix.png)
2. [full_report/figures/01_summary.png](/home/fish/GEMM_For_Myself/SMEM_kernel/profiling/full_report/figures/01_summary.png)
3. [full_report/figures/02_speed_of_light.png](/home/fish/GEMM_For_Myself/SMEM_kernel/profiling/full_report/figures/02_speed_of_light.png)
4. [full_report/figures/03_memory_workload.png](/home/fish/GEMM_For_Myself/SMEM_kernel/profiling/full_report/figures/03_memory_workload.png)
5. [full_report/figures/04_scheduler_statistics.png](/home/fish/GEMM_For_Myself/SMEM_kernel/profiling/full_report/figures/04_scheduler_statistics.png)
6. [full_report/figures/05_warp_state.png](/home/fish/GEMM_For_Myself/SMEM_kernel/profiling/full_report/figures/05_warp_state.png)

如果只看一张最能体现“shared memory 真的进场了”的图，那就是第一张 instruction mix。
如果只看一张最能体现“它也带来了新代价”的图，那就是 `warp_state`。

## 这一版最重要的观察

这套图给我的最大感受不是“shared memory 很强”，而是：

- shared memory 让问题的形态变了

在前两版里，主线大致是：

- `LDG` 很重
- global access pattern 差
- `LG Throttle` 明显

但到了这一版，图里最显眼的变化已经变成：

- `LDS` 成了第一大项
- `STS`、`BAR` 出现
- `Shared Store Bank Conflicts` 被点出来
- `MIO Throttle` 开始主导 stall

这说明当前实现已经不再主要受 naïve 式 global-memory 低效访问支配，而是进入了 shared-memory kernel 常见的新阶段：

- 数据复用起来了
- 片上通信和同步也开始成为真实成本

## 为什么这版没有自动领先很多

这是我觉得最值得写进学习记录的一点。

很多时候我们在看 CUDA GEMM 教程时，会很容易形成一种错觉：

- 只要加了 shared memory，就应该自动更快

但这版 profiling 明确提醒我，事情没那么简单。

当前这版里，至少有几类代价已经能在图里直接看到：

- `MIO Throttle`
- `Shared Store Bank Conflicts`
- `Theoretical Occupancy`
- `Barrier` 相关 stall

所以这版的意义并不是“shared memory 一加就赢了”，而是：

- shared memory 这一步我已经亲手走通了
- 现在新的问题也被它一起暴露出来了
- 后续该做的事情变得更清楚了

比如：

- 调整 tile 和访问布局，减少 bank conflict
- 让一个 thread 计算多个输出值，提高复用摊销
- 控制 occupancy 和寄存器、shared memory 之间的平衡

## 这个目录里放了什么

- [capture_commands.md](/home/fish/GEMM_For_Myself/SMEM_kernel/profiling/capture_commands.md)
- [visual_checklist.md](/home/fish/GEMM_For_Myself/SMEM_kernel/profiling/visual_checklist.md)
- [instr_report/README.md](/home/fish/GEMM_For_Myself/SMEM_kernel/profiling/instr_report/README.md)
- [full_report/README.md](/home/fish/GEMM_For_Myself/SMEM_kernel/profiling/full_report/README.md)

如果后面我继续补图，这一版最值得再追加的是：

- `occupancy`
- 更专门的 `shared_memory` 页面

因为那会更直接地把“为什么这版还没完全放大收益”讲透。
