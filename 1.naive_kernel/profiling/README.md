# Naive Kernel Profiling

这一组图对应的是整个项目里最朴素、也最重要的一版 baseline。

我很想把这个目录保留下来，不是因为 naive kernel 本身有多强，而是因为后面所有优化版本，其实都要回到这里做对照。如果没有这套 profiling，后面看到 coalesced 或 shared-memory 版变快时，我们很容易只记住“结果变好了”，但说不清到底是哪一类问题被解决了。

## 这一版想回答什么

这一版 profiling 主要想回答三个问题：

1. 这个 naive kernel 到底是不是 memory-bound
2. 它慢，慢在 global memory 访问模式，还是慢在算术本身
3. 这些问题能不能在 Nsight Compute 里被直接看见

答案其实很统一：可以，而且证据很集中。

## 这一组图怎么读

建议先按下面这个顺序看：

1. [full report summary](./full_report/figures/01_summary.png)
2. [speed of light](./full_report/figures/02_speed_of_light.png)
3. [memory workload](./full_report/figures/03_memory_workload.png)
4. [scheduler statistics](./full_report/figures/04_scheduler_statistics.png)
5. [warp state](./full_report/figures/05_warp_state.png)
6. [instruction statistics and mix](./instr_report/figures/01_instruction_statistics_and_mix.png)

如果想先抓主线，可以只看前 3 张。  
如果想把“问题是怎么一层层传导到调度和 stall 上”的链条看完整，再继续看后 3 张。

## 从这些图里看到的主线

这版最重要的结论可以直接写成一句话：

- 这不是一个“算力不够”的 naive kernel，而是一个被低效 global-memory 访问模式拖住的 baseline

之所以敢这么写，是因为几张图给出的信号非常一致：

- `Summary` 里 compute throughput 明显低，memory throughput 却几乎拉满
- `Speed Of Light` 里 memory 和 compute 的差距非常夸张
- `Memory Workload Analysis` 直接指出了 uncoalesced global load/store
- `Scheduler Statistics` 说明 eligible warps 很少
- `Warp State Statistics` 里 `LG Throttle` 几乎是一眼就能看出来的主导 stall
- `Instruction Statistics` 里 `LDG` 占比又非常重

换句话说，这些图不是彼此独立的，它们讲的是同一个故事，只是站在不同层面重复确认：

- 代码层面看是 non-coalesced 访问
- 内存层面看是 sector 利用率差
- 调度层面看是 warp 发不出来
- stall 层面看是 `LG Throttle`
- 指令层面看是 `LDG` 太重

## 为什么这一版值得保留

我现在反而觉得，学习型项目里最不能删掉的就是这种“很基础、但问题很清楚”的版本。

因为后面无论是：

- `Global_Memory_Coalescing_kernel`
- `SMEM_kernel`
- 还是以后做 `block tiling`、`register blocking`

都需要有一个清楚的起点，才能回答“优化到底解决了什么”。

对这个项目来说，naive 版 profiling 的价值不在于它跑得慢，而在于它把“为什么慢”暴露得足够清楚。

## 这个目录里放了什么

- [capture_commands.md](./capture_commands.md)
  记录当前采集命令
- [visual_checklist.md](./visual_checklist.md)
  记录这一版最值得保留的图
- [instr_report/README.md](./instr_report/README.md)
  指令报告逐图解读
- [full_report/README.md](./full_report/README.md)
  完整报告逐图解读

`Source` 截图这次我没有强行保留，因为这套材料已经足够把主线讲清楚了。后面如果我想专门讲 kernel 代码热点，再单独补也完全来得及。
