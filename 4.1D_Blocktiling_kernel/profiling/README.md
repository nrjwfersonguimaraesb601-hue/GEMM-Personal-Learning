# 1D Blocktiling Profiling

这一组图对应的是当前项目里性能最强的一版 kernel。

和前面的 `naive / coalesced / SMEM` 相比，这一版 profiling 最值得看的地方不是“又快了一点”，而是瓶颈形态已经明显变了：

- `FFMA` 已经成为主导指令
- `LDS` 很重，说明 shared-memory 复用在真正工作
- `MIO Throttle` 和 `Long Scoreboard` 成了更核心的 stall
- theoretical occupancy 也开始被寄存器数量限制

也就是说，这一版已经不再主要是“global memory 访问方向不对”的问题，而是进入了更典型的 tiling / register blocking 阶段。

## 建议怎么看

建议按这个顺序看：

1. [full_report/figures/01_summary.png](/home/fish/GEMM_For_Myself/1D_Blocktiling_kernel/profiling/full_report/figures/01_summary.png)
2. [full_report/figures/02_speed_of_light.png](/home/fish/GEMM_For_Myself/1D_Blocktiling_kernel/profiling/full_report/figures/02_speed_of_light.png)
3. [instr_report/figures/01_instruction_statistics_and_mix.png](/home/fish/GEMM_For_Myself/1D_Blocktiling_kernel/profiling/instr_report/figures/01_instruction_statistics_and_mix.png)
4. [full_report/figures/03_memory_chart.png](/home/fish/GEMM_For_Myself/1D_Blocktiling_kernel/profiling/full_report/figures/03_memory_chart.png)
5. [full_report/figures/04_scheduler_statistics.png](/home/fish/GEMM_For_Myself/1D_Blocktiling_kernel/profiling/full_report/figures/04_scheduler_statistics.png)
6. [full_report/figures/05_warp_state.png](/home/fish/GEMM_For_Myself/1D_Blocktiling_kernel/profiling/full_report/figures/05_warp_state.png)

## 这一版最值得记住的点

- `Summary` 里 compute / memory throughput 都在 `~78.74%`
- `Instruction Statistics` 里 `FFMA` 第一，`LDS` 第二
- `Summary` 里已经直接提示 `Mio Throttle Stalls` 和 `Theoretical Occupancy`
- `Memory Chart` 里 shared-memory traffic 很重，`L1/TEX hit rate` 很低，但 `L2 hit rate` 很高
- `Warp State` 里 `MIO Throttle` 是第一大 stall，`Long Scoreboard` 紧随其后

对这个目录来说，这组图最适合支持这样的结论：

- 1D block tiling 确实把计算强度和数据复用往上推了一大截
- 但现在新的压力主要落在 shared-memory / MIO pipeline / registers 这一侧
- 这正是后面继续做 `2D block tiling`、`vectorized load`、`warp tiling` 的出发点

## 目录内容

- [capture_commands.md](/home/fish/GEMM_For_Myself/1D_Blocktiling_kernel/profiling/capture_commands.md)
- [visual_checklist.md](/home/fish/GEMM_For_Myself/1D_Blocktiling_kernel/profiling/visual_checklist.md)
- [full_report/README.md](/home/fish/GEMM_For_Myself/1D_Blocktiling_kernel/profiling/full_report/README.md)
- [instr_report/README.md](/home/fish/GEMM_For_Myself/1D_Blocktiling_kernel/profiling/instr_report/README.md)
