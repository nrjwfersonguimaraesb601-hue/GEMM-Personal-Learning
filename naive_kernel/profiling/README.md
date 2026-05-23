# Naive Kernel Profiling Assets

这个目录用来收集 `naive_kernel` 的 Nsight Compute 分析资料。

建议使用方式：

1. 先看 [capture_commands.md](/home/fish/GEMM_For_Myself/naive_kernel/profiling/capture_commands.md) 里的采集命令。
2. 按 [visual_checklist.md](/home/fish/GEMM_For_Myself/naive_kernel/profiling/visual_checklist.md) 的建议截图。
3. 把指令报告截图放进 `instr_report/figures/`。
4. 把完整报告截图放进 `full_report/figures/`。
5. 每张图截完后，在对应 README 里补一句你自己的观察。

推荐截图命名方式：

- `instr_report/figures/01_instruction_statistics_and_mix.png`
- `full_report/figures/01_summary.png`
- `full_report/figures/02_speed_of_light.png`
- `full_report/figures/03_memory_workload.png`
- `full_report/figures/04_scheduler_statistics.png`
- `full_report/figures/05_warp_state.png`

当前这版 naive 已经适配成“无 Source 截图也可以”的格式：

- `Instruction Statistics` 总览、上方统计表、`Executed Instruction Mix` 柱状图合并为 1 张
- `Source` 截图改为可选，不再强制要求
- `Summary` 图建议保留下面的 optimization opportunities

这一版最值得强调的是：

- non-coalesced global memory access
- 指令组成里 `LDG` 占比高
- 没有 shared memory 复用
- 适合作为后续两个版本的对照组
