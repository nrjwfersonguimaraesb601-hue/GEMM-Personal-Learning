# Global Memory Coalescing Kernel Profiling Assets

这个目录用来收集 `Global_Memory_Coalescing_kernel` 的 profiling 资料。

这一版的关键不是“绝对值多高”，而是：

- 和 naive 版相比，global memory access 更连续
- 应该能从 memory 相关图表里看到更干净的访问模式
- 很适合放“和 naive 对照”的截图

建议先看：

- [capture_commands.md](/home/fish/GEMM_For_Myself/Global_Memory_Coalescing_kernel/profiling/capture_commands.md)
- [visual_checklist.md](/home/fish/GEMM_For_Myself/Global_Memory_Coalescing_kernel/profiling/visual_checklist.md)

建议和 naive 保持同一套文件命名：

- `instr_report/figures/01_instruction_statistics_and_mix.png`
- `full_report/figures/01_summary.png`
- `full_report/figures/02_speed_of_light.png`
- `full_report/figures/03_memory_workload.png`
- `full_report/figures/04_scheduler_statistics.png`
- `full_report/figures/05_warp_state.png`

如果这一版额外想强调 memory transaction 改善，可以再补：

- `full_report/figures/06_memory_chart.png`
