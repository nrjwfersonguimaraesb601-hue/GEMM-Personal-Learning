# SMEM Kernel Profiling Assets

这个目录用来收集 `SMEM_kernel` 的 profiling 资料。

这一版最值得展示的，不只是吞吐变化，而是：

- shared memory 把 block 内数据复用这件事做起来了
- 但同步、shared load/store、occupancy 也开始带来代价
- 它很适合展示“shared memory 不是免费午餐”

建议先看：

- [capture_commands.md](/home/fish/GEMM_For_Myself/SMEM_kernel/profiling/capture_commands.md)
- [visual_checklist.md](/home/fish/GEMM_For_Myself/SMEM_kernel/profiling/visual_checklist.md)

建议和前两版尽量统一命名：

- `instr_report/figures/01_instruction_statistics_and_mix.png`
- `full_report/figures/01_summary.png`
- `full_report/figures/02_speed_of_light.png`
- `full_report/figures/03_memory_workload.png`
- `full_report/figures/04_scheduler_statistics.png`
- `full_report/figures/05_warp_state.png`

如果后面你补更多图，这一版再额外建议：

- `full_report/figures/06_occupancy.png`
- `full_report/figures/07_shared_memory.png`
