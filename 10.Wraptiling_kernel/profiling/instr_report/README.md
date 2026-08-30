# Warp Tiling Instruction Report

InstructionStats 原始报告记录：

- Executed instructions：40429568
- Issued instructions：40432677
- Avg. executed instructions/scheduler：421141.33
- Avg. issued instructions/scheduler：421173.72

源码中的外积结构意味着 FFMA 是核心计算指令，但 v1.0 没有导出 Instruction Mix
截图，因此不声明具体 `LDG/LDS/STG` 宽度。需要在 `ncu-ui` 或 SASS 中确认后再补充。
