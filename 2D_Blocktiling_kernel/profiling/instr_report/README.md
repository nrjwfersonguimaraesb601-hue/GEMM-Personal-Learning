# 2D Blocktiling Instruction Report

## Figure

- `01_instruction_statistics_and_mix.png`: SASS 指令统计与指令混合

## Interpretation

`FFMA` 是绝对主导指令，`LDS` 排名第二，符合 2D register tiling 使用寄存器外积并反复读取 shared memory 的结构。后续优化应在保留 FFMA 密度的同时，降低 LDS 冲突与 MIO pipeline 压力。
