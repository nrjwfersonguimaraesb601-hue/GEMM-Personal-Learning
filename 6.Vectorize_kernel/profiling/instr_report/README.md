# Vectorized GEMM Instruction Report

## Figure

- `01_instruction_statistics_and_mix.png`: SASS 指令统计与 executed instruction mix

## Interpretation

报告记录约 `38.14M` 条 executed instructions。`FFMA` 仍然绝对主导，`LDS`
排名第二，`STS` 排名第三，符合“寄存器外积 + shared-memory tile”的计算结构。

与源码和 cuobjdump 结果结合，可以确认编译器生成了 `LDG.E.128` 和
`STG.E.128`。后续分析不应只看 LDG/STG 的条数，还应同时检查每次 transaction
实际利用的 sector bytes；当前 full report 显示 global store 仍只有 `16/32 B`。
