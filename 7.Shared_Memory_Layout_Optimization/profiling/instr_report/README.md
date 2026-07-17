# Shared-memory Padding GEMM Instruction Report

## Figure

- [01_instruction_statistics_and_mix.png](./figures/01_instruction_statistics_and_mix.png)：
  SASS 指令统计与 executed instruction mix

## Interpretation

报告记录：

- Executed instructions：`39,711,744`
- Issued instructions：`39,716,440`
- Avg. executed instructions/scheduler：`413,664`
- Avg. issued instructions/scheduler：`413,712.92`

Executed Instruction Mix 中 `FFMA` 仍然绝对主导，符合每个线程在寄存器中计算
`8 x 8` micro tile 的结构。随后依次是 `LDS`、`IADD3` 和 `STS`：

- `LDS`/`STS` 对应每个 K tile 的 shared-memory 读取与写入
- `IADD3` 排名上升与 padding 后的物理 stride 和地址计算相符
- `LDG`/global store 指令数较少，说明 `float4` 向量化仍然生效

Stage 6 约执行 `38.14M` 条指令，本阶段为 `39.71M`。padding 增加了一部分地址
计算，但它消除了 shared-memory bank conflict；最终 full report 的 duration 从
`319.46 us` 降到 `282.98 us`，说明减少 replay/wavefront 的收益超过了额外整数
地址指令的成本。
