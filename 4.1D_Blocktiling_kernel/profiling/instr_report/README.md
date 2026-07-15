# 1D Blocktiling Instruction Report

把 instruction-only 报告截图放到 `figures/` 目录。

当前保留：

- `01_instruction_statistics_and_mix.png`

## 当前解读

### `01_instruction_statistics_and_mix.png`

这一张图和前几版相比，最醒目的变化非常明确：

- `FFMA` 是第一大项
- `LDS` 排在第二
- `STS`、`BAR` 也已经进入主要指令成分

这组结构很能说明这版的性质：

- 算术本身已经很重
- shared-memory 复用已经真正参与主路径
- 同步与 shared-memory 开销也开始变成真实成本

如果只想用一句话概括这张图，我会写：

- “这版已经不是 `LDG` 主导的 baseline，而是 `FFMA + LDS` 主导的 block-tiling kernel。”

这也解释了为什么这版能明显快于前面的 SMEM 版本：

- 不只是搬进 shared memory 了
- 而是每个线程开始在寄存器里连续消费更多数据
