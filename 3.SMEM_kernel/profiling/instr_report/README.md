# SMEM Instruction Report

把 instruction-only 报告截图放到 `figures/`。

当前保留：

- `01_instruction_statistics_and_mix.png`

同样建议把统计表和 `Executed Instruction Mix` 合并成 1 张主图。

`Source` 截图现在改为可选。

## 当前解读

### `01_instruction_statistics_and_mix.png`

这一张图和前两版最大的差别非常明显：

- 第一大项从 `LDG` 变成了 `LDS`
- `STS`、`BAR` 这些 shared-memory / 同步相关指令开始出现
- `FFMA` 仍然很高，但不再是唯一值得看的部分

这意味着 shared-memory 版本确实已经把数据流改写了：

- 更多读写发生在片上 shared memory
- global-memory 访问不再像 naive 那样占统治地位
- kernel 进入了“用 shared memory 换数据复用”的新阶段

这张图最适合配的一句话就是：

- “shared memory 让数据复用起来了，但代价也一起进来了。”

这里的“代价”在 instruction mix 里已经能看到一些苗头：

- `LDS` 很高，说明 shared-memory traffic 很重
- `BAR` 出现，说明同步也开始参与开销结构
