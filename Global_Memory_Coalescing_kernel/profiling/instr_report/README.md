# Coalesced Instruction Report

把指令报告截图放到 `figures/`。

当前保留：

- `01_instruction_statistics_and_mix.png`

这张图建议和 naive 保持同样格式：

- `Instruction Statistics` 总览
- 统计表
- `Executed Instruction Mix`

`Source` 截图现在改为可选。

## 当前解读

### `01_instruction_statistics_and_mix.png`

这一版最有意思的地方在于，instruction mix 看起来和 naive 没有“换了一个世界”：

- `LDG` 仍然很高
- `FFMA` 仍然不是绝对主导
- `IMAD` 这类地址计算指令也还在

但性能却已经和 naive 拉开了明显差距。  
这恰好说明一个很重要的学习点：

- coalescing 的收益不一定会先体现在“指令种类完全改变”
- 更大的变化往往体现在“同样的 load 指令，访问效率更高了”

所以这张图最适合配的结论不是“opcode 变了很多”，而是：

- “这一步主要改的是 global-memory access pattern，而不是算法结构本身。”

也正因为如此，这张图更适合和 full report 里的 memory 页一起看，而不是单独下结论。
