# Naive Instruction Report

把 instruction-only 报告的截图放到 `figures/` 目录。

当前保留 1 张主图：

- `01_instruction_statistics_and_mix.png`

这张图已经同时覆盖：

- `Instruction Statistics` 总览
- 上方的指令统计数值表
- `Executed Instruction Mix` 柱状图

`Source` 截图现在是可选项，不再要求必须上传。

## 当前解读

### `01_instruction_statistics_and_mix.png`

这张图最值得注意的不是“指令很多”，而是指令的组成非常偏向 global-memory baseline 的形态：

- `LDG` 是绝对主导项，明显高于 `FFMA`
- `IMAD` 这类地址计算相关整数指令也很多
- `FFMA` 虽然不少，但没有形成“计算主导”的结构

对这个 naive kernel 来说，这组现象非常合理：

- 每个 thread 只算一个输出元素
- `K` 维循环里不断从 global memory 反复取数
- 没有 shared memory 复用
- 线程映射又是刻意保留的 non-coalesced baseline

所以这张图支持的核心结论是：

- 这一版不是算力没有被调用，而是大量执行机会都耗在了访存和地址生成上
- `LDG` 占比高，和后面 full report 里 memory-bound 的结论是一致的
- 它非常适合作为 coalesced 版和 SMEM 版的对照起点

如果以后继续补图，这一版最适合对照的是：

- 和 `Global_Memory_Coalescing_kernel` 比 `LDG` 主导程度有没有缓和
- 和 `SMEM_kernel` 比 `LDS` / `STS` 是否开始取代 `LDG` 成为显著成分
