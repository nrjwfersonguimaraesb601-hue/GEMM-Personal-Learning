# Coalesced Kernel Visual Checklist

这一版最值得放的是“和 naive 形成因果关系”的图。

## 指令报告值得放的图

- `Instruction Statistics` 整页合并图
  一张图同时保留统计表和 `Executed Instruction Mix` 柱状图即可。
- `Source`
  改为可选，不再强制要求。

## 完整报告值得放的图

- `Memory Workload Analysis`
  这一页很关键，最适合解释 coalescing 到底改善了什么。
- `Memory Chart`
  如果图里能看出 global memory transaction 更合理，很值得保留。
- `Speed Of Light`
  观察优化后是不是离 compute side 更近一些。
- `Summary`
  保留整体时间和 launch 配置。

## 这一版特别值得讲的观察

- 和 naive 并排对比时，最适合看的不是单独 GFLOPS，而是 memory-related 图。
- 如果 `sector/request` 或 load efficiency 更好，这就是这版最有说服力的证据。
- 如果方阵提升明显、特殊 shape 不明显，也可以单独截一张 benchmark 表来辅助说明。
