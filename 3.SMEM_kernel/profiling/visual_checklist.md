# SMEM Kernel Visual Checklist

这一版最值得放的图，是那些能说明“为什么 shared memory 没有自动把性能拉开”的图。

## 指令报告值得放的图

- `Instruction Statistics` 整页合并图
  一张图同时保留统计表和 `Executed Instruction Mix` 柱状图即可。
- `Source`
  改为可选，不再强制要求。

## 完整报告值得放的图

- `Shared Memory` 相关页面
  如果有 shared load/store、bank conflict、throughput 类指标，很值得保留。
- `Occupancy`
  `32x32 = 1024 threads/block` 这一点很值得结合 occupancy 图来讲。
- `Scheduler Statistics` / `Warp State`
  看 stall 是不是开始有 barrier / memory dependency 成分。
- `Speed Of Light`
  看是不是仍然没有明显逼近更高 compute 利用率。
- `Summary`
  保留时间、block 配置和 kernel 名。

## 这一版特别值得讲的观察

- shared memory 带来了复用，但也带来了同步和额外访存开销。
- 如果和 coalesced 版几乎打平，那么 occupancy、同步、每线程只算一个输出值这些点都值得截图支撑。
- 这一版最适合引出下一步：register blocking / block tiling / vectorized access。
