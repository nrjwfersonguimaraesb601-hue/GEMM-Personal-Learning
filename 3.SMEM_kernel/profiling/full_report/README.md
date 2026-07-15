# SMEM Full Report

把完整报告截图放到 `figures/`。

当前保留：

- `01_summary.png`
- `02_speed_of_light.png`
- `03_memory_workload.png`
- `04_scheduler_statistics.png`
- `05_warp_state.png`

如果后面补图，这一版再额外建议：

- `06_occupancy.png`
- `07_shared_memory.png`

当前这套 README 已经适配成“先放核心 5 张 full report 图片” 的格式，不要求一开始就把 occupancy 和 shared-memory 页补齐。

## 当前解读

### `01_summary.png`

这一页说明 shared-memory 版本已经不是 naive 那种 baseline 状态了，但也没有像 coalesced 版那样把两边都拉到接近满值：

- kernel time 大约 `2.47 ms`
- compute throughput `80.15%`
- memory throughput `80.15%`

下面的 opportunities 也很有代表性：

- `Mio Throttle Stalls`
- `Theoretical Occupancy`
- `Shared Store Bank Conflicts`

这几条几乎就是 shared-memory kernel 的“问题画像”：

- 片上 memory pipeline 很忙
- occupancy 受到约束
- shared store 还有 bank conflict

### `02_speed_of_light.png`

这一张图很适合和 coalesced 版一起看：

- coalesced 版大约是 `97.82% / 97.82%`
- 这里是 `80.15% / 80.15%`

这说明 shared memory 这一步虽然走通了，但没有自动把吞吐再推高一大截。  
对学习项目来说，这个结果反而很有价值，因为它提醒我们：

- shared memory 不是“加上就一定更快”
- 数据复用带来的收益，会和同步、occupancy、bank conflict 的代价一起出现

### `03_memory_workload.png`

这一页是 shared-memory 版本最有信息量的一张之一：

- `Shared Store Bank Conflicts` 被明确指出
- 平均大约 `1.3-way` bank conflict
- shared store 请求里有可见的冲突成本

同时图里还能看到：

- shared memory traffic 已经很重
- L1/TEX hit rate 甚至为 `0%`
- 数据路径明显和前两版不一样

这说明这版已经把 block 内复用做起来了，但 shared-memory 访问模式本身还不够理想。

### `04_scheduler_statistics.png`

调度层面比 naive 好，但和 coalesced 相比没有明显占优：

- `Eligible Warps Per Scheduler` 约 `1.14`
- `Issued Warp Per Scheduler` 约 `0.26`
- `No Eligible` 约 `73.64%`

这个数值和 coalesced 版在同一档附近，说明：

- shared memory 并没有把 warp 的发射条件明显改善到新台阶
- 至少在这个实现里，收益被额外开销吃掉了一部分

### `05_warp_state.png`

这一页直接把 stall 主因指出来了：

- `Stall MIO Throttle` 大约 `16.91`
- `Stall Barrier` 也已经出现
- `Stall Long Scoreboard`、`Stall Wait` 仍然存在

和前两版对比的话：

- naive 主要是 `LG Throttle`
- coalesced 还是 `LG Throttle`，但大幅减轻
- SMEM 这里开始转成 `MIO Throttle`

这说明瓶颈类型已经发生了变化：

- 不是纯 global-memory 访问效率问题了
- 而是 shared-memory / memory input-output pipeline 本身开始成为限制因素

这一版 full report 最适合写成这样的总结：

- shared memory 确实把数据复用引进来了
- 但当前实现还带着 occupancy、bank conflict、MIO throttle 这些新的代价
- 它是一个很好的 checkpoint，但还不是成熟的高性能终点
