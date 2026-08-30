# TODO

## v1.0 已完成

- [x] Stage 1–3：Naive、coalescing、shared-memory tiling
- [x] Stage 4–5：1D/2D register tiling
- [x] Stage 6：`float4` vectorized load/store 与对应 profiling
- [x] Stage 7：shared-memory padding 与 bank-conflict 分析
- [x] Stage 8：14 组编译期配置 quick/full autotuning、CSV 与 C00/C08 报告
- [x] Stage 10：Warp Tiling kernel、benchmark、correctness 与 Nsight Compute
  - [x] 保留其相对 Stage 8 的性能回退，作为负优化结果
- [x] Stage 11：软件 Double Buffering kernel、benchmark、correctness 与 profiling
  - [x] 明确当前版本不是 `cp.async` pipeline
- [x] cuBLAS baseline
  - [x] FP32 `CUBLAS_COMPUTE_32F` 正式基准
  - [x] TF32 Tensor Core 独立参考，不混入 FP32 排名
- [x] v1.0 工程整理
  - [x] 根 Makefile 与统一 build 输出
  - [x] correctness、benchmark、profiling 脚本分离
  - [x] results / notes / profiling 分类
  - [x] 4096³ CSV 与 SVG 性能图
  - [x] 保留原始 `.ncu-rep`、截图、日志和学习笔记

## 下一阶段优先级

- [ ] 正确性与通用性
  - [ ] 为非整除尺寸增加边界处理或独立 tail kernel
  - [ ] 增加随机矩阵、非方阵和更多维度组合的 correctness tests
  - [ ] 将 CPU reference 结果与纯测速结果分文件归档
- [ ] 可复现性能
  - [ ] 在固定 power/clock/temperature 条件下统一复测所有 FP32 kernel
  - [ ] 重复 C08、Stage 11 与 cuBLAS FP32，报告均值、方差和置信区间
  - [ ] 增加 GFLOPS-vs-size 与 latency-vs-size 曲线
  - [ ] 记录 CUDA、driver、cuBLAS 与 Nsight Compute 版本
- [ ] Warp Tiling 复盘
  - [ ] 对 Stage 10 的 168 registers/thread、25% 理论 occupancy 做 ablation
  - [ ] 重构 A 的 warp-level cooperative load，降低 shared-store conflict
  - [ ] 分别测量 warp 映射、load mapping、tile 尺寸的独立影响
- [ ] Pipeline 实验
  - [ ] 量化 Stage 11 软件预取的 overlap 与额外寄存器开销
  - [ ] 新建独立 `cp.async` 实验，不把它混写为当前 Stage 11
  - [ ] 探索多 stage pipeline 和 barrier 管理
- [ ] Shared-memory layout 扩展
  - [ ] 新建 XOR swizzle 实验，与 padding 做相同配置对比
  - [ ] 同时比较 bank conflict、指令开销、occupancy 和正式吞吐

## 每个新版本的完成标准

- [ ] 权威 kernel 与 benchmark 关系清楚
- [ ] 小尺寸 CPU correctness check 实际执行并 `PASS`
- [ ] 多尺寸 CUDA Event benchmark 可复现
- [ ] README 记录 tile 约束、精度、编译和运行命令
- [ ] 与上一阶段做相同尺寸、相同测试口径的 Avg GFLOPS 对比
- [ ] 保存原始 CSV/log 和 Nsight Compute `.ncu-rep`
- [ ] profiler latency 与正式 benchmark 数据分开
