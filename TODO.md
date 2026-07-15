# TODO

这个文件按当前项目的实际进度维护后续学习路线。

## 已完成

- [x] Stage 1: Naive kernel、benchmark 与 profiling
- [x] Stage 2: Global memory coalescing kernel、benchmark 与 profiling
- [x] Stage 3: Shared memory tiling kernel、benchmark 与 profiling
- [x] Stage 4: 1D register tiling kernel、benchmark 与 profiling
- [x] Stage 5: 2D register tiling kernel、benchmark 与 profiling
- [x] Stage 6: Vectorized memory access kernel
  - [x] A/B 的 `float4` global load
  - [x] A tile 转置写入 shared memory
  - [x] C 的 `float4` global store
  - [x] 多尺寸 benchmark 与 CPU reference check
  - [x] InstructionStats / Full Nsight Compute 报告采集
  - [x] README 和统一性能汇总

## 下一步优先级

- [x] 分析 Vectorized kernel 的 Nsight Compute 报告
  - [x] 确认生成 `LDG.E.128` / `STG.E.128`
  - [x] 对比 2D 版本的 global-store sector 利用率
  - [x] 检查 shared-memory bank conflict 和 excessive wavefront
  - [x] 记录 registers/thread、scheduler utilization 和主要 stall reason

- [ ] 改进 Vectorized kernel
  - [ ] 重新设计 A 的协作式加载映射，使一个 warp 的 global 地址更连续
  - [ ] 分离 A/B load、A transpose、C store，做 ablation benchmark
  - [ ] 增加非整除尺寸的边界处理或 tail kernel
  - [ ] 尝试 `BK=16`、更大的 block tile 和不同的 thread tile

- [ ] Warp tiling / pipeline
  - [ ] 引入 warp-level 分工
  - [ ] 尝试 double buffering，重叠 global-to-shared load 与计算
  - [ ] 评估 `cp.async` 作为后续实验方向

- [ ] cuBLAS baseline
  - [ ] 在统一尺寸和相同数据类型下测试 cuBLAS SGEMM
  - [ ] 记录各版本达到 cuBLAS 性能的百分比

## 持续补充

- [ ] 用统一编译参数复测所有 kernel
- [ ] 把 correctness run 和 pure-speed run 分开归档
- [ ] 增加 `GFLOPS vs size` 曲线
- [ ] 增加 `runtime vs size` 曲线
- [ ] 记录 GPU power mode、频率和温度，减少 Laptop GPU 测试波动
- [ ] 整理每一阶段的 Nsight Compute 截图和结论

## 每一版完成标准

每新增一个 kernel 版本，至少完成：

- [ ] correctness check 可运行且通过
- [ ] 多尺寸 benchmark 可运行
- [ ] README 记录核心技术变化与限制
- [ ] README 记录最新实测结果
- [ ] 与上一版使用 Avg GFLOPS 做同尺寸对比
- [ ] 保存可复现的编译与 Nsight Compute 命令
