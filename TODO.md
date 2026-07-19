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
- [x] Stage 7: Shared-memory layout padding
  - [x] 使用 Nsight Compute 定位 shared-memory bank conflict
  - [x] 为 Bs 增加 padding 并统一物理读写 stride
  - [x] 为 As 增加 padding，进一步降低 bank conflict
  - [x] 保持原有 `float4`、2D register tiling 和计算映射
  - [x] 多尺寸 benchmark 与 CPU reference check
  - [x] InstructionStats / Full Nsight Compute 分析
  - [x] README、根目录说明和统一性能汇总
- [x] Stage 8: Compile-time autotuning
  - [x] 保持 Stage 7 kernel 的计算、`float4` 访存和 padding 布局不变
  - [x] 枚举 `BM/BN/BK/TM/TN`，比较 14 组编译期配置
  - [x] quick/full suite、CSV 输出和几何平均排名
  - [x] 14 组配置全部通过 `256^3` CPU reference check
  - [x] 保存 C00/C08 的 Nsight Compute 报告和分类截图
  - [x] 更新 Stage 8 README、根目录说明和统一性能汇总

## 下一步优先级

- [x] 分析 Vectorized kernel 的 Nsight Compute 报告
  - [x] 确认生成 `LDG.E.128` / `STG.E.128`
  - [x] 对比 2D 版本的 global-store sector 利用率
  - [x] 检查 shared-memory bank conflict 和 excessive wavefront
  - [x] 记录 registers/thread、scheduler utilization 和主要 stall reason

- [ ] 继续改进当前 GEMM kernel
  - [ ] 重新设计 A 的协作式加载映射，使一个 warp 的 global 地址更连续
  - [ ] 分离 A/B load、A transpose、C store，做 ablation benchmark
  - [ ] 增加非整除尺寸的边界处理或 tail kernel
  - [x] 初步比较 `BK=16`、更大的 block tile 和不同的 thread tile
  - [ ] 重复验证 C08/C13 等候选，并固定下一版 kernel 参数

- [ ] Warp tiling / pipeline
  - [ ] 引入 warp-level 分工
  - [ ] 尝试 double buffering，重叠 global-to-shared load 与计算
  - [ ] 评估 `cp.async` 作为后续实验方向

- [ ] cuBLAS baseline
  - [ ] 在统一尺寸和相同数据类型下测试 cuBLAS SGEMM
  - [ ] 记录各版本达到 cuBLAS 性能的百分比

- [ ] Shared-memory layout 扩展（暂缓）
  - [ ] 学习 XOR swizzle 的地址映射原理
  - [ ] 实现独立的 XOR-swizzled kernel 和 benchmark
  - [ ] 在相同参数下与 padding 版本对比 bank conflict、指令开销和性能

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
