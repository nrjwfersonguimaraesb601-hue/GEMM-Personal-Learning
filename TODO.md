# TODO

这个文件按当前项目的实际进度维护下一步路线。

## 已完成

- [x] `naive_kernel` baseline
- [x] `naive_kernel` benchmark
- [x] `Global_Memory_Coalescing_kernel`
- [x] `Global_Memory_Coalescing_kernel` benchmark
- [x] `SMEM_kernel`
- [x] `SMEM_kernel` benchmark
- [x] `SMEM_kernel` teacher notes
- [x] `1D_Blocktiling_kernel`
- [x] `1D_Blocktiling_kernel` benchmark
- [x] `1D_Blocktiling_kernel` profiling 目录、截图和说明整理
- [x] 四版 kernel 的 README 按最新测速结果重写
- [x] 根目录总结文档按统一 benchmark 口径更新

## 下一步优先级

- [ ] `2D_Blocktiling_kernel`
  - [ ] 让每个 thread 在两个方向上都负责多个输出值
  - [ ] 和当前 `1D_Blocktiling_kernel` 做正面对比

- [ ] `Vectorized_Memory_Access_kernel`
  - [ ] 尝试 `float4` load/store
  - [ ] 检查对齐要求和实际收益

- [ ] `WarpTiling_kernel`
  - [ ] 引入 warp-level 分工
  - [ ] 看能不能继续把 `3 TFLOPS+` 再往上推

- [ ] `cuBLAS` baseline
  - [ ] 补一份统一尺寸下的 cuBLAS 对照
  - [ ] 记录相对 cuBLAS 的百分比

## 持续补的内容

- [ ] 用统一口径继续复测所有版本
- [ ] 增加 block size 对比实验
- [ ] 画 `GFLOPS vs size` 曲线
- [ ] 画 `runtime vs size` 曲线
- [ ] 补 `1D_Blocktiling_kernel` 的更多 Nsight Compute 页面
  - [ ] occupancy
  - [ ] shared-memory 相关页面
- [ ] 把 correctness benchmark 和 pure speed benchmark 分开记录得更清楚

## 每一版完成标准

每新增一个 kernel 版本，至少补齐下面这些内容：

- [x] correctness 版本可运行
- [x] benchmark 版本可运行
- [x] README 记录核心变化
- [x] README 记录最新实测数据
- [x] 和上一版做性能对比
