# Warp Tiling Profiling

采集对象为
`sgemm_2d_register_tiling<128,128,16,64,64,8,4,2,2>`，workload 为
`1024³`，grid `(8,8,1)`，block `(128,1,1)`。原始报告：

- [Full report](./raw/Wraptiling_full.ncu-rep)
- [InstructionStats](./raw/Wraptiling_instr.ncu-rep)

## Key Findings

| Metric | Value |
| --- | ---: |
| Compute throughput | 52.46% |
| Memory throughput | 33.03% |
| Registers/thread | 168 |
| Static shared memory | 16.38 KiB |
| Theoretical / achieved occupancy | 25.00% / 22.33% |
| Active / eligible / issued warps per scheduler | 2.68 / 1.00 / 0.54 |
| No Eligible | 45.84% |
| Warp cycles / issued instruction | 4.94 |
| Shared-store bank conflicts | 786432, about 4-way |
| Executed instructions | 40429568 |

报告表明当前 warp tile 方案受 register、shared memory、occupancy 和 shared-store
访问共同限制。正式 `4096³` Avg 只有 6.671 TFLOPS，因此该阶段被保留为负优化
案例。

## Reading Order

1. [Full report notes](./full_report/README.md)
2. [Instruction report notes](./instr_report/README.md)
3. [Capture commands](./capture_commands.md)
4. [Visual checklist](./visual_checklist.md)

目前原始报告完整，但尚未从 `ncu-ui` 导出分类截图。v1.0 不用伪造截图填补该
缺口；可按 visual checklist 人工补充。
