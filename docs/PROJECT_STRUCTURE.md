# Project Structure

本仓库采用“每个优化阶段独立可读、可编译、可比较”的教学结构。v1.0 没有把
所有 kernel 合并为一个模板，也没有重编号历史目录。

```text
GEMM_For_Myself/
├── README.md
├── PERFORMANCE_SUMMARY.md
├── TODO.md
├── Makefile
├── docs/assets/             # generated comparison SVG/PNG
├── results/rtx4060_laptop/  # controlled same-script CSV
├── scripts/
├── 1.naive_kernel/ ... 8.Autoing_kernel/
├── 10.Wraptiling_kernel/
├── 11.Double_Buffering/
└── cuBLAS_baseline/
```

## Stage Numbering

当前历史仓库没有 Stage 9。Git 历史中没有找到足以解释这一编号空缺的明确记录，
因此 v1.0 保留 Stage 10 和 Stage 11 的原编号，不创建虚构阶段，也不重编号。

Stage 8 之后是两个高级实验方向：Warp Tiling 和 Double Buffering。Stage 11
的模板不包含 Stage 10 的 `WM/WN/WMITER/WNITER`，因此两者不是严格的线性继承。

## File Roles

- Stage README：本阶段目标、结构、构建、正确性、性能与 profiler 结论。
- `*.cu`：kernel、benchmark 或设备工具。权威关系见
  `REORGANIZATION_REPORT.md`。
- `notes/`：用户学习笔记和教学资产，只分类，不自动重写。
- `results/`：CSV、console output 和实验日志；`rtx4060_laptop/` 保存同轮主比较。
- `profiling/raw/`：原始 `.ncu-rep` 证据。
- `profiling/full_report/`：Full report 截图与解释。
- `profiling/instr_report/`：InstructionStats 截图与解释。
- `build/`：本地生成的 ELF，不提交 Git。

## Historical Names

`Autoing`、`Wraptiling`、`kerenl`、`benchmarker` 等拼写保留在历史路径中，避免
破坏已有命令和 Git 历史。README 正文统一使用 Autotuning、Warp Tiling、kernel
等正确技术名称。未来若重命名，应单独进行一次可审计迁移。

## Why Benchmark Code Is Not Fully Shared

Stage 1–5 的 benchmark 内嵌了历史 kernel 实现，部分还与独立源文件存在结构差异。
为了不在项目封版时改变算法行为，v1.0 保留这些副本并明确标注实际 benchmark
入口。Stage 6–8、10、11 已直接 include 对应权威 kernel。后续去重必须逐阶段
完成等价性证明、编译和 correctness 验证。
