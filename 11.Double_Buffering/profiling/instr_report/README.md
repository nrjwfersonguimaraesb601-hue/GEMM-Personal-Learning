# Double Buffering Instruction Report

InstructionStats 记录：

- Executed instructions：37746688
- Issued instructions：37750655
- Avg. executed instructions/scheduler：393194.67
- Avg. issued instructions/scheduler：393235.99

截图中 FFMA 占主导，符合 `TM×TN` register outer product。LDS、地址计算和控制
指令反映双 stage shared-memory 访问与 pipeline 管理。具体向量指令宽度应以 SASS
为准，不仅凭源码推断。
