# Ω v3.3 实验报告

## 1. 实验概览

- 项目：Omega v3.3
- 实验时间：2026-11-25 02:14:09
- 每组请求轮数：20

## 2. 实验汇总

| 实验 | 轮数 | APPLY | HOLD | ESCALATE | 平均误差 | 最大误差 | 平均不确定性 | 最大校准步长 | 最终 View |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| stable | 20 | 20 | 0 | 0 | 0.0000 | 0.0000 | 0.0500 | 0.0000 | SHAN |
| model_bias | 20 | 0 | 20 | 0 | 4.0000 | 4.0000 | 0.0500 | 0.0000 | NOT_SHAN |
| sensor_spike | 20 | 0 | 0 | 20 | 0.9000 | 18.0000 | 0.0500 | 0.0000 | SHAN |
| input_uncertainty | 20 | 10 | 10 | 0 | 0.0000 | 0.0000 | 0.4250 | 0.0000 | SHAN |
| constraint_violation | 1 | 0 | 0 | 1 | 78.0000 | 78.0000 | 0.0500 | 0.0000 | SHAN |
| long_run | 20 | 20 | 0 | 0 | 0.2839 | 0.4998 | 0.0500 | 0.0500 | SHAN |

## 3. Dry-run 安全性

- Decision：`HOLD`
- Reason：`dry_run`
- Environment calls：`0`
- State unchanged：`True`
- Token preserved：`True`

## 4. Audit 完整性

- Audit records：8
- Tampering 前验证：`True`
- Tampering 后验证：`False`

## 5. 结果解释

实验结果用于验证 Ω 的状态机、边界、Token、预测误差、归因、校准、暂停/升级和审计机制是否按照设计运行。

具体性能结论应以实际运行生成的数值为准，不应在程序运行前预设实验结果。
