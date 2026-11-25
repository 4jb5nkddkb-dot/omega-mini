Ω v3.3

受约束决策与安全学习框架

Ω v3.3 是一个基于 Python 标准库实现的实验型受约束决策与安全控制框架。

项目重点不是无限制地执行动作，而是在明确的能力边界、授权条件、不确定性和硬约束下进行受控决策。

系统主要支持：

* APPLY
* HOLD
* ESCALATE
* Owner Authorization
* Immutable Capability Boundary
* Single-use Token
* Prediction / Observation
* Uncertainty
* Attribution
* Bounded Calibration
* Pause / Resume
* Dry-run
* Replay
* Reflection
* Persistence
* Audit Hash Chain

⸻

1. 项目结构

Omega/
│
├── omega.py
├── test_omega.py
├── experiment.py
├── omega_results.json
├── omega_results.md
└── 项目报告.docx

omega.py

Ω v3.3 核心实现。

test_omega.py

标准库测试程序。

当前包含 25 项测试。

experiment.py

实验程序，用于运行不同环境条件下的决策实验。

omega_results.json

机器可读实验结果。

omega_results.md

Markdown 格式实验结果。

⸻

2. 环境要求

建议：

Python 3.10+

项目核心测试不依赖 pytest。

因此不需要执行：

pip install pytest

直接使用 Python 标准库即可。

⸻

3. 基础自检

运行：

python omega.py

如果系统正常，应显示：

Omega self-test: PASS

⸻

4. 完整测试

运行：

python test_omega.py

当前版本实际测试结果：

TOTAL : 25
PASS  : 25
FAIL  : 0

最终：

RESULT: ALL TESTS PASSED

⸻

5. 实验

运行：

python experiment.py --rounds 20

实验包含：

stable
model_bias
sensor_spike
input_uncertainty
constraint_violation
long_run

⸻

6. 实验结果

当前实际运行结果：

stable
rounds=20
APPLY=20
HOLD=0
ESC=0
error=0.0000
model_bias
rounds=20
APPLY=0
HOLD=20
ESC=0
error=4.0000
sensor_spike
rounds=20
APPLY=0
HOLD=0
ESC=20
error=0.9000
input_uncertainty
rounds=20
APPLY=10
HOLD=10
ESC=0
error=0.0000
constraint_violation
rounds=1
APPLY=0
HOLD=0
ESC=1
error=78.0000
long_run
rounds=20
APPLY=20
HOLD=0
ESC=0
error=0.2839

⸻

7. Dry-run

Dry-run 用于验证执行前的安全状态。

当前实际结果：

{
  "decision": "HOLD",
  "reason": "dry_run",
  "environment_calls": 0,
  "state_unchanged": true,
  "token_preserved": true
}

这意味着：

* 没有调用环境；
* 系统状态没有变化；
* Token 没有被消耗。

⸻

8. Audit

当前 Audit 实验：

{
  "audit_length": 8,
  "valid_before_tampering": true,
  "valid_after_tampering": false
}

说明：

正常 Audit：验证通过
篡改 Audit 数据后：验证失败

⸻

9. 核心决策

Ω v3.3 使用三类主要决策：

APPLY
HOLD
ESCALATE

APPLY

满足正常运行条件时执行。

HOLD

条件不足或不确定性较高时暂缓执行。

ESCALATE

检测到硬约束违反或其他需要升级处理的安全状态时进入升级状态。

⸻

10. Single-use Token

真实执行需要有效 Token。

Token 使用后不能重复使用。

重复使用已经消费的 Token 会被拒绝，并不会再次调用环境。

⸻

11. Capability Boundary

系统使用 Immutable Boundary 限制可执行能力。

当前测试验证了：

Boundary 不可修改
未经授权的 Capability 不可获得

⸻

12. Pause / Resume

系统支持暂停：

pause()

暂停状态下不会执行普通环境操作。

恢复：

resume()

之后系统可以重新进行正常决策。

⸻

13. Persistence

系统支持状态保存和恢复。

保存内容包括运行状态、Episode、Audit 等数据。

Capability Token 不进行持久化恢复。

⸻

14. Replay

Replay 可以用于读取历史运行记录。

可以根据历史 Episode 对过去的运行过程进行检查。

⸻

15. Reflection

Reflection 用于汇总运行结果，包括：

* Episode 数量；
* APPLY 数量；
* HOLD 数量；
* ESCALATE 数量；
* 平均误差；
* Attribution；
* View；
* Pause 状态；
* Escalation 状态。

⸻

16. 项目验证状态

当前版本：

Ω v3.3

核心测试：

25 / 25 PASS

实验：

6 scenarios completed

Dry-run：

PASS

Audit：

normal verification = PASS
tampering detection = PASS

⸻

17. 注意事项

本项目属于实验型软件框架。

当前实验结果只代表当前代码版本、参数和测试环境下的行为。

实验结果不能直接推导出系统在所有真实环境中的表现。

如果用于生产环境，还需要进一步进行：

* 安全审计；
* 压力测试；
* 随机测试；
* 故障注入；
* 权限管理；
* 日志保护；
* 更复杂的环境验证；
* 长时间运行测试。

⸻

18. 快速开始

完整验证可以按照以下顺序执行：

python omega.py

然后：

python test_omega.py

最后：

python experiment.py --rounds 20

如果三个步骤均正常完成，则可以查看：

omega_results.json
omega_results.md

以及项目报告。

⸻

19. 当前最终验证

========================================================================
Ω v3.3 FINAL VERIFICATION
========================================================================
Core Test Suite
TOTAL : 25
PASS  : 25
FAIL  : 0
Experiment Scenarios
stable                   PASS
model_bias               PASS
sensor_spike             PASS
input_uncertainty        PASS
constraint_violation     PASS
long_run                 PASS
Dry-run                  PASS
Audit
valid_before_tampering   true
valid_after_tampering    false
========================================================================
Ω v3.3 VERIFICATION COMPLETE
========================================================================