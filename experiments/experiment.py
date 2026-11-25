"""
Ω v3.3
Experiment Runner

用途：
1. 稳定环境实验
2. 模型偏差实验
3. 传感器异常实验
4. 输入不确定性实验
5. 约束违反实验
6. 连续运行稳定性实验
7. 统计 APPLY / HOLD / ESCALATE
8. 统计 attribution
9. 输出 JSON / Markdown 实验报告

运行：
    python experiment.py

可选：
    python experiment.py --rounds 20
    python experiment.py --output results.json
    python experiment.py --markdown results.md
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from omega import (
    Decision,
    Hypothesis,
    Input,
    Omega,
    OmegaConfig,
    Prediction,
    StaticEnv,
    View,
)


# ============================================================
# Data structures
# ============================================================

@dataclass
class TrialResult:
    name: str
    decision: str
    attribution: Optional[str]
    error: float
    uncertainty: Dict[str, float]
    difference: Dict[str, float]
    calibration_delta: Dict[str, float]
    view: str
    reason: str
    environment_calls: int
    epoch: int


@dataclass
class ExperimentSummary:
    name: str
    rounds: int
    apply: int
    hold: int
    escalate: int
    sensor_attribution: int
    model_attribution: int
    input_attribution: int
    unknown_attribution: int
    mean_error: float
    max_error: float
    mean_uncertainty: float
    max_calibration_step: float
    final_view: str
    final_epoch: int
    duration_seconds: float


# ============================================================
# Utility functions
# ============================================================

def create_omega() -> tuple[Omega, Any]:
    """
    Create a fresh Ω instance for each independent experiment.
    """

    config = OmegaConfig(
        cal_step_max=0.10,
        cal_abs_max=2.00,
        bad_rounds_to_leave_shan=3,
        good_rounds_to_shan_again=3,
    )

    omega = Omega(
        secret="experiment-owner-secret",
        config=config,
    )

    owner = omega.owner_handle()

    omega.set_goal(
        owner,
        {
            "temperature": 22.0,
        },
    )

    omega.set_limits(
        owner,
        hard_min={
            "temperature": 10.0,
        },
        hard_max={
            "temperature": 30.0,
        },
    )

    return omega, owner


def new_token(
    omega: Omega,
    owner: Any,
) -> str:
    return omega.new_token(
        owner,
        "measure",
    )


def run_once(
    omega: Omega,
    owner: Any,
    *,
    expected: Dict[str, float],
    observed: Dict[str, float],
    confidence: float = 0.95,
    external: Optional[Input] = None,
) -> TrialResult:

    token = new_token(
        omega,
        owner,
    )

    env = StaticEnv(
        observed,
    )

    started = time.perf_counter()

    result = omega.run(
        env,
        token,
        prediction=Prediction(
            expected=expected,
            conf=confidence,
        ),
        external=external,
    )

    elapsed = (
        time.perf_counter()
        - started
    )

    attribution = getattr(
        result,
        "attribution",
        None,
    )

    if hasattr(attribution, "value"):
        attribution = attribution.value

    decision = result.decision

    if hasattr(decision, "value"):
        decision = decision.value

    view = omega.s.view

    if hasattr(view, "value"):
        view = view.value

    reason = getattr(
        result,
        "reason",
        "",
    )

    error = float(
        getattr(
            result,
            "error",
            0.0,
        )
    )

    uncertainty = getattr(
        result,
        "uncertainty",
        {},
    ) or {}

    difference = getattr(
        result,
        "difference",
        {},
    ) or {}

    calibration_delta = getattr(
        result,
        "calibration_delta",
        {},
    ) or {}

    return TrialResult(
        name="trial",
        decision=str(decision),
        attribution=attribution,
        error=error,
        uncertainty={
            str(k): float(v)
            for k, v in uncertainty.items()
            if isinstance(v, (int, float))
        },
        difference={
            str(k): float(v)
            for k, v in difference.items()
            if isinstance(v, (int, float))
        },
        calibration_delta={
            str(k): float(v)
            for k, v in calibration_delta.items()
            if isinstance(v, (int, float))
        },
        view=str(view),
        reason=str(reason),
        environment_calls=env.calls,
        epoch=int(
            omega.s.epoch
        ),
    )


def max_abs_value(
    mapping: Dict[str, float],
) -> float:
    if not mapping:
        return 0.0

    return max(
        abs(value)
        for value in mapping.values()
    )


def mean_or_zero(
    values: List[float],
) -> float:
    if not values:
        return 0.0

    return statistics.mean(
        values
    )


# ============================================================
# Summary
# ============================================================

def summarize(
    name: str,
    trials: List[TrialResult],
    omega: Omega,
    duration: float,
) -> ExperimentSummary:

    decisions = [
        trial.decision
        for trial in trials
    ]

    attributions = [
        trial.attribution
        for trial in trials
    ]

    errors = [
        trial.error
        for trial in trials
    ]

    uncertainties = []

    for trial in trials:
        if trial.uncertainty:
            uncertainties.append(
                max(
                    trial.uncertainty.values()
                )
            )

    calibration_steps = []

    for trial in trials:
        calibration_steps.append(
            max_abs_value(
                trial.calibration_delta
            )
        )

    view = omega.s.view

    if hasattr(view, "value"):
        view = view.value

    return ExperimentSummary(
        name=name,
        rounds=len(trials),
        apply=sum(
            1
            for x in decisions
            if x == "APPLY"
        ),
        hold=sum(
            1
            for x in decisions
            if x == "HOLD"
        ),
        escalate=sum(
            1
            for x in decisions
            if x == "ESCALATE"
        ),
        sensor_attribution=sum(
            1
            for x in attributions
            if x == Hypothesis.SENSOR.value
        ),
        model_attribution=sum(
            1
            for x in attributions
            if x == Hypothesis.MODEL.value
        ),
        input_attribution=sum(
            1
            for x in attributions
            if x == Hypothesis.INPUT.value
        ),
        unknown_attribution=sum(
            1
            for x in attributions
            if x is None
            or x == "None"
            or x == "unknown"
        ),
        mean_error=mean_or_zero(
            errors
        ),
        max_error=max(
            errors,
            default=0.0,
        ),
        mean_uncertainty=mean_or_zero(
            uncertainties
        ),
        max_calibration_step=max(
            calibration_steps,
            default=0.0,
        ),
        final_view=str(view),
        final_epoch=int(
            omega.s.epoch
        ),
        duration_seconds=duration,
    )


# ============================================================
# Experiment 1 — Stable environment
# ============================================================

def experiment_stable(
    rounds: int,
) -> tuple[ExperimentSummary, List[TrialResult]]:

    omega, owner = create_omega()

    trials = []

    started = time.perf_counter()

    for _ in range(rounds):

        trial = run_once(
            omega,
            owner,
            expected={
                "temperature": 22.0,
            },
            observed={
                "temperature": 22.0,
            },
            confidence=0.95,
        )

        trial.name = "stable"

        trials.append(
            trial
        )

    duration = (
        time.perf_counter()
        - started
    )

    return (
        summarize(
            "stable",
            trials,
            omega,
            duration,
        ),
        trials,
    )


# ============================================================
# Experiment 2 — Model bias
# ============================================================

def experiment_model_bias(
    rounds: int,
) -> tuple[ExperimentSummary, List[TrialResult]]:

    omega, owner = create_omega()

    trials = []

    started = time.perf_counter()

    for _ in range(rounds):

        trial = run_once(
            omega,
            owner,
            expected={
                "temperature": 20.0,
            },
            observed={
                "temperature": 24.0,
            },
            confidence=0.95,
        )

        trial.name = "model_bias"

        trials.append(
            trial
        )

    duration = (
        time.perf_counter()
        - started
    )

    return (
        summarize(
            "model_bias",
            trials,
            omega,
            duration,
        ),
        trials,
    )


# ============================================================
# Experiment 3 — Sensor spike
# ============================================================

def experiment_sensor_spike(
    rounds: int,
) -> tuple[ExperimentSummary, List[TrialResult]]:

    omega, owner = create_omega()

    trials = []

    started = time.perf_counter()

    # Establish baseline first.
    warmup = max(
        3,
        min(rounds, 5),
    )

    for _ in range(warmup):

        run_once(
            omega,
            owner,
            expected={
                "temperature": 22.0,
            },
            observed={
                "temperature": 22.0,
            },
            confidence=0.95,
        )

    for index in range(rounds):

        if index % 4 == 0:
            observed = {
                "temperature": 40.0,
            }
        else:
            observed = {
                "temperature": 22.0,
            }

        trial = run_once(
            omega,
            owner,
            expected={
                "temperature": 22.0,
            },
            observed=observed,
            confidence=0.95,
        )

        trial.name = "sensor_spike"

        trials.append(
            trial
        )

    duration = (
        time.perf_counter()
        - started
    )

    return (
        summarize(
            "sensor_spike",
            trials,
            omega,
            duration,
        ),
        trials,
    )


# ============================================================
# Experiment 4 — Input uncertainty
# ============================================================

def experiment_input_uncertainty(
    rounds: int,
) -> tuple[ExperimentSummary, List[TrialResult]]:

    omega, owner = create_omega()

    trials = []

    started = time.perf_counter()

    for index in range(rounds):

        confidence = (
            0.20
            if index % 2 == 0
            else 0.95
        )

        external = Input(
            content={
                "temperature": 22.0,
            },
            conf=confidence,
            source="external_sensor",
        )

        trial = run_once(
            omega,
            owner,
            expected={
                "temperature": 22.0,
            },
            observed={
                "temperature": 22.0,
            },
            confidence=0.95,
            external=external,
        )

        trial.name = "input_uncertainty"

        trials.append(
            trial
        )

    duration = (
        time.perf_counter()
        - started
    )

    return (
        summarize(
            "input_uncertainty",
            trials,
            omega,
            duration,
        ),
        trials,
    )


# ============================================================
# Experiment 5 — Constraint violation
# ============================================================

def experiment_constraint_violation(
    rounds: int,
) -> tuple[ExperimentSummary, List[TrialResult]]:

    omega, owner = create_omega()

    trials = []

    started = time.perf_counter()

    for _ in range(rounds):

        trial = run_once(
            omega,
            owner,
            expected={
                "temperature": 22.0,
            },
            observed={
                "temperature": 100.0,
            },
            confidence=0.95,
        )

        trial.name = "constraint_violation"

        trials.append(
            trial
        )

        # Once escalation locks the system,
        # later rounds should demonstrate
        # that execution remains blocked.
        if omega.s.escalated:
            break

    duration = (
        time.perf_counter()
        - started
    )

    return (
        summarize(
            "constraint_violation",
            trials,
            omega,
            duration,
        ),
        trials,
    )


# ============================================================
# Experiment 6 — Long-run stability
# ============================================================

def experiment_long_run(
    rounds: int,
) -> tuple[ExperimentSummary, List[TrialResult]]:

    omega, owner = create_omega()

    trials = []

    started = time.perf_counter()

    for index in range(rounds):

        # Small deterministic oscillation.
        observed = (
            22.0
            + 0.5
            * math.sin(
                index / 5.0
            )
        )

        trial = run_once(
            omega,
            owner,
            expected={
                "temperature": 22.0,
            },
            observed={
                "temperature": observed,
            },
            confidence=0.95,
        )

        trial.name = "long_run"

        trials.append(
            trial
        )

        if omega.s.escalated:
            break

    duration = (
        time.perf_counter()
        - started
    )

    return (
        summarize(
            "long_run",
            trials,
            omega,
            duration,
        ),
        trials,
    )


# ============================================================
# Experiment 7 — Dry-run safety
# ============================================================

def experiment_dry_run() -> Dict[str, Any]:

    omega, owner = create_omega()

    token = new_token(
        omega,
        owner,
    )

    env = StaticEnv(
        {
            "temperature": 22.0,
        }
    )

    before = omega.snapshot()

    result = omega.run(
        env,
        token,
        prediction=Prediction(
            expected={
                "temperature": 22.0,
            },
            conf=0.95,
        ),
        dry_run=True,
    )

    after = omega.snapshot()

    decision = result.decision

    if hasattr(
        decision,
        "value",
    ):
        decision = decision.value

    return {
        "decision": str(
            decision
        ),
        "reason": getattr(
            result,
            "reason",
            "",
        ),
        "environment_calls": env.calls,
        "state_unchanged": (
            before["episodes"]
            == after["episodes"]
        ),
        "token_preserved": (
            token in omega.s.tokens
        ),
    }


# ============================================================
# Experiment 8 — Audit integrity
# ============================================================

def experiment_audit() -> Dict[str, Any]:

    omega, owner = create_omega()

    for _ in range(3):

        run_once(
            omega,
            owner,
            expected={
                "temperature": 22.0,
            },
            observed={
                "temperature": 22.0,
            },
            confidence=0.95,
        )

    valid_before = (
        omega.verify_audit()
    )

    audit_length = len(
        omega.s.audit
    )

    if omega.s.audit:
        omega.s.audit[0]["p"][
            "tampered"
        ] = True

    valid_after = (
        omega.verify_audit()
    )

    return {
        "audit_length": audit_length,
        "valid_before_tampering": valid_before,
        "valid_after_tampering": valid_after,
    }


# ============================================================
# Experiment runner
# ============================================================

def run_all(
    rounds: int,
) -> Dict[str, Any]:

    experiments = [
        experiment_stable,
        experiment_model_bias,
        experiment_sensor_spike,
        experiment_input_uncertainty,
        experiment_constraint_violation,
        experiment_long_run,
    ]

    summaries = []
    trials = []

    for experiment in experiments:

        summary, result = experiment(
            rounds
        )

        summaries.append(
            asdict(summary)
        )

        trials.extend(
            asdict(item)
            for item in result
        )

    dry_run = (
        experiment_dry_run()
    )

    audit = (
        experiment_audit()
    )

    return {
        "project": "Omega v3.3",
        "timestamp": time.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "rounds_requested": rounds,
        "summaries": summaries,
        "trials": trials,
        "dry_run": dry_run,
        "audit": audit,
    }


# ============================================================
# Markdown report
# ============================================================

def markdown_report(
    results: Dict[str, Any],
) -> str:

    lines = []

    lines.append(
        "# Ω v3.3 实验报告"
    )
    lines.append("")

    lines.append(
        "## 1. 实验概览"
    )
    lines.append("")

    lines.append(
        f"- 项目：{results['project']}"
    )
    lines.append(
        f"- 实验时间：{results['timestamp']}"
    )
    lines.append(
        f"- 每组请求轮数：{results['rounds_requested']}"
    )
    lines.append("")

    lines.append(
        "## 2. 实验汇总"
    )
    lines.append("")

    lines.append(
        "| 实验 | 轮数 | APPLY | HOLD | ESCALATE | 平均误差 | 最大误差 | 平均不确定性 | 最大校准步长 | 最终 View |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"
    )

    for item in results[
        "summaries"
    ]:

        lines.append(
            "| {name} | {rounds} | {apply} | {hold} | {escalate} | {mean_error:.4f} | {max_error:.4f} | {mean_uncertainty:.4f} | {max_calibration_step:.4f} | {final_view} |".format(
                **item
            )
        )

    lines.append("")

    lines.append(
        "## 3. Dry-run 安全性"
    )
    lines.append("")

    dry = results[
        "dry_run"
    ]

    lines.append(
        f"- Decision：`{dry['decision']}`"
    )
    lines.append(
        f"- Reason：`{dry['reason']}`"
    )
    lines.append(
        f"- Environment calls：`{dry['environment_calls']}`"
    )
    lines.append(
        f"- State unchanged：`{dry['state_unchanged']}`"
    )
    lines.append(
        f"- Token preserved：`{dry['token_preserved']}`"
    )
    lines.append("")

    lines.append(
        "## 4. Audit 完整性"
    )
    lines.append("")

    audit = results[
        "audit"
    ]

    lines.append(
        f"- Audit records：{audit['audit_length']}"
    )
    lines.append(
        f"- Tampering 前验证：`{audit['valid_before_tampering']}`"
    )
    lines.append(
        f"- Tampering 后验证：`{audit['valid_after_tampering']}`"
    )
    lines.append("")

    lines.append(
        "## 5. 结果解释"
    )
    lines.append("")

    lines.append(
        "实验结果用于验证 Ω 的状态机、边界、Token、预测误差、归因、校准、暂停/升级和审计机制是否按照设计运行。"
    )

    lines.append("")

    lines.append(
        "具体性能结论应以实际运行生成的数值为准，不应在程序运行前预设实验结果。"
    )

    lines.append("")

    return "\n".join(
        lines
    )


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Ω v3.3 experiment runner"
    )

    parser.add_argument(
        "--rounds",
        type=int,
        default=20,
        help="每组实验轮数",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="omega_results.json",
        help="JSON 输出路径",
    )

    parser.add_argument(
        "--markdown",
        type=str,
        default="omega_results.md",
        help="Markdown 输出路径",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    if args.rounds <= 0:
        raise ValueError(
            "--rounds 必须大于 0"
        )

    results = run_all(
        args.rounds
    )

    json_path = Path(
        args.output
    )

    markdown_path = Path(
        args.markdown
    )

    json_path.write_text(
        json.dumps(
            results,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    markdown_path.write_text(
        markdown_report(
            results
        ),
        encoding="utf-8",
    )

    print(
        "=" * 72
    )
    print(
        "Ω v3.3 EXPERIMENT COMPLETE"
    )
    print(
        "=" * 72
    )

    for item in results[
        "summaries"
    ]:

        print(
            f"{item['name']:24s} "
            f"rounds={item['rounds']:3d} "
            f"APPLY={item['apply']:3d} "
            f"HOLD={item['hold']:3d} "
            f"ESC={item['escalate']:3d} "
            f"error={item['mean_error']:.4f}"
        )

    print(
        "-" * 72
    )

    print(
        "Dry-run:"
    )

    print(
        json.dumps(
            results["dry_run"],
            ensure_ascii=False,
            indent=2,
        )
    )

    print(
        "-" * 72
    )

    print(
        "Audit:"
    )

    print(
        json.dumps(
            results["audit"],
            ensure_ascii=False,
            indent=2,
        )
    )

    print(
        "-" * 72
    )

    print(
        f"JSON    : {json_path}"
    )

    print(
        f"Markdown: {markdown_path}"
    )


if __name__ == "__main__":
    main()