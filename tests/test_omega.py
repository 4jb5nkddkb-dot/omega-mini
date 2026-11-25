"""
Omega v3.3
==========

标准库测试程序。

不依赖 pytest。

运行：
    python test_omega.py

返回：
    全部通过 -> exit code 0
    有测试失败 -> exit code 1
"""

from __future__ import annotations

import copy
import sys
import tempfile
from pathlib import Path

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
# Test helpers
# ============================================================

PASSED = 0
FAILED = 0


def test(
    name,
    function,
):
    global PASSED
    global FAILED

    try:
        function()
        print(f"[PASS] {name}")
        PASSED += 1

    except Exception as exc:
        print(f"[FAIL] {name}")
        print(f"       {type(exc).__name__}: {exc}")
        FAILED += 1


def assert_true(
    condition,
    message="condition is false",
):
    if not condition:
        raise AssertionError(
            message
        )


def assert_equal(
    actual,
    expected,
    message="values are not equal",
):
    if actual != expected:
        raise AssertionError(
            f"{message}: "
            f"actual={actual!r}, "
            f"expected={expected!r}"
        )


def assert_not_equal(
    actual,
    expected,
    message="values are equal",
):
    if actual == expected:
        raise AssertionError(
            f"{message}: "
            f"value={actual!r}"
        )


def assert_raises(
    exception_type,
    function,
):
    try:
        function()
    except exception_type:
        return
    except Exception as exc:
        raise AssertionError(
            f"expected {exception_type.__name__}, "
            f"got {type(exc).__name__}"
        )
    else:
        raise AssertionError(
            f"expected {exception_type.__name__}"
        )


# ============================================================
# Factory
# ============================================================

def make_omega():

    config = OmegaConfig(
        cal_step_max=0.10,
        cal_abs_max=2.00,
        bad_rounds_to_leave_shan=3,
        good_rounds_to_shan_again=3,
    )

    omega = Omega(
        secret="test-secret-123456",
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


def run_normal(
    omega,
    owner,
    observed=22.0,
    expected=22.0,
    confidence=0.95,
):

    token = omega.new_token(
        owner,
        "measure",
    )

    env = StaticEnv(
        {
            "temperature": observed,
        }
    )

    result = omega.run(
        env,
        token,
        prediction=Prediction(
            expected={
                "temperature": expected,
            },
            conf=confidence,
        ),
    )

    return result, env


# ============================================================
# 1. Basic APPLY
# ============================================================

def test_basic_apply():

    omega, owner = make_omega()

    result, env = run_normal(
        omega,
        owner,
        observed=22.0,
        expected=22.0,
    )

    assert_equal(
        result.decision,
        Decision.APPLY,
    )

    assert_equal(
        env.calls,
        1,
    )

    assert_equal(
        result.error,
        0.0,
    )

    assert_equal(
        omega.s.epoch,
        1,
    )


# ============================================================
# 2. Token is single-use
# ============================================================

def test_single_use_token():

    omega, owner = make_omega()

    token = omega.new_token(
        owner,
        "measure",
    )

    env1 = StaticEnv(
        {
            "temperature": 22.0,
        }
    )

    result1 = omega.run(
        env1,
        token,
        prediction=Prediction(
            expected={
                "temperature": 22.0,
            },
            conf=0.95,
        ),
    )

    assert_equal(
        result1.decision,
        Decision.APPLY,
    )

    env2 = StaticEnv(
        {
            "temperature": 22.0,
        }
    )

    result2 = omega.run(
        env2,
        token,
        prediction=Prediction(
            expected={
                "temperature": 22.0,
            },
            conf=0.95,
        ),
    )

    assert_equal(
        result2.decision,
        Decision.ESCALATE,
    )

    assert_equal(
        result2.reason,
        "token_already_used",
    )

    assert_equal(
        env2.calls,
        0,
    )


# ============================================================
# 3. Invalid token
# ============================================================

def test_invalid_token():

    omega, owner = make_omega()

    env = StaticEnv(
        {
            "temperature": 22.0,
        }
    )

    result = omega.run(
        env,
        "not-a-real-token",
        prediction=Prediction(
            expected={
                "temperature": 22.0,
            },
            conf=0.95,
        ),
    )

    assert_equal(
        result.decision,
        Decision.ESCALATE,
    )

    assert_equal(
        result.reason,
        "invalid_token",
    )

    assert_equal(
        env.calls,
        0,
    )


# ============================================================
# 4. Dry-run does not execute environment
# ============================================================

def test_dry_run():

    omega, owner = make_omega()

    token = omega.new_token(
        owner,
        "measure",
    )

    before = omega.snapshot()

    env = StaticEnv(
        {
            "temperature": 22.0,
        }
    )

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

    assert_equal(
        result.reason,
        "dry_run",
    )

    assert_equal(
        env.calls,
        0,
    )

    assert_equal(
        before,
        after,
    )

    # Token must still be available.
    assert_true(
        token in omega.s.tokens,
        "dry-run consumed the token",
    )


# ============================================================
# 5. Dry-run can subsequently execute
# ============================================================

def test_dry_run_then_real_run():

    omega, owner = make_omega()

    token = omega.new_token(
        owner,
        "measure",
    )

    env = StaticEnv(
        {
            "temperature": 22.0,
        }
    )

    dry = omega.run(
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

    assert_equal(
        dry.reason,
        "dry_run",
    )

    real = omega.run(
        env,
        token,
        prediction=Prediction(
            expected={
                "temperature": 22.0,
            },
            conf=0.95,
        ),
    )

    assert_equal(
        real.decision,
        Decision.APPLY,
    )

    assert_equal(
        env.calls,
        1,
    )


# ============================================================
# 6. Hard limit violation
# ============================================================

def test_hard_limit():

    omega, owner = make_omega()

    result, env = run_normal(
        omega,
        owner,
        observed=100.0,
        expected=22.0,
    )

    assert_equal(
        result.decision,
        Decision.ESCALATE,
    )

    assert_equal(
        result.reason,
        "hard_limit_violation",
    )

    assert_true(
        omega.s.escalated,
        "system did not enter escalated state",
    )

    assert_equal(
        env.calls,
        1,
    )


# ============================================================
# 7. Escalated system stays blocked
# ============================================================

def test_escalation_blocks_followup():

    omega, owner = make_omega()

    result1, _ = run_normal(
        omega,
        owner,
        observed=100.0,
        expected=22.0,
    )

    assert_equal(
        result1.decision,
        Decision.ESCALATE,
    )

    token = omega.new_token(
        owner,
        "measure",
    )

    env = StaticEnv(
        {
            "temperature": 22.0,
        }
    )

    result2 = omega.run(
        env,
        token,
        prediction=Prediction(
            expected={
                "temperature": 22.0,
            },
            conf=0.95,
        ),
    )

    assert_equal(
        result2.decision,
        Decision.ESCALATE,
    )

    assert_equal(
        result2.reason,
        "already_escalated",
    )

    assert_equal(
        env.calls,
        0,
    )


# ============================================================
# 8. Resume after pause
# ============================================================

def test_pause_resume():

    omega, owner = make_omega()

    omega.pause(
        owner
    )

    result, env = run_normal(
        omega,
        owner,
        observed=22.0,
        expected=22.0,
    )

    assert_equal(
        result.decision,
        Decision.ESCALATE,
    )

    assert_equal(
        result.reason,
        "paused",
    )

    assert_equal(
        env.calls,
        0,
    )

    omega.resume(
        owner
    )

    result, env = run_normal(
        omega,
        owner,
        observed=22.0,
        expected=22.0,
    )

    assert_equal(
        result.decision,
        Decision.APPLY,
    )

    assert_equal(
        env.calls,
        1,
    )


# ============================================================
# 9. Prediction error produces HOLD
# ============================================================

def test_prediction_error_hold():

    omega, owner = make_omega()

    result, env = run_normal(
        omega,
        owner,
        observed=25.0,
        expected=20.0,
    )

    assert_equal(
        result.decision,
        Decision.HOLD,
    )

    assert_equal(
        result.reason,
        "prediction_error_too_large",
    )

    assert_equal(
        env.calls,
        1,
    )


# ============================================================
# 10. High uncertainty
# ============================================================

def test_high_uncertainty():

    omega, owner = make_omega()

    result, env = run_normal(
        omega,
        owner,
        observed=22.0,
        expected=22.0,
        confidence=0.05,
    )

    assert_equal(
        result.decision,
        Decision.HOLD,
    )

    assert_equal(
        result.reason,
        "high_uncertainty",
    )

    assert_true(
        result.uncertainty["total"]
        > 0.90,
        "uncertainty is not high enough",
    )


# ============================================================
# 11. External input uncertainty
# ============================================================

def test_external_input_uncertainty():

    omega, owner = make_omega()

    token = omega.new_token(
        owner,
        "measure",
    )

    env = StaticEnv(
        {
            "temperature": 22.0,
        }
    )

    external = Input(
        content={
            "temperature": 22.0,
        },
        conf=0.05,
        source="external_sensor",
    )

    result = omega.run(
        env,
        token,
        prediction=Prediction(
            expected={
                "temperature": 22.0,
            },
            conf=0.95,
        ),
        external=external,
    )

    assert_true(
        result.uncertainty["input"]
        > 0.90,
        "input uncertainty missing",
    )

    assert_equal(
        result.decision,
        Decision.HOLD,
    )


# ============================================================
# 12. Attribution exists on non-zero error
# ============================================================

def test_attribution():

    omega, owner = make_omega()

    result, _ = run_normal(
        omega,
        owner,
        observed=25.0,
        expected=22.0,
    )

    assert_true(
        result.attribution is not None,
        "attribution missing",
    )

    assert_true(
        result.attribution
        in {
            Hypothesis.SENSOR.value,
            Hypothesis.MODEL.value,
            Hypothesis.INPUT.value,
            Hypothesis.UNKNOWN.value,
        },
        "invalid attribution",
    )

    weights = (
        result.attribution_weights
    )

    assert_true(
        len(weights) > 0,
        "attribution weights missing",
    )


# ============================================================
# 13. Calibration is bounded
# ============================================================

def test_calibration_bound():

    omega, owner = make_omega()

    for _ in range(100):

        result, _ = run_normal(
            omega,
            owner,
            observed=22.5,
            expected=22.0,
        )

        # If the system enters another state,
        # the test still verifies the existing calibration.
        calibration = (
            omega.s.calibration
        )

        for value in calibration.values():

            assert_true(
                abs(value)
                <= omega.config.cal_abs_max
                + 1e-12,
                "calibration exceeded absolute bound",
            )

        for value in result.calibration_delta.values():

            assert_true(
                abs(value)
                <= omega.config.cal_step_max
                + 1e-12,
                "calibration step exceeded bound",
            )


# ============================================================
# 14. Boundary is immutable
# ============================================================

def test_boundary_immutable():

    omega, owner = make_omega()

    original = (
        omega.B.allowed
    )

    assert_true(
        "measure" in original,
    )

    assert_raises(
        AttributeError,
        lambda: setattr(
            omega.B,
            "_allowed",
            frozenset(
                {
                    "measure",
                    "dangerous_action",
                }
            ),
        ),
    )

    assert_equal(
        omega.B.allowed,
        original,
    )


# ============================================================
# 15. Unauthorized owner
# ============================================================

def test_owner_authorization():

    omega, owner = make_omega()

    fake = copy.deepcopy(
        owner
    )

    fake = type(owner)(
        id="fake-owner",
        digest=fake.digest,
        salt=fake.salt,
    )

    assert_raises(
        PermissionError,
        lambda: omega.set_goal(
            fake,
            {
                "temperature": 25.0,
            },
        ),
    )


# ============================================================
# 16. Audit chain valid
# ============================================================

def test_audit_valid():

    omega, owner = make_omega()

    run_normal(
        omega,
        owner,
        observed=22.0,
        expected=22.0,
    )

    assert_true(
        omega.verify_audit(),
        "audit should be valid",
    )


# ============================================================
# 17. Audit tampering detected
# ============================================================

def test_audit_tampering():

    omega, owner = make_omega()

    run_normal(
        omega,
        owner,
        observed=22.0,
        expected=22.0,
    )

    assert_true(
        omega.verify_audit(),
    )

    assert_true(
        len(omega.s.audit) > 0,
    )

    omega.s.audit[0]["p"][
        "tampered"
    ] = True

    assert_true(
        not omega.verify_audit(),
        "audit tampering was not detected",
    )


# ============================================================
# 18. Reflection
# ============================================================

def test_reflection():

    omega, owner = make_omega()

    run_normal(
        omega,
        owner,
        observed=22.0,
        expected=22.0,
    )

    reflection = omega.reflect()

    assert_equal(
        reflection["episodes"],
        1,
    )

    assert_true(
        "decisions"
        in reflection,
    )

    assert_true(
        "attributions"
        in reflection,
    )


# ============================================================
# 19. Replay
# ============================================================

def test_replay():

    omega, owner = make_omega()

    run_normal(
        omega,
        owner,
        observed=22.0,
        expected=22.0,
    )

    replay = omega.replay()

    assert_equal(
        len(replay),
        1,
    )

    episode = omega.replay(
        1
    )

    assert_equal(
        episode["epoch"],
        1,
    )


# ============================================================
# 20. Persistence
# ============================================================

def test_persistence():

    omega, owner = make_omega()

    run_normal(
        omega,
        owner,
        observed=22.0,
        expected=22.0,
    )

    before = omega.snapshot()

    with tempfile.TemporaryDirectory() as tmp:

        path = Path(
            tmp
        ) / "omega_state.json"

        omega.save_state(
            path
        )

        assert_true(
            path.exists(),
            "state file was not created",
        )

        restored, _ = make_omega()

        restored.load_state(
            path
        )

        after = restored.snapshot()

        assert_equal(
            after["epoch"],
            before["epoch"],
        )

        assert_equal(
            after["episodes"],
            before["episodes"],
        )

        assert_equal(
            after["calibration"],
            before["calibration"],
        )

        assert_true(
            restored.verify_audit(),
        )


# ============================================================
# 21. Loaded state does not restore tokens
# ============================================================

def test_tokens_not_persisted():

    omega, owner = make_omega()

    token = omega.new_token(
        owner,
        "measure",
    )

    with tempfile.TemporaryDirectory() as tmp:

        path = Path(
            tmp
        ) / "omega_state.json"

        omega.save_state(
            path
        )

        restored, _ = make_omega()

        restored.load_state(
            path
        )

        assert_true(
            token not in restored.s.tokens,
            "capability token was persisted",
        )


# ============================================================
# 22. Capability cannot exceed boundary
# ============================================================

def test_capability_boundary():

    omega, owner = make_omega()

    assert_raises(
        PermissionError,
        lambda: omega.new_token(
            owner,
            "delete_everything",
        ),
    )

    assert_raises(
        PermissionError,
        lambda: omega.grant(
            owner,
            "delete_everything",
            "abcdefghijklmnop",
        ),
    )


# ============================================================
# 23. Invalid limits rejected
# ============================================================

def test_invalid_limits():

    omega, owner = make_omega()

    assert_raises(
        ValueError,
        lambda: omega.set_limits(
            owner,
            hard_min={
                "temperature": 30.0,
            },
            hard_max={
                "temperature": 10.0,
            },
        ),
    )


# ============================================================
# 24. Invalid numeric input rejected
# ============================================================

def test_invalid_numeric():

    omega, owner = make_omega()

    assert_raises(
        ValueError,
        lambda: omega.set_goal(
            owner,
            {
                "temperature": float(
                    "nan"
                ),
            },
        ),
    )


# ============================================================
# 25. Status
# ============================================================

def test_status():

    omega, owner = make_omega()

    status = omega.status()

    assert_equal(
        status["epoch"],
        0,
    )

    assert_equal(
        status["view"],
        View.SHAN.value,
    )

    assert_equal(
        status["episodes"],
        0,
    )


# ============================================================
# Test registry
# ============================================================

TESTS = [
    (
        "basic APPLY",
        test_basic_apply,
    ),
    (
        "single-use token",
        test_single_use_token,
    ),
    (
        "invalid token",
        test_invalid_token,
    ),
    (
        "dry-run state safety",
        test_dry_run,
    ),
    (
        "dry-run then real run",
        test_dry_run_then_real_run,
    ),
    (
        "hard limit violation",
        test_hard_limit,
    ),
    (
        "escalation blocks follow-up",
        test_escalation_blocks_followup,
    ),
    (
        "pause/resume",
        test_pause_resume,
    ),
    (
        "prediction error HOLD",
        test_prediction_error_hold,
    ),
    (
        "high uncertainty",
        test_high_uncertainty,
    ),
    (
        "external input uncertainty",
        test_external_input_uncertainty,
    ),
    (
        "attribution",
        test_attribution,
    ),
    (
        "bounded calibration",
        test_calibration_bound,
    ),
    (
        "immutable boundary",
        test_boundary_immutable,
    ),
    (
        "owner authorization",
        test_owner_authorization,
    ),
    (
        "audit valid",
        test_audit_valid,
    ),
    (
        "audit tampering detection",
        test_audit_tampering,
    ),
    (
        "reflection",
        test_reflection,
    ),
    (
        "replay",
        test_replay,
    ),
    (
        "persistence",
        test_persistence,
    ),
    (
        "tokens not persisted",
        test_tokens_not_persisted,
    ),
    (
        "capability boundary",
        test_capability_boundary,
    ),
    (
        "invalid limits",
        test_invalid_limits,
    ),
    (
        "invalid numeric input",
        test_invalid_numeric,
    ),
    (
        "status",
        test_status,
    ),
]


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 72)
    print("Ω v3.3 STANDARD-LIBRARY TEST SUITE")
    print("=" * 72)

    for name, function in TESTS:
        test(
            name,
            function,
        )

    print("-" * 72)

    total = (
        PASSED
        + FAILED
    )

    print(
        f"TOTAL : {total}"
    )
    print(
        f"PASS  : {PASSED}"
    )
    print(
        f"FAIL  : {FAILED}"
    )

    print("=" * 72)

    if FAILED:
        print(
            "RESULT: FAIL"
        )
        return 1

    print(
        "RESULT: ALL TESTS PASSED"
    )

    return 0

if __name__ == "__main__":
    code = main()

    if sys.stdin.isatty():
        input("\n按回车键退出...")

    sys.exit(code)