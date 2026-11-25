"""
Omega v3.3
==========

一个带有以下机制的受约束决策/学习框架：

- Owner authorization
- Immutable boundary
- Single-use capability token
- Prediction -> Observation -> Difference
- Uncertainty
- Attribution
- Bounded calibration
- HOLD / APPLY / ESCALATE
- Pause / Resume
- SHAN / NOT_SHAN / SHAN_AGAIN
- Audit hash chain
- Persistence
- Replay
- Reflection
- Dry-run

本文件设计为单文件核心实现。

Python:
    3.10+

运行测试：
    pytest -q

运行实验：
    python experiment.py --rounds 20
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import math
import secrets
import time
from dataclasses import (
    asdict,
    dataclass,
    field,
)
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
)


# ============================================================
# Constants
# ============================================================

EPSILON = 1e-12


# ============================================================
# Enums
# ============================================================

class Decision(str, Enum):
    APPLY = "APPLY"
    HOLD = "HOLD"
    ESCALATE = "ESCALATE"


class Hypothesis(str, Enum):
    SENSOR = "sensor"
    MODEL = "model"
    INPUT = "input"
    UNKNOWN = "unknown"


class View(str, Enum):
    SHAN = "SHAN"
    NOT_SHAN = "NOT_SHAN"
    SHAN_AGAIN = "SHAN_AGAIN"


# ============================================================
# Data classes
# ============================================================

@dataclass(frozen=True)
class OwnerHandle:
    id: str
    digest: str
    salt: str


@dataclass
class OmegaConfig:
    # Calibration.
    cal_step_max: float = 0.10
    cal_abs_max: float = 2.00

    # View state machine.
    bad_rounds_to_leave_shan: int = 3
    good_rounds_to_shan_again: int = 3

    # Decision thresholds.
    max_apply_error: float = 1.0
    max_apply_uncertainty: float = 0.75
    max_hold_uncertainty: float = 0.90

    # Attribution.
    attribution_sensor_weight: float = 0.55
    attribution_model_weight: float = 0.35
    attribution_input_weight: float = 0.10

    # Persistence.
    schema_version: int = 1


@dataclass
class Prediction:
    expected: Dict[str, float]
    conf: float = 1.0

    def __post_init__(self):
        self.conf = clamp(
            float(self.conf),
            0.0,
            1.0,
        )


@dataclass
class Input:
    content: Dict[str, float]
    conf: float = 1.0
    source: str = "external"

    def __post_init__(self):
        self.conf = clamp(
            float(self.conf),
            0.0,
            1.0,
        )


@dataclass
class Episode:
    epoch: int
    timestamp: float
    action: str

    prediction: Dict[str, float]
    observation: Dict[str, float]
    difference: Dict[str, float]

    error: float
    confidence: float

    uncertainty: Dict[str, float]
    attribution: Optional[str]
    attribution_weights: Dict[str, float]

    decision: str
    reason: str

    calibration_delta: Dict[str, float]

    view_before: str
    view_after: str

    goal_distance: float

    dry_run: bool = False


@dataclass
class OmegaState:
    epoch: int = 0

    goal: Dict[str, float] = field(
        default_factory=dict
    )

    hard_min: Dict[str, float] = field(
        default_factory=dict
    )

    hard_max: Dict[str, float] = field(
        default_factory=dict
    )

    calibration: Dict[str, float] = field(
        default_factory=dict
    )

    episodes: List[Dict[str, Any]] = field(
        default_factory=list
    )

    audit: List[Dict[str, Any]] = field(
        default_factory=list
    )

    tokens: Dict[str, str] = field(
        default_factory=dict
    )

    used: List[str] = field(
        default_factory=list
    )

    paused: bool = False
    escalated: bool = False

    view: View = View.SHAN

    bad_rounds: int = 0
    good_rounds: int = 0


# ============================================================
# Utility functions
# ============================================================

def clamp(
    value: float,
    low: float,
    high: float,
) -> float:

    return max(
        low,
        min(
            high,
            value,
        ),
    )


def finite_number(
    value: Any,
) -> bool:

    return (
        isinstance(
            value,
            (int, float),
        )
        and math.isfinite(
            float(value)
        )
    )


def normalize_vector(
    data: Mapping[str, Any],
) -> Dict[str, float]:

    result = {}

    for key, value in data.items():

        if not isinstance(
            key,
            str,
        ):
            raise TypeError(
                "vector keys must be strings"
            )

        if not finite_number(value):
            raise ValueError(
                f"invalid numeric value for {key}"
            )

        result[key] = float(value)

    return result


def vector_difference(
    expected: Mapping[str, float],
    observed: Mapping[str, float],
) -> Dict[str, float]:

    keys = (
        set(expected)
        | set(observed)
    )

    result = {}

    for key in keys:

        e = float(
            expected.get(
                key,
                0.0,
            )
        )

        o = float(
            observed.get(
                key,
                0.0,
            )
        )

        result[key] = o - e

    return result


def vector_error(
    difference: Mapping[str, float],
) -> float:

    if not difference:
        return 0.0

    squared = sum(
        value * value
        for value in difference.values()
    )

    return math.sqrt(
        squared
        / len(difference)
    )


def canonical_json(
    value: Any,
) -> str:

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_text(
    value: str,
) -> str:

    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


# ============================================================
# Environment
# ============================================================

class StaticEnv:
    """
    Deterministic test environment.

    execute(action) always returns the configured
    observation and records the number of calls.
    """

    def __init__(
        self,
        observation: Mapping[str, float],
    ):
        self.observation = normalize_vector(
            observation
        )

        self.calls = 0

    def execute(
        self,
        action: str,
    ) -> Dict[str, float]:

        self.calls += 1

        return copy.deepcopy(
            self.observation
        )


# ============================================================
# Immutable boundary
# ============================================================

class Boundary:
    """
    Immutable capability boundary.

    Learning may change calibration/state,
    but must not add new capabilities.
    """

    __slots__ = (
        "_allowed",
        "_frozen",
    )

    def __init__(
        self,
        allowed: Iterable[str],
    ):

        self._allowed = frozenset(
            allowed
        )

        self._frozen = True

    @property
    def allowed(self):
        return self._allowed

    def permits(
        self,
        action: str,
    ) -> bool:

        return action in self._allowed

    def __setattr__(
        self,
        name,
        value,
    ):

        if getattr(
            self,
            "_frozen",
            False,
        ):
            raise AttributeError(
                "Boundary is immutable"
            )

        object.__setattr__(
            self,
            name,
            value,
        )


# ============================================================
# Omega
# ============================================================

class Omega:

    def __init__(
        self,
        secret: str,
        config: Optional[OmegaConfig] = None,
    ):

        if not isinstance(
            secret,
            str,
        ):
            raise TypeError(
                "secret must be a string"
            )

        if len(secret) < 8:
            raise ValueError(
                "secret must contain at least 8 characters"
            )

        self.config = (
            config
            or OmegaConfig()
        )

        self._secret = secret

        self._owner = self._create_owner()

        # Boundary is created once and never replaced.
        self.B = Boundary(
            allowed={
                "measure",
            }
        )

        self.s = OmegaState()

    # ========================================================
    # Owner
    # ========================================================

    def _create_owner(
        self,
    ) -> OwnerHandle:

        salt = secrets.token_hex(
            16
        )

        owner_id = secrets.token_hex(
            16
        )

        digest = self._owner_digest(
            owner_id,
            salt,
        )

        return OwnerHandle(
            id=owner_id,
            digest=digest,
            salt=salt,
        )

    def _owner_digest(
        self,
        owner_id: str,
        salt: str,
    ) -> str:

        message = (
            owner_id
            + ":"
            + salt
        ).encode(
            "utf-8"
        )

        return hmac.new(
            self._secret.encode(
                "utf-8"
            ),
            message,
            hashlib.sha256,
        ).hexdigest()

    def owner_handle(
        self,
    ) -> OwnerHandle:

        return copy.deepcopy(
            self._owner
        )

    def _require_owner(
        self,
        owner: OwnerHandle,
    ):

        if not isinstance(
            owner,
            OwnerHandle,
        ):
            raise PermissionError(
                "owner authorization required"
            )

        expected = self._owner_digest(
            owner.id,
            owner.salt,
        )

        if not hmac.compare_digest(
            expected,
            owner.digest,
        ):
            raise PermissionError(
                "invalid owner authorization"
            )

        if not hmac.compare_digest(
            owner.id,
            self._owner.id,
        ):
            raise PermissionError(
                "wrong owner"
            )

    # ========================================================
    # Boundary
    # ========================================================

    def grant(
        self,
        owner: OwnerHandle,
        action: str,
        token: str,
    ):

        self._require_owner(
            owner
        )

        if not self.B.permits(
            action
        ):
            raise PermissionError(
                "action is outside immutable boundary"
            )

        if not isinstance(
            token,
            str,
        ):
            raise TypeError(
                "token must be a string"
            )

        if len(token) < 16:
            raise ValueError(
                "token must contain at least 16 characters"
            )

        self.s.tokens[token] = action

        self._audit(
            "grant",
            {
                "action": action,
                "token_hash": sha256_text(
                    token
                ),
            },
        )

    def new_token(
        self,
        owner: OwnerHandle,
        action: str = "measure",
    ) -> str:

        self._require_owner(
            owner
        )

        if not self.B.permits(
            action
        ):
            raise PermissionError(
                "action outside boundary"
            )

        token = secrets.token_urlsafe(
            24
        )

        self.s.tokens[token] = action

        self._audit(
            "token_created",
            {
                "action": action,
                "token_hash": sha256_text(
                    token
                ),
            },
        )

        return token

    def _consume_token(
        self,
        token: str,
    ) -> Optional[str]:

        if not isinstance(
            token,
            str,
        ):
            return None

        if token in self.s.used:
            return None

        action = self.s.tokens.pop(
            token,
            None,
        )

        if action is None:
            return None

        self.s.used.append(
            token
        )

        return action

    # ========================================================
    # Goal and limits
    # ========================================================

    def set_goal(
        self,
        owner: OwnerHandle,
        goal: Mapping[str, float],
    ):

        self._require_owner(
            owner
        )

        self.s.goal = normalize_vector(
            goal
        )

        self._audit(
            "set_goal",
            {
                "goal": self.s.goal,
            },
        )

    def set_limits(
        self,
        owner: OwnerHandle,
        hard_min: Optional[
            Mapping[str, float]
        ] = None,
        hard_max: Optional[
            Mapping[str, float]
        ] = None,
    ):

        self._require_owner(
            owner
        )

        if hard_min is not None:
            self.s.hard_min = normalize_vector(
                hard_min
            )

        if hard_max is not None:
            self.s.hard_max = normalize_vector(
                hard_max
            )

        for key in (
            set(self.s.hard_min)
            | set(self.s.hard_max)
        ):

            if (
                key in self.s.hard_min
                and key in self.s.hard_max
                and self.s.hard_min[key]
                > self.s.hard_max[key]
            ):
                raise ValueError(
                    f"invalid hard limit for {key}"
                )

        self._audit(
            "set_limits",
            {
                "hard_min": self.s.hard_min,
                "hard_max": self.s.hard_max,
            },
        )

    # ========================================================
    # Pause / resume / escalation
    # ========================================================

    def pause(
        self,
        owner: OwnerHandle,
    ):

        self._require_owner(
            owner
        )

        self.s.paused = True

        self._audit(
            "pause",
            {},
        )

    def resume(
        self,
        owner: OwnerHandle,
    ):

        self._require_owner(
            owner
        )

        self.s.paused = False

        self._audit(
            "resume",
            {},
        )

    def deescalate(
        self,
        owner: OwnerHandle,
    ):

        self._require_owner(
            owner
        )

        self.s.escalated = False

        self._audit(
            "deescalate",
            {},
        )

    # ========================================================
    # Uncertainty
    # ========================================================

    def _calculate_uncertainty(
        self,
        prediction: Prediction,
        external: Optional[Input],
    ) -> Dict[str, float]:

        prediction_uncertainty = (
            1.0
            - prediction.conf
        )

        input_uncertainty = 0.0

        if external is not None:
            input_uncertainty = (
                1.0
                - external.conf
            )

        total = clamp(
            max(
                prediction_uncertainty,
                input_uncertainty,
            ),
            0.0,
            1.0,
        )

        return {
            "prediction": prediction_uncertainty,
            "input": input_uncertainty,
            "total": total,
        }

    # ========================================================
    # Attribution
    # ========================================================

    def _attribute(
        self,
        difference: Mapping[str, float],
        uncertainty: Mapping[str, float],
        external: Optional[Input],
    ) -> Tuple[
        Optional[str],
        Dict[str, float],
    ]:

        error = vector_error(
            difference
        )

        if error <= EPSILON:
            return (
                None,
                {
                    Hypothesis.SENSOR.value: 0.0,
                    Hypothesis.MODEL.value: 0.0,
                    Hypothesis.INPUT.value: 0.0,
                },
            )

        sensor = (
            self.config.attribution_sensor_weight
        )

        model = (
            self.config.attribution_model_weight
        )

        input_weight = (
            self.config.attribution_input_weight
        )

        # Low-confidence external input increases
        # the input hypothesis.
        if external is not None:
            input_weight += (
                1.0
                - external.conf
            )

        # Large error with high prediction confidence
        # increases model attribution.
        model += (
            0.25
            * self._prediction_confidence_factor(
                uncertainty
            )
        )

        # If the world is highly stable but current
        # observation is abnormal, sensor attribution rises.
        sensor += (
            0.25
            * min(
                1.0,
                error / 10.0,
            )
        )

        weights = {
            Hypothesis.SENSOR.value: max(
                0.0,
                sensor,
            ),
            Hypothesis.MODEL.value: max(
                0.0,
                model,
            ),
            Hypothesis.INPUT.value: max(
                0.0,
                input_weight,
            ),
        }

        total = sum(
            weights.values()
        )

        if total <= EPSILON:
            return (
                Hypothesis.UNKNOWN.value,
                {
                    **weights,
                },
            )

        weights = {
            key: value / total
            for key, value in weights.items()
        }

        attribution = max(
            weights,
            key=weights.get,
        )

        return (
            attribution,
            weights,
        )

    def _prediction_confidence_factor(
        self,
        uncertainty: Mapping[str, float],
    ) -> float:

        prediction_uncertainty = uncertainty.get(
            "prediction",
            0.0,
        )

        return clamp(
            1.0
            - prediction_uncertainty,
            0.0,
            1.0,
        )

    # ========================================================
    # Hard limits
    # ========================================================

    def _violates_limits(
        self,
        observation: Mapping[str, float],
    ) -> bool:

        for key, value in observation.items():

            if (
                key in self.s.hard_min
                and value < self.s.hard_min[key]
            ):
                return True

            if (
                key in self.s.hard_max
                and value > self.s.hard_max[key]
            ):
                return True

        return False

    # ========================================================
    # Goal distance
    # ========================================================

    def _goal_distance(
        self,
        observation: Mapping[str, float],
    ) -> float:

        if not self.s.goal:
            return 0.0

        difference = vector_difference(
            self.s.goal,
            observation,
        )

        return vector_error(
            difference
        )

    # ========================================================
    # Calibration
    # ========================================================

    def _calibrate(
        self,
        difference: Mapping[str, float],
    ) -> Dict[str, float]:

        delta = {}

        for key, error in difference.items():

            raw = (
                error
                * 0.10
            )

            step = clamp(
                raw,
                -self.config.cal_step_max,
                self.config.cal_step_max,
            )

            current = self.s.calibration.get(
                key,
                0.0,
            )

            updated = clamp(
                current + step,
                -self.config.cal_abs_max,
                self.config.cal_abs_max,
            )

            actual_step = (
                updated
                - current
            )

            if abs(actual_step) > EPSILON:
                delta[key] = actual_step

                self.s.calibration[key] = updated

        return delta

    # ========================================================
    # View state machine
    # ========================================================

    def _update_view(
        self,
        *,
        good: bool,
    ) -> Tuple[
        View,
        View,
    ]:

        before = self.s.view

        if good:

            self.s.good_rounds += 1
            self.s.bad_rounds = 0

            if (
                self.s.view
                is View.NOT_SHAN
                and self.s.good_rounds
                >= self.config.good_rounds_to_shan_again
            ):
                self.s.view = View.SHAN_AGAIN

            elif (
                self.s.view
                is View.SHAN
            ):
                pass

        else:

            self.s.bad_rounds += 1
            self.s.good_rounds = 0

            if (
                self.s.bad_rounds
                >= self.config.bad_rounds_to_leave_shan
            ):
                self.s.view = View.NOT_SHAN

        return (
            before,
            self.s.view,
        )

    # ========================================================
    # Decision
    # ========================================================

    def _decide(
        self,
        *,
        error: float,
        uncertainty: Mapping[str, float],
        observation: Mapping[str, float],
        attribution: Optional[str],
    ) -> Tuple[
        Decision,
        str,
    ]:

        if self.s.paused:
            return (
                Decision.ESCALATE,
                "paused",
            )

        if self.s.escalated:
            return (
                Decision.ESCALATE,
                "already_escalated",
            )

        if self._violates_limits(
            observation
        ):
            self.s.escalated = True

            return (
                Decision.ESCALATE,
                "hard_limit_violation",
            )

        total_uncertainty = uncertainty.get(
            "total",
            1.0,
        )

        if (
            total_uncertainty
            >= self.config.max_hold_uncertainty
        ):
            return (
                Decision.HOLD,
                "high_uncertainty",
            )

        if (
            error
            > self.config.max_apply_error
        ):
            return (
                Decision.HOLD,
                "prediction_error_too_large",
            )

        if (
            total_uncertainty
            > self.config.max_apply_uncertainty
        ):
            return (
                Decision.HOLD,
                "uncertainty_above_apply_threshold",
            )

        return (
            Decision.APPLY,
            "within_threshold",
        )

    # ========================================================
    # Main run
    # ========================================================

    def run(
        self,
        env: StaticEnv,
        token: str,
        prediction: Optional[Prediction] = None,
        external: Optional[Input] = None,
        dry_run: bool = False,
    ) -> Any:

        # ----------------------------------------------------
        # Validate token WITHOUT consuming it for dry-run.
        # ----------------------------------------------------

        if not isinstance(
            token,
            str,
        ):
            return self._result(
                decision=Decision.ESCALATE,
                reason="invalid_token",
                observation={},
                prediction={},
                difference={},
                error=0.0,
                uncertainty={},
                attribution=None,
                attribution_weights={},
                calibration_delta={},
                view_before=self.s.view,
                view_after=self.s.view,
                goal_distance=0.0,
                dry_run=dry_run,
            )

        if token in self.s.used:
            return self._result(
                decision=Decision.ESCALATE,
                reason="token_already_used",
                observation={},
                prediction={},
                difference={},
                error=0.0,
                uncertainty={},
                attribution=None,
                attribution_weights={},
                calibration_delta={},
                view_before=self.s.view,
                view_after=self.s.view,
                goal_distance=0.0,
                dry_run=dry_run,
            )

        action = self.s.tokens.get(
            token
        )

        if action is None:
            return self._result(
                decision=Decision.ESCALATE,
                reason="invalid_token",
                observation={},
                prediction={},
                difference={},
                error=0.0,
                uncertainty={},
                attribution=None,
                attribution_weights={},
                calibration_delta={},
                view_before=self.s.view,
                view_after=self.s.view,
                goal_distance=0.0,
                dry_run=dry_run,
            )

        if not self.B.permits(
            action
        ):
            return self._result(
                decision=Decision.ESCALATE,
                reason="action_outside_boundary",
                observation={},
                prediction={},
                difference={},
                error=0.0,
                uncertainty={},
                attribution=None,
                attribution_weights={},
                calibration_delta={},
                view_before=self.s.view,
                view_after=self.s.view,
                goal_distance=0.0,
                dry_run=dry_run,
            )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        if prediction is None:
            prediction = Prediction(
                expected={},
                conf=0.0,
            )

        expected = normalize_vector(
            prediction.expected
        )

        uncertainty = (
            self._calculate_uncertainty(
                prediction,
                external,
            )
        )

        # ----------------------------------------------------
        # Dry-run
        # ----------------------------------------------------

        if dry_run:

            # Do NOT consume token.
            # Do NOT execute environment.
            # Do NOT change state.

            return self._result(
                decision=Decision.HOLD,
                reason="dry_run",
                observation={},
                prediction=expected,
                difference={},
                error=0.0,
                uncertainty=uncertainty,
                attribution=None,
                attribution_weights={},
                calibration_delta={},
                view_before=self.s.view,
                view_after=self.s.view,
                goal_distance=0.0,
                dry_run=True,
            )
        # ----------------------------------------------------
        # Pre-execution safety gate.
        #
        # A paused or already-escalated system must not
        # execute the environment at all.
        # ----------------------------------------------------

        if self.s.paused:
            return self._result(
                decision=Decision.ESCALATE,
                reason="paused",
                observation={},
                prediction=expected,
                difference={},
                error=0.0,
                uncertainty=uncertainty,
                attribution=None,
                attribution_weights={},
                calibration_delta={},
                view_before=self.s.view,
                view_after=self.s.view,
                goal_distance=0.0,
                dry_run=False,
            )

        if self.s.escalated:
            return self._result(
                decision=Decision.ESCALATE,
                reason="already_escalated",
                observation={},
                prediction=expected,
                difference={},
                error=0.0,
                uncertainty=uncertainty,
                attribution=None,
                attribution_weights={},
                calibration_delta={},
                view_before=self.s.view,
                view_after=self.s.view,
                goal_distance=0.0,
                dry_run=False,
            )

        # ----------------------------------------------------
        # Consume token immediately before execution.
        # ----------------------------------------------------

        action = self._consume_token(
            token
        )


        if action is None:
            return self._result(
                decision=Decision.ESCALATE,
                reason="token_consumption_failed",
                observation={},
                prediction=expected,
                difference={},
                error=0.0,
                uncertainty=uncertainty,
                attribution=None,
                attribution_weights={},
                calibration_delta={},
                view_before=self.s.view,
                view_after=self.s.view,
                goal_distance=0.0,
                dry_run=False,
            )

        # ----------------------------------------------------
        # Execute environment.
        # ----------------------------------------------------

        observation = normalize_vector(
            env.execute(
                action
            )
        )

        difference = vector_difference(
            expected,
            observation,
        )

        error = vector_error(
            difference
        )

        (
            attribution,
            attribution_weights,
        ) = self._attribute(
            difference,
            uncertainty,
            external,
        )

        (
            decision,
            reason,
        ) = self._decide(
            error=error,
            uncertainty=uncertainty,
            observation=observation,
            attribution=attribution,
        )

        # ----------------------------------------------------
        # Learning only on APPLY.
        # ----------------------------------------------------

        calibration_delta = {}

        if decision is Decision.APPLY:

            calibration_delta = (
                self._calibrate(
                    difference
                )
            )

        # ----------------------------------------------------
        # View update.
        # ----------------------------------------------------

        good = (
            decision is Decision.APPLY
            and error
            <= self.config.max_apply_error
        )

        (
            view_before,
            view_after,
        ) = self._update_view(
            good=good
        )

        # ----------------------------------------------------
        # Epoch.
        # ----------------------------------------------------

        self.s.epoch += 1

        goal_distance = (
            self._goal_distance(
                observation
            )
        )

        episode = Episode(
            epoch=self.s.epoch,
            timestamp=time.time(),
            action=action,
            prediction=expected,
            observation=observation,
            difference=difference,
            error=error,
            confidence=prediction.conf,
            uncertainty=uncertainty,
            attribution=attribution,
            attribution_weights=attribution_weights,
            decision=decision.value,
            reason=reason,
            calibration_delta=calibration_delta,
            view_before=view_before.value,
            view_after=view_after.value,
            goal_distance=goal_distance,
            dry_run=False,
        )

        episode_dict = asdict(
            episode
        )

        self.s.episodes.append(
            episode_dict
        )

        self._audit(
            "episode",
            episode_dict,
        )

        return self._result(
            decision=decision,
            reason=reason,
            observation=observation,
            prediction=expected,
            difference=difference,
            error=error,
            uncertainty=uncertainty,
            attribution=attribution,
            attribution_weights=attribution_weights,
            calibration_delta=calibration_delta,
            view_before=view_before,
            view_after=view_after,
            goal_distance=goal_distance,
            dry_run=False,
        )

    # ========================================================
    # Result
    # ========================================================

    def _result(
        self,
        *,
        decision: Decision,
        reason: str,
        observation: Dict[str, float],
        prediction: Dict[str, float],
        difference: Dict[str, float],
        error: float,
        uncertainty: Dict[str, float],
        attribution: Optional[str],
        attribution_weights: Dict[str, float],
        calibration_delta: Dict[str, float],
        view_before: View,
        view_after: View,
        goal_distance: float,
        dry_run: bool,
    ):

        class Result:
            pass

        result = Result()

        result.decision = decision
        result.reason = reason

        result.observation = copy.deepcopy(
            observation
        )

        result.prediction = copy.deepcopy(
            prediction
        )

        result.difference = copy.deepcopy(
            difference
        )

        result.error = float(
            error
        )

        result.uncertainty = copy.deepcopy(
            uncertainty
        )

        result.attribution = attribution

        result.attribution_weights = (
            copy.deepcopy(
                attribution_weights
            )
        )

        result.calibration_delta = (
            copy.deepcopy(
                calibration_delta
            )
        )

        result.view_before = view_before
        result.view_after = view_after

        result.meta = {
            "goal_distance": goal_distance,
            "dry_run": dry_run,
        }

        return result

    # ========================================================
    # Audit
    # ========================================================

    def _audit(
        self,
        event: str,
        payload: Mapping[str, Any],
    ):

        previous_hash = (
            self.s.audit[-1]["hash"]
            if self.s.audit
            else "GENESIS"
        )

        record = {
            "index": len(
                self.s.audit
            ),
            "timestamp": time.time(),
            "event": event,
            "p": copy.deepcopy(
                dict(payload)
            ),
            "prev": previous_hash,
        }

        body = canonical_json(
            record
        )

        record["hash"] = sha256_text(
            body
        )

        self.s.audit.append(
            record
        )

    def verify_audit(
        self,
    ) -> bool:

        previous_hash = "GENESIS"

        for index, record in enumerate(
            self.s.audit
        ):

            if record.get(
                "index"
            ) != index:
                return False

            if record.get(
                "prev"
            ) != previous_hash:
                return False

            stored_hash = record.get(
                "hash"
            )

            if not stored_hash:
                return False

            unsigned = {
                key: value
                for key, value in record.items()
                if key != "hash"
            }

            expected_hash = sha256_text(
                canonical_json(
                    unsigned
                )
            )

            if not hmac.compare_digest(
                expected_hash,
                stored_hash,
            ):
                return False

            previous_hash = stored_hash

        return True

    # ========================================================
    # Snapshot
    # ========================================================

    def snapshot(
        self,
    ) -> Dict[str, Any]:

        return {
            "schema_version": (
                self.config.schema_version
            ),
            "epoch": self.s.epoch,
            "goal": copy.deepcopy(
                self.s.goal
            ),
            "hard_min": copy.deepcopy(
                self.s.hard_min
            ),
            "hard_max": copy.deepcopy(
                self.s.hard_max
            ),
            "calibration": copy.deepcopy(
                self.s.calibration
            ),
            "episodes": copy.deepcopy(
                self.s.episodes
            ),
            "audit": copy.deepcopy(
                self.s.audit
            ),
            "paused": self.s.paused,
            "escalated": self.s.escalated,
            "view": self.s.view.value,
            "bad_rounds": self.s.bad_rounds,
            "good_rounds": self.s.good_rounds,
        }

    # ========================================================
    # Status
    # ========================================================

    def status(
        self,
    ) -> Dict[str, Any]:

        snapshot = self.snapshot()

        return {
            "epoch": snapshot["epoch"],
            "goal": snapshot["goal"],
            "limits": {
                "hard_min": snapshot[
                    "hard_min"
                ],
                "hard_max": snapshot[
                    "hard_max"
                ],
            },
            "calibration": snapshot[
                "calibration"
            ],
            "episodes": len(
                snapshot["episodes"]
            ),
            "audit_records": len(
                snapshot["audit"]
            ),
            "paused": snapshot[
                "paused"
            ],
            "escalated": snapshot[
                "escalated"
            ],
            "view": snapshot[
                "view"
            ],
        }

    # ========================================================
    # Reflection
    # ========================================================

    def reflect(
        self,
    ) -> Dict[str, Any]:

        decisions = {}

        errors = []

        attributions = {}

        for episode in self.s.episodes:

            decision = episode[
                "decision"
            ]

            decisions[decision] = (
                decisions.get(
                    decision,
                    0,
                )
                + 1
            )

            errors.append(
                episode["error"]
            )

            attribution = episode.get(
                "attribution"
            )

            if attribution:
                attributions[
                    attribution
                ] = (
                    attributions.get(
                        attribution,
                        0,
                    )
                    + 1
                )

        mean_error = (
            sum(errors)
            / len(errors)
            if errors
            else 0.0
        )

        return {
            "epoch": self.s.epoch,
            "episodes": len(
                self.s.episodes
            ),
            "decisions": decisions,
            "attributions": attributions,
            "mean_error": mean_error,
            "view": self.s.view.value,
            "escalated": self.s.escalated,
            "paused": self.s.paused,
        }

    # ========================================================
    # Replay
    # ========================================================

    def replay(
        self,
        epoch: Optional[int] = None,
    ) -> Any:

        episodes = copy.deepcopy(
            self.s.episodes
        )

        if epoch is None:
            return episodes

        for episode in episodes:

            if episode["epoch"] == epoch:
                return episode

        raise KeyError(
            f"epoch {epoch} not found"
        )

    # ========================================================
    # Persistence
    # ========================================================

    def save_state(
        self,
        path: str | Path,
    ):

        path = Path(
            path
        )

        payload = {
            "schema_version": (
                self.config.schema_version
            ),
            "owner": {
                "id": self._owner.id,
                "digest": self._owner.digest,
                "salt": self._owner.salt,
            },
            "state": self.snapshot(),
        }

        path.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def load_state(
        self,
        path: str | Path,
    ):

        path = Path(
            path
        )

        payload = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        if (
            payload.get(
                "schema_version"
            )
            != self.config.schema_version
        ):
            raise ValueError(
                "schema version mismatch"
            )

        state = payload.get(
            "state"
        )

        if not isinstance(
            state,
            dict,
        ):
            raise ValueError(
                "invalid state"
            )

        self.s.epoch = int(
            state.get(
                "epoch",
                0,
            )
        )

        self.s.goal = normalize_vector(
            state.get(
                "goal",
                {},
            )
        )

        self.s.hard_min = normalize_vector(
            state.get(
                "hard_min",
                {},
            )
        )

        self.s.hard_max = normalize_vector(
            state.get(
                "hard_max",
                {},
            )
        )

        self.s.calibration = normalize_vector(
            state.get(
                "calibration",
                {},
            )
        )

        self.s.episodes = copy.deepcopy(
            state.get(
                "episodes",
                [],
            )
        )

        self.s.audit = copy.deepcopy(
            state.get(
                "audit",
                [],
            )
        )

        self.s.paused = bool(
            state.get(
                "paused",
                False,
            )
        )

        self.s.escalated = bool(
            state.get(
                "escalated",
                False,
            )
        )

        view = state.get(
            "view",
            View.SHAN.value,
        )

        self.s.view = View(
            view
        )

        self.s.bad_rounds = int(
            state.get(
                "bad_rounds",
                0,
            )
        )

        self.s.good_rounds = int(
            state.get(
                "good_rounds",
                0,
            )
        )

        # Capability tokens are intentionally NOT restored.
        self.s.tokens = {}
        self.s.used = []

        if not self.verify_audit():
            raise ValueError(
                "audit verification failed"
            )


# ============================================================
# Minimal self-test
# ============================================================

def _self_test():

    omega = Omega(
        secret="self-test-secret"
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

    token = omega.new_token(
        owner,
        "measure",
    )

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
    )

    assert result.decision is Decision.APPLY
    assert env.calls == 1
    assert omega.verify_audit()

    print(
        "Omega self-test: PASS"
    )


if __name__ == "__main__":
    _self_test()