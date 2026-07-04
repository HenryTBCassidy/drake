"""Post-hoc probability calibration (docs/03-EVALUATION-PLAN.md § Post-Hoc Calibration)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.linear_model import LogisticRegression

if TYPE_CHECKING:
    from numpy.typing import NDArray

_EPSILON = 1e-7


class PlattCalibrator:
    """Platt scaling: p_calibrated = sigmoid(a * logit(p_raw) + b).

    Fit on the held-out calibration split, never on train or test.
    """

    def __init__(self) -> None:
        self._regression: LogisticRegression | None = None

    def fit(self, labels: NDArray[np.int_], probabilities: NDArray[np.float64]) -> None:
        self._regression = LogisticRegression(C=1e6)  # effectively unregularised 2-parameter fit
        self._regression.fit(_logit(probabilities).reshape(-1, 1), labels)

    def apply(self, probabilities: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._regression is None:
            raise RuntimeError("PlattCalibrator is not fitted")
        calibrated = self._regression.predict_proba(_logit(probabilities).reshape(-1, 1))[:, 1]
        return np.asarray(calibrated, dtype=np.float64)


def _logit(probabilities: NDArray[np.float64]) -> NDArray[np.float64]:
    clipped = np.clip(probabilities, _EPSILON, 1 - _EPSILON)
    result: NDArray[np.float64] = np.log(clipped / (1 - clipped))
    return result
