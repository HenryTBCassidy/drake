"""Tests for drake.evaluation.calibration."""

from __future__ import annotations

import numpy as np
import pytest

from drake.evaluation.calibration import PlattCalibrator
from drake.evaluation.metrics import expected_calibration_error


def test_platt_fixes_systematic_overconfidence() -> None:
    rng = np.random.default_rng(3)
    true_probabilities = rng.uniform(0.2, 0.8, size=4000)
    labels = (rng.random(4000) < true_probabilities).astype(int)
    # Overconfident model: pushes probabilities toward the extremes.
    logits = np.log(true_probabilities / (1 - true_probabilities))
    overconfident = 1 / (1 + np.exp(-3.0 * logits))

    calibrator = PlattCalibrator()
    calibrator.fit(labels[:2000], overconfident[:2000])
    calibrated = calibrator.apply(overconfident[2000:])

    raw_ece = expected_calibration_error(labels[2000:], overconfident[2000:])
    calibrated_ece = expected_calibration_error(labels[2000:], calibrated)
    assert calibrated_ece < raw_ece / 2, f"Platt must shrink ECE ({raw_ece:.3f} -> {calibrated_ece:.3f})"


def test_apply_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        PlattCalibrator().apply(np.array([0.5]))
