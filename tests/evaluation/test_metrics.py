"""Tests for drake.evaluation.metrics."""

from __future__ import annotations

import numpy as np
import pytest

from drake.evaluation.metrics import (
    auc,
    brier_score,
    compute_metrics,
    expected_calibration_error,
    log_loss,
    reliability_curve,
)


def test_log_loss_matches_hand_computation() -> None:
    labels = np.array([1, 0])
    probabilities = np.array([0.8, 0.4])
    expected = -(np.log(0.8) + np.log(0.6)) / 2
    assert log_loss(labels, probabilities) == pytest.approx(expected)


def test_log_loss_is_safe_at_probability_extremes() -> None:
    labels = np.array([1, 0])
    probabilities = np.array([0.0, 1.0])  # maximally wrong AND degenerate
    assert np.isfinite(log_loss(labels, probabilities))


def test_brier_score_perfect_and_worst() -> None:
    labels = np.array([1, 0, 1])
    assert brier_score(labels, labels.astype(np.float64)) == 0.0
    assert brier_score(labels, 1.0 - labels.astype(np.float64)) == 1.0


def test_auc_perfect_random_and_inverted() -> None:
    labels = np.array([0, 0, 1, 1])
    assert auc(labels, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert auc(labels, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert auc(labels, np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.5), "all-tied ranks give AUC 0.5"


def test_auc_single_class_is_nan() -> None:
    assert np.isnan(auc(np.array([1, 1]), np.array([0.5, 0.6])))


def test_ece_zero_for_perfectly_calibrated_buckets() -> None:
    # 100 predictions at 0.3 with 30% positives, 100 at 0.7 with 70% positives.
    labels = np.concatenate([np.zeros(70), np.ones(30), np.zeros(30), np.ones(70)]).astype(int)
    probabilities = np.concatenate([np.full(100, 0.3), np.full(100, 0.7)])
    assert expected_calibration_error(labels, probabilities, num_bins=10) == pytest.approx(0.0)


def test_ece_detects_systematic_overconfidence() -> None:
    labels = np.concatenate([np.ones(50), np.zeros(50)]).astype(int)  # 50% base rate
    probabilities = np.full(100, 0.9)  # says 90%
    assert expected_calibration_error(labels, probabilities, num_bins=10) == pytest.approx(0.4)


def test_reliability_curve_bins_line_up() -> None:
    labels = np.array([0, 1, 1, 1])
    probabilities = np.array([0.05, 0.05, 0.95, 0.95])
    mean_predicted, observed_rate, counts = reliability_curve(labels, probabilities, num_bins=10)
    assert counts.sum() == 4
    assert mean_predicted[0] == pytest.approx(0.05)
    assert observed_rate[0] == pytest.approx(0.5)
    assert observed_rate[9] == pytest.approx(1.0)


def test_compute_metrics_bundles_everything() -> None:
    labels = np.array([0, 1, 0, 1, 1])
    probabilities = np.array([0.2, 0.7, 0.4, 0.9, 0.6])
    metrics = compute_metrics(labels, probabilities)
    assert metrics.num_rows == 5
    assert metrics.accuracy == 1.0
    assert metrics.auc == 1.0
    assert 0 < metrics.log_loss < 0.7
