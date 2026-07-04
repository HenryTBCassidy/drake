"""Probability-prediction metrics (docs/03-EVALUATION-PLAN.md § Primary Metrics)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

_EPSILON = 1e-12


@dataclass(frozen=True)
class MetricSet:
    """All primary metrics for one slice of predictions."""

    num_rows: int
    log_loss: float
    brier_score: float
    auc: float
    ece: float
    accuracy: float


def compute_metrics(labels: NDArray[np.int_], probabilities: NDArray[np.float64], ece_bins: int = 20) -> MetricSet:
    """Compute every primary metric for one prediction slice."""
    return MetricSet(
        num_rows=int(labels.size),
        log_loss=log_loss(labels, probabilities),
        brier_score=brier_score(labels, probabilities),
        auc=auc(labels, probabilities),
        ece=expected_calibration_error(labels, probabilities, ece_bins),
        accuracy=float(((probabilities > 0.5).astype(int) == labels).mean()),
    )


def log_loss(labels: NDArray[np.int_], probabilities: NDArray[np.float64]) -> float:
    clipped = np.clip(probabilities, _EPSILON, 1 - _EPSILON)
    return float(-(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped)).mean())


def brier_score(labels: NDArray[np.int_], probabilities: NDArray[np.float64]) -> float:
    return float(((probabilities - labels) ** 2).mean())


def auc(labels: NDArray[np.int_], probabilities: NDArray[np.float64]) -> float:
    """Area under the ROC curve via the rank-sum (Mann-Whitney) formulation.

    Returns NaN when a slice contains only one class — AUC is undefined there.
    """
    positives = probabilities[labels == 1]
    negatives = probabilities[labels == 0]
    if positives.size == 0 or negatives.size == 0:
        return float("nan")
    combined = np.concatenate([positives, negatives])
    # Average rank per tie group (1-based), assigned back through the inverse index.
    _, inverse, counts = np.unique(combined, return_inverse=True, return_counts=True)
    average_ranks = (np.cumsum(counts) - (counts - 1) / 2)[inverse]
    positive_rank_sum = average_ranks[: positives.size].sum()
    u_statistic = positive_rank_sum - positives.size * (positives.size + 1) / 2
    return float(u_statistic / (positives.size * negatives.size))


def expected_calibration_error(
    labels: NDArray[np.int_], probabilities: NDArray[np.float64], num_bins: int = 20
) -> float:
    """Bucketed |confidence - accuracy| gap, weighted by bucket occupancy."""
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    bin_indices = np.clip(np.digitize(probabilities, bin_edges) - 1, 0, num_bins - 1)
    ece_total = 0.0
    for bin_index in range(num_bins):
        in_bin = bin_indices == bin_index
        if not in_bin.any():
            continue
        confidence = probabilities[in_bin].mean()
        observed = labels[in_bin].mean()
        ece_total += in_bin.mean() * abs(confidence - observed)
    return float(ece_total)


def reliability_curve(
    labels: NDArray[np.int_], probabilities: NDArray[np.float64], num_bins: int = 20
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int_]]:
    """Per-bin (mean predicted, observed win rate, count) triples for reliability diagrams."""
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    bin_indices = np.clip(np.digitize(probabilities, bin_edges) - 1, 0, num_bins - 1)
    mean_predicted = np.full(num_bins, np.nan)
    observed_rate = np.full(num_bins, np.nan)
    counts = np.zeros(num_bins, dtype=np.int_)
    for bin_index in range(num_bins):
        in_bin = bin_indices == bin_index
        counts[bin_index] = int(in_bin.sum())
        if counts[bin_index]:
            mean_predicted[bin_index] = probabilities[in_bin].mean()
            observed_rate[bin_index] = labels[in_bin].mean()
    return mean_predicted, observed_rate, counts
