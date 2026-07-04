"""Evaluation plots: reliability diagram and metric-vs-timestamp curves.

Uses matplotlib's object-oriented API (no pyplot) so plotting is backend- and
global-state-free — safe in headless CI.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger
from matplotlib.figure import Figure

from drake.evaluation.metrics import reliability_curve

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from numpy.typing import NDArray


def save_reliability_diagram(
    labels: NDArray[np.int_],
    probabilities: NDArray[np.float64],
    path: Path,
    title: str,
    num_bins: int = 20,
) -> None:
    """Predicted probability vs observed win rate; the diagonal is perfect calibration."""
    mean_predicted, observed_rate, counts = reliability_curve(labels, probabilities, num_bins)
    figure = Figure(figsize=(6, 6))
    axes = figure.subplots()
    axes.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfect calibration")
    occupied = counts > 0
    axes.plot(mean_predicted[occupied], observed_rate[occupied], marker="o", color="tab:blue", label="Model")
    axes.set_xlabel("Predicted P(blue win)")
    axes.set_ylabel("Observed blue win rate")
    axes.set_title(title)
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    axes.legend()
    _save(figure, path)


def save_metric_vs_timestamp(timestamp_metrics: pd.DataFrame, metric: str, path: Path, title: str) -> None:
    """One line per calibration state: `metric` across the docs/03 evaluation timestamps."""
    figure = Figure(figsize=(8, 5))
    axes = figure.subplots()
    for calibrated, rows in timestamp_metrics.groupby("calibrated"):
        ordered = rows.sort_values("timestamp_minutes")
        label = "Calibrated" if calibrated else "Raw"
        axes.plot(ordered["timestamp_minutes"], ordered[metric], marker="o", label=label)
    axes.set_xlabel("Game time (minutes; 0 = draft)")
    axes.set_ylabel(metric.replace("_", " "))
    axes.set_title(title)
    axes.legend()
    _save(figure, path)


def _save(figure: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    logger.info("Wrote plot {}", path)
