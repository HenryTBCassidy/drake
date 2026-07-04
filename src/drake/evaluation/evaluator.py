"""Evaluation harness: runs a model over the test split and writes the report.

Produces the docs/03 evaluation matrix — every primary metric at every
timestamp (draft, 5m, 10m, ...), per tier, raw and Platt-calibrated — as a
tidy metrics Parquet, a markdown report, and plots.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from loguru import logger

from drake.domain import Tier
from drake.evaluation.calibration import PlattCalibrator
from drake.evaluation.metrics import compute_metrics
from drake.evaluation.plots import save_metric_vs_timestamp, save_reliability_diagram

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from drake.config import EvaluationConfig
    from drake.protocols import IWinProbabilityModel

METRICS_FILE = "metrics.parquet"
REPORT_FILE = "report.md"
_REPORT_METRIC_COLUMNS = ["num_rows", "log_loss", "brier_score", "auc", "ece", "accuracy"]


class Evaluator:
    """Evaluates any IWinProbabilityModel on the held-out test split."""

    def __init__(self, config: EvaluationConfig) -> None:
        self._config = config

    def evaluate(
        self,
        model: IWinProbabilityModel,
        by_split: dict[str, pd.DataFrame],
        results_dir: Path,
        model_name: str,
    ) -> pd.DataFrame:
        """Run the full evaluation matrix and write metrics, report, and plots.

        Returns the tidy metrics DataFrame (one row per slice x calibration state).
        """
        test = by_split["test"]
        test_labels = test["label"].to_numpy()
        raw_probabilities = model.predict(test)

        probability_sets: dict[bool, NDArray[np.float64]] = {False: raw_probabilities}
        if self._config.calibrate:
            calibrator = PlattCalibrator()
            calibration_rows = by_split["calibration"]
            calibrator.fit(calibration_rows["label"].to_numpy(), model.predict(calibration_rows))
            probability_sets[True] = calibrator.apply(raw_probabilities)

        metric_rows: list[dict[str, object]] = []
        for calibrated, probabilities in probability_sets.items():
            metric_rows.append(
                self._metric_row(model_name, "overall", "all", None, calibrated, test_labels, probabilities)
            )
            for minutes in self._config.timestamps_minutes:
                slice_mask = _timestamp_mask(test, minutes)
                if not slice_mask.any():
                    logger.warning("No test rows at t={}m — skipping that timestamp", minutes)
                    continue
                metric_rows.append(
                    self._metric_row(
                        model_name,
                        "timestamp",
                        f"{minutes}m" if minutes else "draft",
                        minutes,
                        calibrated,
                        test_labels[slice_mask],
                        probabilities[slice_mask],
                    )
                )
            if self._config.per_tier:
                for tier_code in sorted(test["tier"].unique()):
                    tier_mask = (test["tier"] == tier_code).to_numpy()
                    metric_rows.append(
                        self._metric_row(
                            model_name,
                            "tier",
                            list(Tier)[int(tier_code)].value,
                            None,
                            calibrated,
                            test_labels[tier_mask],
                            probabilities[tier_mask],
                        )
                    )

        metrics_frame = pd.DataFrame(metric_rows)
        results_dir.mkdir(parents=True, exist_ok=True)
        metrics_frame.to_parquet(results_dir / METRICS_FILE, index=False)
        self._write_plots(test_labels, probability_sets, metrics_frame, results_dir, model_name)
        _write_report(metrics_frame, results_dir, model_name, num_test_matches=test["match_id"].nunique())
        logger.info("Evaluation written to {}", results_dir)
        return metrics_frame

    def _metric_row(
        self,
        model_name: str,
        slice_type: str,
        slice_name: str,
        timestamp_minutes: int | None,
        calibrated: bool,
        labels: NDArray[np.int_],
        probabilities: NDArray[np.float64],
    ) -> dict[str, object]:
        metrics = compute_metrics(labels, probabilities, self._config.ece_bins)
        return {
            "model": model_name,
            "slice_type": slice_type,
            "slice": slice_name,
            "timestamp_minutes": timestamp_minutes,
            "calibrated": calibrated,
            **asdict(metrics),
        }

    def _write_plots(
        self,
        test_labels: NDArray[np.int_],
        probability_sets: dict[bool, NDArray[np.float64]],
        metrics_frame: pd.DataFrame,
        results_dir: Path,
        model_name: str,
    ) -> None:
        for calibrated, probabilities in probability_sets.items():
            suffix = "calibrated" if calibrated else "raw"
            save_reliability_diagram(
                test_labels,
                probabilities,
                results_dir / f"reliability_{suffix}.png",
                f"{model_name} reliability ({suffix}, all test rows)",
                self._config.reliability_bins,
            )
        timestamp_metrics = metrics_frame[metrics_frame["slice_type"] == "timestamp"]
        for metric in ("log_loss", "auc", "ece"):
            save_metric_vs_timestamp(
                timestamp_metrics,
                metric,
                results_dir / f"{metric}_vs_timestamp.png",
                f"{model_name}: {metric.replace('_', ' ')} across the game",
            )


def _timestamp_mask(test: pd.DataFrame, minutes: int) -> NDArray[np.bool_]:
    if minutes == 0:
        return (test["timestep"] == 0).to_numpy()
    return (test["game_time_sec"] == minutes * 60).to_numpy()


def _write_report(metrics_frame: pd.DataFrame, results_dir: Path, model_name: str, num_test_matches: int) -> None:
    lines = [
        f"# DRAKE evaluation — {model_name}",
        "",
        f"Test split: {num_test_matches} matches (time-based holdout, newest matches).",
        "Metrics per docs/03-EVALUATION-PLAN.md; calibration = Platt scaling fit on the calibration split.",
        "",
        "## Evaluation matrix (by timestamp)",
        "",
        _markdown_table(metrics_frame[(metrics_frame["slice_type"] == "timestamp") & ~metrics_frame["calibrated"]]),
    ]
    if metrics_frame["calibrated"].any():
        lines += [
            "",
            "## Calibrated (Platt) — by timestamp",
            "",
            _markdown_table(metrics_frame[(metrics_frame["slice_type"] == "timestamp") & metrics_frame["calibrated"]]),
        ]
    per_tier = metrics_frame[(metrics_frame["slice_type"] == "tier") & ~metrics_frame["calibrated"]]
    if len(per_tier) > 1:
        lines += ["", "## Per-tier (raw)", "", _markdown_table(per_tier)]
    overall = metrics_frame[(metrics_frame["slice_type"] == "overall")]
    lines += [
        "",
        "## Overall (all test rows)",
        "",
        _markdown_table(overall),
        "",
        "Plots: `reliability_raw.png`"
        + (", `reliability_calibrated.png`," if metrics_frame["calibrated"].any() else ","),
        "`log_loss_vs_timestamp.png`, `auc_vs_timestamp.png`, `ece_vs_timestamp.png`.",
        "",
    ]
    (results_dir / REPORT_FILE).write_text("\n".join(lines))


def _markdown_table(rows: pd.DataFrame) -> str:
    header = "| slice | " + " | ".join(_REPORT_METRIC_COLUMNS) + " |"
    divider = "|" + "---|" * (len(_REPORT_METRIC_COLUMNS) + 1)
    body = [
        "| "
        + str(row["slice"])
        + " | "
        + " | ".join(_format_metric(row[column]) for column in _REPORT_METRIC_COLUMNS)
        + " |"
        for _, row in rows.iterrows()
    ]
    return "\n".join([header, divider, *body])


def _format_metric(value: object) -> str:
    if isinstance(value, float):
        return "nan" if np.isnan(value) else f"{value:.4f}"
    return str(value)
