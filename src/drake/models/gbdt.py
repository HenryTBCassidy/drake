"""Model A — GBDT baseline (docs/02-MODEL-ARCHITECTURES.md § Model A).

Two LightGBM binary classifiers: a draft model over champion/context features
(T=0 rows) and an in-game model over game-state features plus the draft
model's own P(win) as an input (T>0 rows). `predict` routes each row to the
right booster by its timestep.

Deviation from docs/01, noted in the plan: champion slots are LightGBM native
categorical features rather than a 1,650-wide one-hot expansion — the same
information, far cheaper.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import lightgbm as lgb
import numpy as np
import pandas as pd
from loguru import logger

from drake.data.features import (
    CONTEXT_COLUMNS,
    DRAFT_COLUMNS,
    GAME_STATE_COLUMNS,
    LABEL_COLUMN,
    build_champion_index,
)
from drake.domain import UNKNOWN_CHAMPION_INDEX
from drake.protocols import IWinProbabilityModel

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from drake.config import GbdtConfig

DRAFT_FEATURE_COLUMNS = DRAFT_COLUMNS + CONTEXT_COLUMNS
DRAFT_PROBABILITY_COLUMN = "draft_probability"
IN_GAME_FEATURE_COLUMNS = GAME_STATE_COLUMNS + CONTEXT_COLUMNS + [DRAFT_PROBABILITY_COLUMN]

_DRAFT_MODEL_FILE = "gbdt_draft.txt"
_IN_GAME_MODEL_FILE = "gbdt_in_game.txt"
_MANIFEST_FILE = "manifest.json"


class GbdtBaseline(IWinProbabilityModel):
    """LightGBM draft + in-game win probability baseline."""

    def __init__(self, config: GbdtConfig) -> None:
        self._config = config
        self._draft_booster: lgb.Booster | None = None
        self._in_game_booster: lgb.Booster | None = None
        self._champion_index: dict[int, int] = {}

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> None:
        """Train the draft model on T=0 rows, then the in-game model on T>0 rows."""
        self._champion_index = build_champion_index(train[train["timestep"] == 0])
        logger.info("Training GBDT draft model on {} draft rows", int((train["timestep"] == 0).sum()))
        self._draft_booster = self._train_booster(
            _draft_matrix(train[train["timestep"] == 0], self._champion_index),
            _draft_matrix(val[val["timestep"] == 0], self._champion_index),
        )
        train_in_game = self._with_draft_probability(train[train["timestep"] > 0])
        val_in_game = self._with_draft_probability(val[val["timestep"] > 0])
        logger.info("Training GBDT in-game model on {} in-game rows", len(train_in_game))
        self._in_game_booster = self._train_booster(
            (train_in_game[IN_GAME_FEATURE_COLUMNS], train_in_game[LABEL_COLUMN]),
            (val_in_game[IN_GAME_FEATURE_COLUMNS], val_in_game[LABEL_COLUMN]),
        )

    def predict(self, rows: pd.DataFrame) -> NDArray[np.float64]:
        """P(blue win) per row; T=0 rows use the draft booster, T>0 the in-game booster."""
        if self._draft_booster is None or self._in_game_booster is None:
            raise RuntimeError("GbdtBaseline is not fitted — call fit() or load() first")
        probabilities = np.empty(len(rows), dtype=np.float64)
        draft_mask = (rows["timestep"] == 0).to_numpy()
        if draft_mask.any():
            features, _ = _draft_matrix(rows[draft_mask], self._champion_index)
            probabilities[draft_mask] = np.asarray(self._draft_booster.predict(features))
        if (~draft_mask).any():
            in_game = self._with_draft_probability(rows[~draft_mask])
            probabilities[~draft_mask] = np.asarray(self._in_game_booster.predict(in_game[IN_GAME_FEATURE_COLUMNS]))
        return probabilities

    def save(self, directory: Path) -> None:
        if self._draft_booster is None or self._in_game_booster is None:
            raise RuntimeError("Nothing to save — the model is not fitted")
        directory.mkdir(parents=True, exist_ok=True)
        self._draft_booster.save_model(str(directory / _DRAFT_MODEL_FILE))
        self._in_game_booster.save_model(str(directory / _IN_GAME_MODEL_FILE))
        manifest = {
            "model": "gbdt",
            "draft_feature_columns": DRAFT_FEATURE_COLUMNS,
            "in_game_feature_columns": IN_GAME_FEATURE_COLUMNS,
            "champion_index": {str(raw): index for raw, index in self._champion_index.items()},
        }
        (directory / _MANIFEST_FILE).write_text(json.dumps(manifest, indent=2))
        logger.info("Saved GBDT baseline to {}", directory)

    def load(self, directory: Path) -> None:
        manifest = json.loads((directory / _MANIFEST_FILE).read_text())
        self._champion_index = {int(raw): int(index) for raw, index in manifest["champion_index"].items()}
        self._draft_booster = lgb.Booster(model_file=str(directory / _DRAFT_MODEL_FILE))
        self._in_game_booster = lgb.Booster(model_file=str(directory / _IN_GAME_MODEL_FILE))

    def _train_booster(
        self,
        train_data: tuple[pd.DataFrame, pd.Series],
        val_data: tuple[pd.DataFrame, pd.Series],
    ) -> lgb.Booster:
        config = self._config
        train_features, train_labels = train_data
        val_features, val_labels = val_data
        train_set = lgb.Dataset(train_features, label=train_labels)
        val_set = lgb.Dataset(val_features, label=val_labels, reference=train_set)
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "num_leaves": config.num_leaves,
            "max_depth": config.max_depth,
            "learning_rate": config.learning_rate,
            "subsample": config.subsample,
            "colsample_bytree": config.colsample_bytree,
            "min_child_samples": config.min_child_samples,
            "reg_alpha": config.reg_alpha,
            "reg_lambda": config.reg_lambda,
            "verbosity": -1,
        }
        return lgb.train(
            params,
            train_set,
            num_boost_round=config.n_estimators,
            valid_sets=[val_set],
            callbacks=[lgb.early_stopping(config.early_stopping_rounds, verbose=False)],
        )

    def _with_draft_probability(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Append the draft model's P(win) as an input feature for the in-game model."""
        if self._draft_booster is None:
            raise RuntimeError("Draft booster must be trained before the in-game model")
        features, _ = _draft_matrix(rows, self._champion_index)
        enriched = rows.copy()
        enriched[DRAFT_PROBABILITY_COLUMN] = np.asarray(self._draft_booster.predict(features))
        return enriched


def _draft_matrix(rows: pd.DataFrame, champion_index: dict[int, int]) -> tuple[pd.DataFrame, pd.Series]:
    """Draft feature matrix with champion slots as fixed-vocabulary categoricals.

    Champion ids are remapped to contiguous indices (unseen -> UNKNOWN) before
    becoming categoricals, so the codes are stable between fit and predict AND
    cover Riot's sparse real ids (1..~950) — not just a contiguous 0-164 range,
    which silently dropped most of the roster to NaN.
    """
    features = rows[DRAFT_FEATURE_COLUMNS].copy()
    for column in DRAFT_COLUMNS:
        remapped = [champion_index.get(int(champion), UNKNOWN_CHAMPION_INDEX) for champion in features[column]]
        features[column] = pd.Categorical(remapped, categories=range(UNKNOWN_CHAMPION_INDEX + 1))
    labels = rows[LABEL_COLUMN] if LABEL_COLUMN in rows.columns else pd.Series(np.zeros(len(rows)))
    return features, labels
