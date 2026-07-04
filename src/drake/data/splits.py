"""Match-level data splits (docs/03-EVALUATION-PLAN.md § Data Splits).

Split by match id, never by row: every timestep of a match lands in the same
split. The newest matches (by gameCreation) form a time-based test set —
train on the past, evaluate on the future — and the remaining development
matches are shuffled into train / validation / calibration.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

    from drake.config import SplitConfig

SPLIT_NAMES = ("train", "val", "calibration", "test")


def create_splits(raw_matches: pd.DataFrame, config: SplitConfig, splits_dir: Path) -> dict[str, list[str]]:
    """Assign every collected match to a split and write the id lists.

    Args:
        raw_matches: Raw collection DataFrame (needs match_id and game_creation_ms).
        config: Split fractions and shuffle seed.
        splits_dir: Where the `{split}_match_ids.txt` files are written.

    Returns:
        Mapping of split name to match ids.
    """
    matches = raw_matches[["match_id", "game_creation_ms"]].drop_duplicates("match_id")
    ordered = matches.sort_values("game_creation_ms")["match_id"].tolist()

    num_test = max(1, int(len(ordered) * config.test_fraction))
    test_ids = ordered[-num_test:]  # newest matches — evaluate on the future
    development_ids = ordered[:-num_test]

    rng = np.random.default_rng(config.seed)
    shuffled = list(rng.permutation(development_ids))
    num_val = max(1, int(len(ordered) * config.val_fraction))
    num_calibration = max(1, int(len(ordered) * config.calibration_fraction))
    splits = {
        "val": [str(match_id) for match_id in shuffled[:num_val]],
        "calibration": [str(match_id) for match_id in shuffled[num_val : num_val + num_calibration]],
        "train": [str(match_id) for match_id in shuffled[num_val + num_calibration :]],
        "test": [str(match_id) for match_id in test_ids],
    }
    if not splits["train"]:
        raise ValueError(f"Only {len(ordered)} matches — not enough to carve out val/calibration/test")

    splits_dir.mkdir(parents=True, exist_ok=True)
    for name, match_ids in splits.items():
        (splits_dir / f"{name}_match_ids.txt").write_text("\n".join(match_ids) + "\n")
    logger.info(
        "Split {} matches -> train={} val={} calibration={} test={}",
        len(ordered),
        len(splits["train"]),
        len(splits["val"]),
        len(splits["calibration"]),
        len(splits["test"]),
    )
    return splits


def load_split_ids(splits_dir: Path, name: str) -> set[str]:
    """Read one split's match ids; raises if the split stage hasn't run."""
    path = splits_dir / f"{name}_match_ids.txt"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run the split stage first")
    return {line for line in path.read_text().splitlines() if line}


def split_rows(processed: pd.DataFrame, splits_dir: Path) -> dict[str, pd.DataFrame]:
    """Partition processed feature rows by the persisted match-id splits."""
    result: dict[str, pd.DataFrame] = {}
    for name in SPLIT_NAMES:
        ids = load_split_ids(splits_dir, name)
        result[name] = processed[processed["match_id"].isin(ids)].reset_index(drop=True)
    return result


def write_split_summary(splits: dict[str, list[str]], splits_dir: Path) -> None:
    """Persist split sizes for quick inspection."""
    summary = {name: len(ids) for name, ids in splits.items()}
    (splits_dir / "summary.json").write_text(json.dumps(summary, indent=2))
