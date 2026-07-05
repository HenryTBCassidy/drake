"""Run configuration: frozen dataclasses loaded from JSON files.

A run is fully described by one JSON file in `run_configurations/` (native JSON
types only). Missing sections/fields fall back to the dataclass defaults, which
mirror the documented hyperparameters (docs/02-MODEL-ARCHITECTURES.md).
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

ConfigT = TypeVar("ConfigT")


@dataclass(frozen=True)
class PathsConfig:
    """Root directories and the derived stage layout under them (docs/01 § Directory Layout)."""

    data_dir: Path = Path("data")
    models_dir: Path = Path("models")
    results_dir: Path = Path("results")

    @property
    def seed_players_dir(self) -> Path:
        return self.data_dir / "raw" / "seed_players"

    @property
    def raw_matches_dir(self) -> Path:
        return self.data_dir / "raw" / "matches"

    @property
    def game_features_dir(self) -> Path:
        return self.data_dir / "processed" / "game_features"

    @property
    def feature_metadata_path(self) -> Path:
        return self.data_dir / "processed" / "metadata.json"

    @property
    def checkpoint_db_path(self) -> Path:
        return self.data_dir / "checkpoints" / "collection.db"

    @property
    def splits_dir(self) -> Path:
        return self.data_dir / "splits"


@dataclass(frozen=True)
class CollectionConfig:
    """Riot collection parameters (docs/01 § Stable-Anchor Player Seeding)."""

    regions: tuple[str, ...] = ("na1",)
    tiers: tuple[str, ...] = ("GOLD",)
    matches_per_player: int = 20
    max_anchors_per_tier: int = 200
    max_league_pages: int = 10
    min_ranked_games: int = 150
    min_win_rate: float = 0.47
    max_win_rate: float = 0.53
    require_veteran: bool = True  # Riot's veteran flag is rare — set false to widen the anchor pool
    requests_per_second: int = 20
    requests_per_two_minutes: int = 100


@dataclass(frozen=True)
class SyntheticConfig:
    """Synthetic data generation (reference mode — no Riot key required)."""

    matches_per_tier: int = 1000  # unique-match pool size per (region, tier)
    seed: int = 7
    anchor_fraction: float = 0.6


@dataclass(frozen=True)
class FeatureConfig:
    """Feature engineering parameters (docs/01 § Feature Engineering)."""

    timestep_seconds: int = 30
    momentum_window_short_seconds: int = 120
    momentum_window_long_seconds: int = 300


@dataclass(frozen=True)
class SplitConfig:
    """Match-level split fractions (docs/03 § Data Splits)."""

    test_fraction: float = 0.10
    val_fraction: float = 0.10
    calibration_fraction: float = 0.10
    seed: int = 7


@dataclass(frozen=True)
class GbdtConfig:
    """LightGBM hyperparameters for Model A (docs/02 § Model A)."""

    num_leaves: int = 127
    max_depth: int = 8
    learning_rate: float = 0.05
    n_estimators: int = 1500
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_samples: int = 50
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    early_stopping_rounds: int = 50


@dataclass(frozen=True)
class TcnConfig:
    """TCN unified model and training hyperparameters (docs/02 § Model B)."""

    champion_embedding_dim: int = 32
    tier_embedding_dim: int = 16
    region_embedding_dim: int = 8
    patch_embedding_dim: int = 8
    draft_vec_dim: int = 128
    draft_hidden_dim: int = 256
    tcn_channels: int = 128
    tcn_kernel_size: int = 3
    tcn_dilations: tuple[int, ...] = (1, 2, 4, 8, 16)
    dropout: float = 0.2
    head_dropout: float = 0.1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 50
    early_stopping_patience: int = 5
    gradient_clip_norm: float = 1.0
    unknown_mask_probability: float = 0.3  # chance a game gets 1-3 champions masked
    device: str = "auto"  # auto -> cuda > mps > cpu


@dataclass(frozen=True)
class EvaluationConfig:
    """Evaluation matrix parameters (docs/03 § Evaluation Matrix)."""

    timestamps_minutes: tuple[int, ...] = (0, 5, 10, 15, 20, 25, 30)
    ece_bins: int = 20
    reliability_bins: int = 20
    per_tier: bool = True
    calibrate: bool = True


@dataclass(frozen=True)
class RunConfig:
    """Top-level run description: which source, which model, and all stage parameters."""

    name: str = "run"
    source: str = "synthetic"  # synthetic | riot
    model: str = "gbdt"  # gbdt | tcn
    paths: PathsConfig = field(default_factory=PathsConfig)
    collection: CollectionConfig = field(default_factory=CollectionConfig)
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    gbdt: GbdtConfig = field(default_factory=GbdtConfig)
    tcn: TcnConfig = field(default_factory=TcnConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @property
    def model_dir(self) -> Path:
        return self.paths.models_dir / self.name / self.model

    @property
    def results_dir(self) -> Path:
        return self.paths.results_dir / self.name / self.model

    @classmethod
    def from_json(cls, path: Path) -> RunConfig:
        """Load a run configuration from a JSON file, defaulting missing fields."""
        raw = json.loads(path.read_text())
        return _build_dataclass(cls, raw)


def _build_dataclass(cls: type[ConfigT], raw: dict[str, Any]) -> ConfigT:
    """Recursively construct a (frozen) dataclass from a plain JSON dict.

    Handles nested dataclasses, Path fields, and list -> tuple coercion. Unknown
    keys are rejected loudly — a typo in a run file should fail, not silently
    fall back to a default.
    """
    field_map = {f.name: f for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    unknown = set(raw) - set(field_map)
    if unknown:
        raise ValueError(f"Unknown config keys for {cls.__name__}: {sorted(unknown)}")

    kwargs: dict[str, Any] = {}
    for name, value in raw.items():
        kwargs[name] = _coerce_field(field_map[name].type, value)
    return cls(**kwargs)


def _coerce_field(field_type: str | type, value: Any) -> Any:
    # Field types are strings because of `from __future__ import annotations`.
    type_name = field_type if isinstance(field_type, str) else field_type.__name__
    if isinstance(value, dict):
        nested = _CONFIG_TYPES.get(type_name)
        if nested is None:
            raise ValueError(f"Unexpected nested object for field of type {type_name}")
        return _build_dataclass(nested, value)
    if type_name == "Path":
        return Path(str(value))
    if isinstance(value, list):
        return tuple(value)
    return value


_CONFIG_TYPES: dict[str, type] = {
    cls.__name__: cls
    for cls in (
        PathsConfig,
        CollectionConfig,
        SyntheticConfig,
        FeatureConfig,
        SplitConfig,
        GbdtConfig,
        TcnConfig,
        EvaluationConfig,
    )
}
