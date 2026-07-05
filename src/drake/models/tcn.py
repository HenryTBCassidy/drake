"""Model B — TCN unified win probability model (docs/02-MODEL-ARCHITECTURES.md § Model B).

One network, draft through nexus: champion/tier/region/patch embeddings feed a
draft encoder MLP; the resulting draft vector is concatenated to every
timestep's game features; a stack of causal dilated residual conv blocks turns
the sequence into per-timestep hidden states; a shared win head emits P(win)
at every timestep (T=0 is the draft, with zeroed game features).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch import nn

from drake.data.features import DRAFT_COLUMNS, GAME_STATE_COLUMNS, LABEL_COLUMN, build_champion_index
from drake.domain import UNKNOWN_CHAMPION_INDEX
from drake.protocols import IWinProbabilityModel
from drake.training.trainer import MatchBatch, TcnTrainer, select_device

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from drake.config import TcnConfig

_NUM_TIERS = 9
_NUM_REGIONS = 10
_NUM_PATCH_MAJOR = 20
_NUM_PATCH_MINOR = 30
_MODEL_FILE = "tcn.pt"
_MANIFEST_FILE = "manifest.json"


class TcnNet(nn.Module):
    """The docs/02 architecture: embeddings -> draft encoder -> causal TCN -> win head."""

    def __init__(self, config: TcnConfig, num_game_features: int) -> None:
        super().__init__()
        champion_dim = config.champion_embedding_dim
        self.champion_embeddings = nn.ModuleList(
            [nn.Embedding(UNKNOWN_CHAMPION_INDEX + 1, champion_dim) for _ in DRAFT_COLUMNS]
        )
        self.tier_embedding = nn.Embedding(_NUM_TIERS, config.tier_embedding_dim)
        self.region_embedding = nn.Embedding(_NUM_REGIONS, config.region_embedding_dim)
        self.patch_major_embedding = nn.Embedding(_NUM_PATCH_MAJOR, config.patch_embedding_dim)
        self.patch_minor_embedding = nn.Embedding(_NUM_PATCH_MINOR, config.patch_embedding_dim)

        draft_input_dim = (
            len(DRAFT_COLUMNS) * champion_dim
            + config.tier_embedding_dim
            + 1  # lp_proxy
            + config.region_embedding_dim
            + 2 * config.patch_embedding_dim
            + 1  # season_progress
        )
        self.draft_encoder = nn.Sequential(
            nn.Linear(draft_input_dim, config.draft_hidden_dim),
            nn.LayerNorm(config.draft_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.draft_hidden_dim, config.draft_vec_dim),
            nn.LayerNorm(config.draft_vec_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )
        self.input_projection = nn.Linear(config.draft_vec_dim + num_game_features, config.tcn_channels)
        self.tcn_blocks = nn.ModuleList(
            [
                _CausalResidualBlock(config.tcn_channels, config.tcn_kernel_size, dilation, config.dropout)
                for dilation in config.tcn_dilations
            ]
        )
        self.win_head = nn.Sequential(
            nn.Linear(config.tcn_channels, 64),
            nn.ReLU(),
            nn.Dropout(config.head_dropout),
            nn.Linear(64, 1),
        )

    def forward(self, batch: MatchBatch) -> torch.Tensor:
        """Per-timestep win logits, shape (batch, max_seq_len)."""
        champion_vectors = [table(batch.champions[:, slot]) for slot, table in enumerate(self.champion_embeddings)]
        draft_input = torch.cat(
            [
                *champion_vectors,
                self.tier_embedding(batch.tier),
                batch.lp_proxy,
                self.region_embedding(batch.region),
                self.patch_major_embedding(batch.patch_major),
                self.patch_minor_embedding(batch.patch_minor),
                batch.season_progress,
            ],
            dim=-1,
        )
        draft_vec = self.draft_encoder(draft_input)  # (batch, draft_vec_dim)

        max_seq_len = batch.game_features.shape[1]
        draft_expanded = draft_vec.unsqueeze(1).expand(-1, max_seq_len, -1)
        timestep_input = torch.cat([draft_expanded, batch.game_features], dim=-1)
        hidden = self.input_projection(timestep_input).transpose(1, 2)  # (batch, channels, T)
        for block in self.tcn_blocks:
            hidden = block(hidden)
        hidden = hidden.transpose(1, 2)  # (batch, T, channels)
        logits: torch.Tensor = self.win_head(hidden).squeeze(-1)
        return logits


class _CausalResidualBlock(nn.Module):
    """Two causal dilated convolutions with LayerNorm/ReLU/Dropout and a residual shortcut.

    Causality: left-pad by (kernel - 1) * dilation so position t never sees t+1.
    """

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self._left_padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self._conv_norm_relu(x, self.conv1, self.norm1)
        x = self._conv_norm_relu(x, self.conv2, self.norm2)
        return x + residual

    def _conv_norm_relu(self, x: torch.Tensor, conv: nn.Conv1d, norm: nn.LayerNorm) -> torch.Tensor:
        x = conv(nn.functional.pad(x, (self._left_padding, 0)))
        x = norm(x.transpose(1, 2)).transpose(1, 2)  # LayerNorm over channels
        return self.dropout(torch.relu(x))


class TcnModel(IWinProbabilityModel):
    """IWinProbabilityModel wrapper: dataframe rows in, probabilities out."""

    def __init__(self, config: TcnConfig) -> None:
        self._config = config
        self._net: TcnNet | None = None
        self._champion_index: dict[int, int] = {}
        self._feature_means: NDArray[np.float64] | None = None
        self._feature_stds: NDArray[np.float64] | None = None

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> None:
        """Train on processed rows with UNKNOWN-token draft masking and early stopping."""
        self._champion_index = build_champion_index(train)
        # Standardise game-state features on train stats — raw gold diffs are in the
        # thousands and would saturate the win head's sigmoid.
        game_state = train[GAME_STATE_COLUMNS].to_numpy(dtype=np.float64)
        self._feature_means = game_state.mean(axis=0)
        self._feature_stds = np.maximum(game_state.std(axis=0), 1e-6)
        self._net = TcnNet(self._config, num_game_features=len(GAME_STATE_COLUMNS))
        trainer = TcnTrainer(self._config)
        trainer.train(
            self._net,
            train_matches=self._to_tensors(train),
            val_matches=self._to_tensors(val),
        )

    def predict(self, rows: pd.DataFrame) -> NDArray[np.float64]:
        """P(blue win) per row.

        The TCN is causal, so each match's rows must form a contiguous timestep
        prefix (0..N) — which is exactly what the processed schema stores.
        """
        if self._net is None:
            raise RuntimeError("TcnModel is not fitted — call fit() or load() first")
        device = select_device(self._config.device)
        net = self._net.to(device)
        net.eval()
        matches = self._to_tensors(rows)
        probabilities = np.empty(len(rows), dtype=np.float64)
        with torch.no_grad():
            for match in matches:
                batch = match.batch_of_one(device)
                logits = net(batch)[0, : len(match.row_positions)]
                probabilities[match.row_positions] = torch.sigmoid(logits).cpu().numpy()
        return probabilities

    def save(self, directory: Path) -> None:
        if self._net is None:
            raise RuntimeError("Nothing to save — the model is not fitted")
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self._net.state_dict(), directory / _MODEL_FILE)
        if self._feature_means is None or self._feature_stds is None:
            raise RuntimeError("Feature scaling stats are missing — the model was never fitted")
        manifest = {
            "model": "tcn",
            "champion_index": {str(raw): index for raw, index in self._champion_index.items()},
            "num_game_features": len(GAME_STATE_COLUMNS),
            "feature_means": self._feature_means.tolist(),
            "feature_stds": self._feature_stds.tolist(),
        }
        (directory / _MANIFEST_FILE).write_text(json.dumps(manifest, indent=2))
        logger.info("Saved TCN model to {}", directory)

    def load(self, directory: Path) -> None:
        manifest = json.loads((directory / _MANIFEST_FILE).read_text())
        self._champion_index = {int(raw): int(index) for raw, index in manifest["champion_index"].items()}
        self._feature_means = np.asarray(manifest["feature_means"], dtype=np.float64)
        self._feature_stds = np.asarray(manifest["feature_stds"], dtype=np.float64)
        self._net = TcnNet(self._config, num_game_features=manifest["num_game_features"])
        self._net.load_state_dict(torch.load(directory / _MODEL_FILE, map_location="cpu", weights_only=True))

    def _to_tensors(self, rows: pd.DataFrame) -> list[MatchTensors]:
        if self._feature_means is None or self._feature_stds is None:
            raise RuntimeError("TcnModel is not fitted — call fit() or load() first")
        return _to_match_tensors(rows, self._champion_index, self._feature_means, self._feature_stds)


class MatchTensors:
    """One match's model inputs, kept on CPU until batched."""

    def __init__(
        self,
        champions: torch.Tensor,  # (10,) int64 — contiguous embedding indices
        tier: int,
        lp_proxy: float,
        region: int,
        patch_major: int,
        patch_minor: int,
        season_progress: float,
        game_features: torch.Tensor,  # (T, F) float32, timestep 0 first
        label: float,
        row_positions: NDArray[np.int_],  # positions of this match's rows in the source frame
    ) -> None:
        self.champions = champions
        self.tier = tier
        self.lp_proxy = lp_proxy
        self.region = region
        self.patch_major = patch_major
        self.patch_minor = patch_minor
        self.season_progress = season_progress
        self.game_features = game_features
        self.label = label
        self.row_positions = row_positions

    @property
    def seq_len(self) -> int:
        return int(self.game_features.shape[0])

    def batch_of_one(self, device: torch.device) -> MatchBatch:
        return MatchBatch.collate([self], device)


def _to_match_tensors(
    rows: pd.DataFrame,
    champion_index: dict[int, int],
    feature_means: NDArray[np.float64],
    feature_stds: NDArray[np.float64],
) -> list[MatchTensors]:
    """Group processed rows into standardised per-match sequences ordered by timestep."""
    matches: list[MatchTensors] = []
    positions = pd.Series(np.arange(len(rows)), index=rows.index)
    for _, match_rows in rows.groupby("match_id", sort=False):
        ordered = match_rows.sort_values("timestep")
        first = ordered.iloc[0]
        champions = torch.tensor(
            [champion_index.get(int(first[column]), UNKNOWN_CHAMPION_INDEX) for column in DRAFT_COLUMNS],
            dtype=torch.int64,
        )
        standardised = (ordered[GAME_STATE_COLUMNS].to_numpy(dtype=np.float64) - feature_means) / feature_stds
        game_features = torch.tensor(standardised.astype(np.float32))
        matches.append(
            MatchTensors(
                champions=champions,
                tier=int(first["tier"]),
                lp_proxy=float(first["lp_proxy"]),
                region=int(first["region"]),
                patch_major=int(first["patch_major"]) % _NUM_PATCH_MAJOR,
                patch_minor=int(first["patch_minor"]) % _NUM_PATCH_MINOR,
                season_progress=float(first["season_progress"]),
                game_features=game_features,
                label=float(first[LABEL_COLUMN]) if LABEL_COLUMN in ordered.columns else 0.0,
                row_positions=positions.loc[ordered.index].to_numpy(),
            )
        )
    return matches
