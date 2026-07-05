"""Tests for drake.models.tcn (and the training loop it drives)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from drake.config import TcnConfig
from drake.data.features import GAME_STATE_COLUMNS, build_champion_index
from drake.domain import UNKNOWN_CHAMPION_INDEX
from drake.models.tcn import TcnModel, TcnNet, _to_match_tensors
from drake.training.trainer import MatchBatch

if TYPE_CHECKING:
    from pathlib import Path

    from tests.conftest import PipelineRun

TINY_CONFIG = TcnConfig(
    champion_embedding_dim=8,
    tier_embedding_dim=4,
    region_embedding_dim=4,
    patch_embedding_dim=4,
    draft_vec_dim=32,
    draft_hidden_dim=64,
    tcn_channels=32,
    tcn_dilations=(1, 2, 4),
    batch_size=16,
    max_epochs=3,
    early_stopping_patience=3,
    device="cpu",
)


def test_default_config_parameter_count_is_in_the_documented_ballpark() -> None:
    """docs/02 estimates ~550k-640k; the exact architecture computes to ~705k.

    The docs' per-block figure (~66k) undercounts two 128x128x3 convolutions
    (~99k with norms) — the architecture itself matches the spec exactly.
    """
    net = TcnNet(TcnConfig(), num_game_features=len(GAME_STATE_COLUMNS))
    num_parameters = sum(parameter.numel() for parameter in net.parameters())
    assert 500_000 <= num_parameters <= 800_000, f"got {num_parameters} parameters"


def test_forward_is_causal(pipeline: PipelineRun) -> None:
    """Perturbing a late timestep's features must not change earlier predictions."""
    torch.manual_seed(0)
    net = TcnNet(TINY_CONFIG, num_game_features=len(GAME_STATE_COLUMNS))
    net.eval()
    val = pipeline.by_split["val"]
    stats = val[GAME_STATE_COLUMNS].to_numpy(dtype=np.float64)
    matches = _to_match_tensors(
        val,
        build_champion_index(pipeline.by_split["train"]),
        stats.mean(axis=0),
        np.maximum(stats.std(axis=0), 1e-6),
    )
    batch = MatchBatch.collate(matches[:1], torch.device("cpu"))
    with torch.no_grad():
        baseline = net(batch)
    perturbed_features = batch.game_features.clone()
    cut = perturbed_features.shape[1] // 2
    perturbed_features[:, cut:, :] += 100.0
    perturbed_batch = MatchBatch(
        champions=batch.champions,
        tier=batch.tier,
        lp_proxy=batch.lp_proxy,
        region=batch.region,
        patch_major=batch.patch_major,
        patch_minor=batch.patch_minor,
        season_progress=batch.season_progress,
        game_features=perturbed_features,
        labels=batch.labels,
        timestep_mask=batch.timestep_mask,
    )
    with torch.no_grad():
        perturbed = net(perturbed_batch)
    assert torch.allclose(baseline[:, :cut], perturbed[:, :cut], atol=1e-5), "future features leaked into the past"
    assert not torch.allclose(baseline[:, cut:], perturbed[:, cut:], atol=1e-3), "perturbation must matter later"


@pytest.fixture(scope="session")
def fitted_tcn(pipeline: PipelineRun) -> TcnModel:
    torch.manual_seed(7)
    model = TcnModel(TINY_CONFIG)
    model.fit(pipeline.by_split["train"], pipeline.by_split["val"])
    return model


@pytest.mark.slow
def test_tiny_training_learns_the_in_game_signal(fitted_tcn: TcnModel, pipeline: PipelineRun) -> None:
    test_rows = pipeline.by_split["test"]
    probabilities = fitted_tcn.predict(test_rows)
    assert probabilities.shape == (len(test_rows),)
    assert np.isfinite(probabilities).all()
    late = test_rows["game_time_sec"] >= 15 * 60
    accuracy = ((probabilities[late] > 0.5).astype(int) == test_rows.loc[late, "label"].to_numpy()).mean()
    assert accuracy > 0.65, f"late-game accuracy {accuracy:.2f} after a tiny training run"


@pytest.mark.slow
def test_save_load_round_trip(fitted_tcn: TcnModel, pipeline: PipelineRun, tmp_path: Path) -> None:
    fitted_tcn.save(tmp_path / "tcn")
    restored = TcnModel(TINY_CONFIG)
    restored.load(tmp_path / "tcn")
    test_rows = pipeline.by_split["test"]
    assert np.allclose(restored.predict(test_rows), fitted_tcn.predict(test_rows), atol=1e-6)


@pytest.mark.slow
def test_unknown_champions_still_predict(fitted_tcn: TcnModel, pipeline: PipelineRun) -> None:
    """Champion ids never seen in training map to the UNKNOWN embedding, not a crash."""
    test_rows = pipeline.by_split["test"].copy()
    test_rows["blue_mid"] = UNKNOWN_CHAMPION_INDEX + 300  # id far outside the vocabulary
    probabilities = fitted_tcn.predict(test_rows)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    assert len(np.unique(np.round(probabilities, 4))) > 10, "predictions must vary, not collapse"


def test_predict_before_fit_raises(pipeline: PipelineRun) -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        TcnModel(TINY_CONFIG).predict(pipeline.by_split["test"])
