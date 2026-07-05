"""PyTorch training loop for the TCN (docs/02 § Training Configuration).

Multi-timestep BCE with uniform time weighting over padded, masked sequences;
AdamW + cosine annealing; UNKNOWN-token draft masking as augmentation; early
stopping on validation loss. Device auto-selection prefers CUDA, then Apple
MPS, then CPU.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from loguru import logger

from drake.domain import UNKNOWN_CHAMPION_INDEX

if TYPE_CHECKING:
    from collections.abc import Sequence

    from drake.config import TcnConfig
    from drake.models.tcn import MatchTensors, TcnNet

_MAX_MASKED_CHAMPIONS = 3


def select_device(requested: str) -> torch.device:
    """Resolve the configured device string; "auto" prefers cuda > mps > cpu."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass(frozen=True)
class MatchBatch:
    """Padded batch of match sequences, ready for TcnNet.forward."""

    champions: torch.Tensor  # (B, 10) int64
    tier: torch.Tensor  # (B,) int64
    lp_proxy: torch.Tensor  # (B, 1) float32
    region: torch.Tensor  # (B,) int64
    patch_major: torch.Tensor  # (B,) int64
    patch_minor: torch.Tensor  # (B,) int64
    season_progress: torch.Tensor  # (B, 1) float32
    game_features: torch.Tensor  # (B, T_max, F) float32, zero-padded
    labels: torch.Tensor  # (B,) float32
    timestep_mask: torch.Tensor  # (B, T_max) bool — True on real timesteps

    @staticmethod
    def collate(matches: Sequence[MatchTensors], device: torch.device) -> MatchBatch:
        max_seq_len = max(match.seq_len for match in matches)
        num_features = matches[0].game_features.shape[1]
        game_features = torch.zeros(len(matches), max_seq_len, num_features, dtype=torch.float32)
        timestep_mask = torch.zeros(len(matches), max_seq_len, dtype=torch.bool)
        for row, match in enumerate(matches):
            game_features[row, : match.seq_len] = match.game_features
            timestep_mask[row, : match.seq_len] = True
        return MatchBatch(
            champions=torch.stack([match.champions for match in matches]).to(device),
            tier=torch.tensor([match.tier for match in matches], dtype=torch.int64, device=device),
            lp_proxy=torch.tensor([[match.lp_proxy] for match in matches], dtype=torch.float32, device=device),
            region=torch.tensor([match.region for match in matches], dtype=torch.int64, device=device),
            patch_major=torch.tensor([match.patch_major for match in matches], dtype=torch.int64, device=device),
            patch_minor=torch.tensor([match.patch_minor for match in matches], dtype=torch.int64, device=device),
            season_progress=torch.tensor(
                [[match.season_progress] for match in matches], dtype=torch.float32, device=device
            ),
            game_features=game_features.to(device),
            labels=torch.tensor([match.label for match in matches], dtype=torch.float32, device=device),
            timestep_mask=timestep_mask.to(device),
        )


class TcnTrainer:
    """Runs the docs/02 training recipe over per-match sequence tensors."""

    def __init__(self, config: TcnConfig) -> None:
        self._config = config
        self._device = select_device(config.device)

    def train(self, net: TcnNet, train_matches: list[MatchTensors], val_matches: list[MatchTensors]) -> None:
        """Train in place, restoring the best-validation weights before returning."""
        config = self._config
        logger.info(
            "Training TCN on {} ({} train / {} val matches, {} params)",
            self._device,
            len(train_matches),
            len(val_matches),
            sum(parameter.numel() for parameter in net.parameters()),
        )
        net.to(self._device)
        optimizer = torch.optim.AdamW(net.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_epochs)
        rng = np.random.default_rng(0)

        best_val_loss = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        epochs_without_improvement = 0
        for epoch in range(1, config.max_epochs + 1):
            epoch_started = time.perf_counter()
            train_loss = self._run_epoch(net, train_matches, optimizer, rng)
            val_loss = self._validation_loss(net, val_matches)
            scheduler.step()
            logger.info(
                "Epoch {}: train loss {:.4f}, val loss {:.4f} ({:.1f}s)",
                epoch,
                train_loss,
                val_loss,
                time.perf_counter() - epoch_started,
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {name: tensor.detach().clone() for name, tensor in net.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= config.early_stopping_patience:
                    logger.info("Early stopping after epoch {} (best val loss {:.4f})", epoch, best_val_loss)
                    break
        if best_state is not None:
            net.load_state_dict(best_state)

    def _run_epoch(
        self,
        net: TcnNet,
        matches: list[MatchTensors],
        optimizer: torch.optim.Optimizer,
        rng: np.random.Generator,
    ) -> float:
        net.train()
        order = rng.permutation(len(matches))
        total_loss = 0.0
        num_batches = 0
        for start in range(0, len(order), self._config.batch_size):
            batch_matches = [matches[i] for i in order[start : start + self._config.batch_size]]
            batch = MatchBatch.collate(batch_matches, self._device)
            batch = self._mask_random_champions(batch, rng)
            optimizer.zero_grad()
            loss = _masked_bce(net(batch), batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), self._config.gradient_clip_norm)
            optimizer.step()
            total_loss += float(loss.item())
            num_batches += 1
        return total_loss / max(num_batches, 1)

    def _validation_loss(self, net: TcnNet, matches: list[MatchTensors]) -> float:
        net.eval()
        total_loss = 0.0
        num_batches = 0
        with torch.no_grad():
            for start in range(0, len(matches), self._config.batch_size):
                batch = MatchBatch.collate(matches[start : start + self._config.batch_size], self._device)
                total_loss += float(_masked_bce(net(batch), batch).item())
                num_batches += 1
        return total_loss / max(num_batches, 1)

    def _mask_random_champions(self, batch: MatchBatch, rng: np.random.Generator) -> MatchBatch:
        """UNKNOWN-token augmentation: mask 1-3 champion slots on some games (docs/02)."""
        champions = batch.champions.clone()
        for row in range(champions.shape[0]):
            if rng.random() >= self._config.unknown_mask_probability:
                continue
            num_masked = int(rng.integers(1, _MAX_MASKED_CHAMPIONS + 1))
            slots = rng.choice(champions.shape[1], size=num_masked, replace=False)
            champions[row, torch.tensor(slots, dtype=torch.int64)] = UNKNOWN_CHAMPION_INDEX
        return MatchBatch(
            champions=champions,
            tier=batch.tier,
            lp_proxy=batch.lp_proxy,
            region=batch.region,
            patch_major=batch.patch_major,
            patch_minor=batch.patch_minor,
            season_progress=batch.season_progress,
            game_features=batch.game_features,
            labels=batch.labels,
            timestep_mask=batch.timestep_mask,
        )


def _masked_bce(logits: torch.Tensor, batch: MatchBatch) -> torch.Tensor:
    """Uniform-weighted multi-timestep BCE, averaged over real timesteps then games."""
    targets = batch.labels.unsqueeze(1).expand_as(logits)
    per_timestep = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    mask = batch.timestep_mask.float()
    per_game = (per_timestep * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    return per_game.mean()
