"""Composition root — the ONLY module that names concrete implementations.

Everything else depends on the protocols in `drake.protocols`; this is where
config strings ("synthetic", "riot", "gbdt", "tcn") become objects.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from dotenv import load_dotenv

from drake.data.rate_limiter import SlidingWindowRateLimiter
from drake.data.riot_client import RiotApiClient
from drake.data.synthetic import SyntheticRiotApi
from drake.models.gbdt import GbdtBaseline

if TYPE_CHECKING:
    from drake.config import RunConfig
    from drake.protocols import IRiotApi, IWinProbabilityModel

RIOT_API_KEY_ENV_VAR = "RIOT_API_KEY"


def build_api(config: RunConfig) -> IRiotApi:
    """Build the configured match source: the synthetic generator or the live client."""
    if config.source == "synthetic":
        return SyntheticRiotApi(config.synthetic)
    if config.source == "riot":
        load_dotenv()
        api_key = os.environ.get(RIOT_API_KEY_ENV_VAR)
        if not api_key:
            raise ValueError(
                f"{RIOT_API_KEY_ENV_VAR} is not set — copy .env.example to .env and add your key "
                "(https://developer.riotgames.com/), or use source='synthetic'"
            )
        limiter = SlidingWindowRateLimiter(
            [
                (config.collection.requests_per_second, 1.0),
                (config.collection.requests_per_two_minutes, 120.0),
            ]
        )
        return RiotApiClient(api_key=api_key, limiter=limiter)
    raise ValueError(f"Unknown source {config.source!r} — expected 'synthetic' or 'riot'")


def build_model(config: RunConfig) -> IWinProbabilityModel:
    """Build the configured win-probability model."""
    if config.model == "gbdt":
        return GbdtBaseline(config.gbdt)
    if config.model == "tcn":
        return _build_tcn(config)
    raise ValueError(f"Unknown model {config.model!r} — expected 'gbdt' or 'tcn'")


def _build_tcn(config: RunConfig) -> IWinProbabilityModel:
    # Imported lazily so the GBDT path never pays the torch import cost.
    from drake.models.tcn import TcnModel  # type: ignore[import-not-found]  # TODO: P12 adds the TCN

    return TcnModel(config.tcn)
