"""Tests for drake.registry."""

from __future__ import annotations

import pytest

from drake.config import RunConfig
from drake.data.riot_client import RiotApiClient
from drake.data.synthetic import SyntheticRiotApi
from drake.models.gbdt import GbdtBaseline
from drake.registry import RIOT_API_KEY_ENV_VAR, build_api, build_model


def test_synthetic_source_builds_the_generator() -> None:
    api = build_api(RunConfig(source="synthetic"))
    assert isinstance(api, SyntheticRiotApi)


def test_riot_source_requires_an_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(RIOT_API_KEY_ENV_VAR, raising=False)
    monkeypatch.setattr("drake.registry.load_dotenv", lambda: None)
    with pytest.raises(ValueError, match=RIOT_API_KEY_ENV_VAR):
        build_api(RunConfig(source="riot"))


def test_riot_source_builds_the_client_when_key_is_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(RIOT_API_KEY_ENV_VAR, "RGAPI-test")
    api = build_api(RunConfig(source="riot"))
    assert isinstance(api, RiotApiClient)


def test_unknown_source_and_model_fail_loudly() -> None:
    with pytest.raises(ValueError, match="Unknown source"):
        build_api(RunConfig(source="csv"))
    with pytest.raises(ValueError, match="Unknown model"):
        build_model(RunConfig(model="transformer"))


def test_gbdt_model_builds() -> None:
    assert isinstance(build_model(RunConfig(model="gbdt")), GbdtBaseline)
