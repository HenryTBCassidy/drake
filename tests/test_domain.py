"""Tests for drake.domain."""

from __future__ import annotations

import pytest

from drake.domain import Division, Region, Role, Side, Tier, compute_lp_proxy


def test_tier_indices_are_stable() -> None:
    assert Tier.IRON.code == 0
    assert Tier.GOLD.code == 3
    assert Tier.CHALLENGER.code == 8
    assert len(Tier) == 9


def test_region_indices_and_routing() -> None:
    assert Region.NA1.code == 0
    assert Region.NA1.routing == "americas"
    assert Region.EUW1.routing == "europe"
    assert Region.KR.routing == "asia"


def test_role_short_names_cover_all_roles() -> None:
    assert [role.short_name for role in Role] == ["top", "jg", "mid", "adc", "sup"]


def test_side_values_match_riot_team_ids() -> None:
    assert Side.BLUE == 100
    assert Side.RED == 200
    assert Side.RED.short_name == "red"


@pytest.mark.parametrize(
    ("division", "lp", "expected"),
    [
        (Division.IV, 0, 0.0),
        (Division.I, 100, 1.0),
        (Division.II, 50, (2 * 100 + 50) / 400),
    ],
)
def test_lp_proxy_within_divisioned_tier(division: Division, lp: int, expected: float) -> None:
    assert compute_lp_proxy(Tier.GOLD, division, lp) == pytest.approx(expected)


def test_lp_proxy_for_master_plus_uses_raw_lp() -> None:
    assert compute_lp_proxy(Tier.MASTER, None, 500) == pytest.approx(0.5)
    assert compute_lp_proxy(Tier.CHALLENGER, None, 2000) == 1.0, "LP proxy must be clamped to 1"
