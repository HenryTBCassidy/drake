"""Protocol interfaces — the seams between pipeline stages.

Everything outside `drake.registry` depends on these, never on concrete
classes. Implementations inherit explicitly (see STYLE-GUIDE.md § Interfaces).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from numpy.typing import NDArray

    from drake.domain import Division, JsonDict, MatchId, Puuid, Region, Tier


class IRiotApi(Protocol):
    """A source of Riot-shaped data: the real HTTP client or the synthetic generator.

    The collection pipeline is written against this interface only, so the
    identical crawl code runs on synthetic data today and the live API once a
    key exists.
    """

    def get_league_entries(self, region: Region, tier: Tier, division: Division, page: int) -> list[JsonDict]:
        """Return one page of League-v4 ranked entries (empty list = no more pages)."""
        ...

    def get_match_ids(self, region: Region, puuid: Puuid, count: int) -> list[MatchId]:
        """Return a player's most recent ranked-solo match ids, newest first."""
        ...

    def get_match(self, region: Region, match_id: MatchId) -> JsonDict | None:
        """Return the Match-v5 payload, or None if the match is gone (404)."""
        ...

    def get_timeline(self, region: Region, match_id: MatchId) -> JsonDict | None:
        """Return the Timeline-v5 payload, or None if the timeline is gone (404)."""
        ...


class IWinProbabilityModel(Protocol):
    """A trainable P(blue win) predictor over processed feature rows.

    `fit` and `predict` consume rows of the processed Parquet schema
    (docs/01 § Parquet Schema); `predict` returns one probability per row.
    """

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> None:
        """Train on processed feature rows, early-stopping against `val`."""
        ...

    def predict(self, rows: pd.DataFrame) -> NDArray[np.float64]:
        """Return P(blue win) for each processed feature row."""
        ...

    def save(self, directory: Path) -> None:
        """Persist the trained model into `directory`."""
        ...

    def load(self, directory: Path) -> None:
        """Restore a model previously written by `save`."""
        ...
