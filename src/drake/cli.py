"""The drake CLI: collect -> features -> split -> train -> evaluate.

Every stage takes `--config <run_configurations/file.json>`; the config's
`source` field decides whether collection hits the synthetic generator
(reference mode, no key needed) or the live Riot API (needs RIOT_API_KEY).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    import pandas as pd

from drake import registry
from drake.config import RunConfig
from drake.data.collector import MatchCollector, load_raw_matches
from drake.data.features import FeatureBuilder, load_processed_features
from drake.data.seeding import collect_seed_players
from drake.data.splits import create_splits, split_rows, write_split_summary
from drake.domain import Region, Tier
from drake.evaluation.evaluator import Evaluator


def main(argv: list[str] | None = None) -> int:
    """CLI entry point (wired through [project.scripts])."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    config = RunConfig.from_json(Path(args.config))
    started = time.perf_counter()
    _COMMANDS[args.command](config)
    logger.info("drake {} finished in {:.1f}s", args.command, time.perf_counter() - started)
    return 0


def run_collect(config: RunConfig) -> None:
    """Seed stable anchors and crawl their recent matches into the raw store."""
    api = registry.build_api(config)
    collector = MatchCollector(api, config.collection, config.paths)
    total = 0
    for region_name in config.collection.regions:
        for tier_name in config.collection.tiers:
            region, tier = Region(region_name), Tier(tier_name)
            anchors = collect_seed_players(api, region, tier, config.collection, config.paths.seed_players_dir)
            if anchors.empty:
                logger.warning("No stable anchors found for {} {} — skipping", region_name, tier_name)
                continue
            total += collector.collect(region, tier, anchors)
    logger.info("Collection finished: {} new matches across all regions/tiers", total)


def run_features(config: RunConfig) -> None:
    """Transform collected raw matches into processed training features."""
    raw = load_raw_matches(config.paths.raw_matches_dir)
    builder = FeatureBuilder(config.features)
    processed = builder.build(raw)
    builder.write(processed, config.paths)


def run_split(config: RunConfig) -> None:
    """Assign collected matches to train/val/calibration/test splits."""
    raw = load_raw_matches(config.paths.raw_matches_dir)
    splits = create_splits(raw, config.split, config.paths.splits_dir)
    write_split_summary(splits, config.paths.splits_dir)


def run_train(config: RunConfig) -> None:
    """Train the configured model on the train split and save it."""
    by_split = _load_splits(config)
    model = registry.build_model(config)
    model.fit(by_split["train"], by_split["val"])
    model.save(config.model_dir)


def run_evaluate(config: RunConfig) -> None:
    """Evaluate the saved model on the test split; write metrics, report, and plots."""
    by_split = _load_splits(config)
    model = registry.build_model(config)
    model.load(config.model_dir)
    Evaluator(config.evaluation).evaluate(model, by_split, config.results_dir, config.model)
    print(f"Report: {config.results_dir / 'report.md'}")  # noqa: T201 — the artifact the user asked for


def _load_splits(config: RunConfig) -> dict[str, pd.DataFrame]:
    processed = load_processed_features(config.paths.game_features_dir)
    return split_rows(processed, config.paths.splits_dir)


_COMMANDS = {
    "collect": run_collect,
    "features": run_features,
    "split": run_split,
    "train": run_train,
    "evaluate": run_evaluate,
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="drake",
        description="DRAKE — LoL win probability: collect, engineer features, train, evaluate.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    descriptions = {
        "collect": "Seed stable-anchor players and crawl matches (synthetic or Riot, per config)",
        "features": "Build processed training features from collected raw matches",
        "split": "Create match-level train/val/calibration/test splits",
        "train": "Train the configured model and save it",
        "evaluate": "Evaluate the saved model: metrics, report, and plots",
    }
    for command, description in descriptions.items():
        subparser = subparsers.add_parser(command, help=description)
        subparser.add_argument("--config", required=True, help="Path to a run_configurations/*.json file")
    return parser


if __name__ == "__main__":
    sys.exit(main())
