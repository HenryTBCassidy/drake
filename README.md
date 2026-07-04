# DRAKE.ai

**Draft and Real-time Knowledge Evaluator**

A unified machine learning system for League of Legends win probability prediction — from champion select through endgame.

> Status: **End-to-end pipeline built** — collection, features, GBDT baseline + TCN, and evaluation all run today on synthetic data; add a Riot API key and the identical code collects real matches.

---

## What is DRAKE?

DRAKE predicts the probability of winning a League of Legends match at every stage of the game:

- **During draft (T=0):** Evaluate champion compositions, per-pick impact, and partial draft states mid-select
- **During gameplay (T>0):** Track evolving win probability as the game unfolds using gold, objectives, kills, vision, and momentum
- **Swing detection:** Identify game-winning and game-losing moments, attributed to specific player actions
- **Tier-aware:** Predictions conditioned on skill bracket (Iron through Challenger) — the same draft plays differently at different elos

The core innovation is a **unified model** where draft and in-game prediction share the same architecture. Draft sets the initial state; game events update it. One model, one forward pass, continuous P(win) from pick/ban through nexus explosion.

---

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11+. On macOS, LightGBM also needs `brew install libomp`.

```bash
uv sync --extra dev        # install the package + dev tooling
uv run pytest -m "not slow"   # verify everything is green
```

### Reference mode — no Riot key needed

The full pipeline runs today against a **synthetic match generator** that emits
Riot-shaped payloads with learnable signal (see `drake/data/synthetic.py`):

```bash
uv run drake collect  --config run_configurations/synthetic_smoke.json   # fake Riot API -> raw Parquet
uv run drake features --config run_configurations/synthetic_smoke.json   # raw -> processed features
uv run drake split    --config run_configurations/synthetic_smoke.json   # train/val/calibration/test
uv run drake train    --config run_configurations/synthetic_smoke.json   # fit the GBDT baseline
uv run drake evaluate --config run_configurations/synthetic_smoke.json   # metrics + report + plots
```

The report lands in `results/synthetic-smoke/gbdt/report.md` with the evaluation
matrix (log loss / Brier / AUC / ECE at draft, 5m, 10m, ...), per-tier slices,
Platt-calibrated variants, and reliability plots.

### Real mode — with a Riot key

```bash
cp .env.example .env       # then paste your key from https://developer.riotgames.com/
```

Create a run config with `"source": "riot"` (and your target regions/tiers under
`collection`), then run the same five commands. Collection is resumable — a SQLite
checkpoint under `data/checkpoints/` means you can stop and restart the crawl freely.

To train the TCN instead of the GBDT, set `"model": "tcn"` in the config. Training
auto-selects CUDA > MPS > CPU; see `docs/guides/REMOTE-TRAINING.md` for the GPU box.

---

## Tech stack

| Component | Technology |
|-----------|------------|
| Primary Model | PyTorch (TCN unified architecture) |
| Baseline Model | LightGBM (GBDT) |
| Calibration | scikit-learn (Platt scaling) |
| Data Source | Riot Games API (Match-v5, Timeline-v5, League-v4) — or the built-in synthetic generator |
| Data Storage | Parquet files (DuckDB optional for exploration) |
| Visualization | Matplotlib / Plotly |

---

## Repository layout

```
drake/
├── docs/                    # Project documentation (see below) + guides + plans
├── run_configurations/      # JSON run configs consumed by the CLI
├── src/drake/
│   ├── domain.py            # Tiers, regions, roles, shared constants
│   ├── config.py            # Frozen-dataclass run configuration
│   ├── protocols.py         # IRiotApi / IWinProbabilityModel seams
│   ├── registry.py          # Composition root — the only module naming concretions
│   ├── cli.py               # drake collect/features/split/train/evaluate
│   ├── data/                # Riot client, rate limiter, synthetic generator,
│   │                        #   seeding, collector, features, splits
│   ├── models/              # GBDT baseline (A), TCN unified (B)
│   ├── training/            # PyTorch training loop for the neural models
│   └── evaluation/          # Metrics, calibration, evaluator, plots
├── tests/                   # Mirrors src/ one-to-one; synthetic fixtures, no mocks
├── data/                    # Collected data (gitignored)
├── models/                  # Trained artifacts (gitignored)
└── results/                 # Evaluation outputs (gitignored)
```

## Documentation

| Document | Description |
|----------|-------------|
| [Project Overview](docs/00-PROJECT-OVERVIEW.md) | Mission, build phases, timeline, key decisions |
| [Data Pipeline](docs/01-DATA-PIPELINE.md) | Riot API details, collection strategy, feature engineering |
| [Model Architectures](docs/02-MODEL-ARCHITECTURES.md) | All model designs with parameters, diagrams, training details |
| [Evaluation Plan](docs/03-EVALUATION-PLAN.md) | Metrics, analysis features, visualization plan |
| [Competitive Landscape](docs/04-COMPETITIVE-LANDSCAPE.md) | Existing tools, academic work, differentiation |
| [Handoff Context](docs/05-HANDOFF.md) | Context for future development sessions |

Development conventions live in `docs/guides/STYLE-GUIDE.md`; active work is planned in `docs/plans/`.

## Development

```bash
uv run ruff check . && uv run ruff format --check .   # lint + format (CI gate)
uv run mypy                                            # strict typing (CI gate)
uv run pytest                                          # full suite incl. slow end-to-end
```

---

## License

TBD
