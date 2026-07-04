# DRAKE.ai

**Draft and Real-time Knowledge Evaluator**

A unified machine learning system for League of Legends win probability prediction — from champion select through endgame.

> Status: **Planning / Pre-Development**

---

## What is DRAKE?

DRAKE predicts the probability of winning a League of Legends match at every stage of the game:

- **During draft (T=0):** Evaluate champion compositions, per-pick impact, and partial draft states mid-select
- **During gameplay (T>0):** Track evolving win probability as the game unfolds using gold, objectives, kills, vision, and momentum
- **Swing detection:** Identify game-winning and game-losing moments, attributed to specific player actions
- **Tier-aware:** Predictions conditioned on skill bracket (Iron through Challenger) — the same draft plays differently at different elos

The core innovation is a **unified model** where draft and in-game prediction share the same architecture. Draft sets the initial state; game events update it. One model, one forward pass, continuous P(win) from pick/ban through nexus explosion.

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Primary Model | PyTorch (TCN unified architecture) |
| Baseline Model | LightGBM (GBDT) |
| Calibration | scikit-learn |
| Data Source | Riot Games API (Match-v5, Timeline-v5, League-v4) |
| Data Storage | Parquet files |
| Visualization | Matplotlib / Plotly |

---

## Documentation

| Document | Description |
|----------|-------------|
| [Project Overview](docs/00-PROJECT-OVERVIEW.md) | Mission, build phases, timeline, key decisions |
| [Data Pipeline](docs/01-DATA-PIPELINE.md) | Riot API details, collection strategy, feature engineering |
| [Model Architectures](docs/02-MODEL-ARCHITECTURES.md) | All model designs with parameters, diagrams, training details |
| [Evaluation Plan](docs/03-EVALUATION-PLAN.md) | Metrics, analysis features, visualization plan |
| [Competitive Landscape](docs/04-COMPETITIVE-LANDSCAPE.md) | Existing tools, academic work, differentiation |
| [Handoff Context](docs/05-HANDOFF.md) | Context for future development sessions |

---

## Quick Start

> *Coming soon — project is in planning phase.*

```bash
# Clone and setup
git clone <repo-url>
cd drake
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Configure API key
cp .env.example .env
# Edit .env with your Riot API key

# Collect data
python scripts/collect_seed_players.py --tier GOLD --region na1
python scripts/collect_matches.py --tier GOLD --count 5000

# Train baseline
python scripts/train_gbdt.py --config configs/gbdt_baseline.yaml

# Train TCN
python scripts/train_tcn.py --config configs/tcn_unified.yaml

# Evaluate
python scripts/evaluate.py --models gbdt tcn --output results/
```

---

## Project Structure

```
drake/
├── docs/                   # Project documentation
├── configs/                # Model and pipeline configuration
├── scripts/                # Data collection, training, evaluation
├── drake/                  # Core Python package
│   ├── data/               # Data loading, feature engineering
│   ├── models/             # Model definitions (TCN, GBDT, GRU)
│   ├── training/           # Training loops, loss functions
│   └── evaluation/         # Metrics, calibration, visualization
├── notebooks/              # Exploratory analysis, embedding viz
├── data/                   # Raw and processed data (gitignored)
├── models/                 # Trained model artifacts (gitignored)
└── results/                # Evaluation outputs, plots
```

---

## License

TBD
