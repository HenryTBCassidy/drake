# DRAKE — AGENTS.md

> Canonical agent instructions. `CLAUDE.md` is a symlink to this file — edit here only.

## What is this project?

DRAKE (Draft and Real-time Knowledge Evaluator) is a unified ML system that predicts
League of Legends win probability from champion select through endgame, conditioned on
skill tier and region, with swing detection and per-champion attribution.

**One model, continuous P(win), draft through nexus.**

The project is **fully specified but not yet built** — there is planning documentation and
this scaffold (tooling, CI, conventions), and essentially no code. Your job is to build it.

## Read this first

The `docs/` set is the source of truth for scope, data, models, and evaluation. Read it
before writing code:

```
docs/
├── 00-PROJECT-OVERVIEW.md      # Mission, build phases, key decisions log
├── 01-DATA-PIPELINE.md         # Riot API, stable-anchor seeding, feature engineering, storage
├── 02-MODEL-ARCHITECTURES.md   # GBDT baseline, TCN (primary), Hybrid, GRU — with param counts
├── 03-EVALUATION-PLAN.md       # Metrics, calibration, analysis features, viz plan
├── 04-COMPETITIVE-LANDSCAPE.md # Prior art and the accuracy bars to beat
├── 05-HANDOFF.md               # Context on Henry, decisions-and-why, things NOT to do
└── guides/
    └── STYLE-GUIDE.md          # Code conventions + tooling contract (reference before writing code)
```

Build order per the overview: **GBDT baseline (A) → TCN unified (B) → Hybrid (C, only if
TCN in-game < GBDT) → GRU (D, optional).**

## You own the architecture

This scaffold deliberately ships **no** `interfaces.py`, `config.py`, `registry.py`, `cli.py`,
or module layout. Those are design decisions, and they are yours to make. What follows is the
design *philosophy* this codebase should inherit — adapt it to DRAKE, don't cargo-cult it.

- **Protocol interfaces with explicit subclassing.** Define the seams (data source, feature
  builder, win-probability model, evaluator) as `typing.Protocol`; implementations inherit
  explicitly. Depend on the protocol, not the concrete class.
- **A single composition root.** One module (a `registry`) is the only place allowed to name
  concrete model/backend classes and wire them together from config. Everything else stays
  decoupled. DRAKE has several models (GBDT, TCN, Hybrid, GRU) over one shared data pipeline
  and one shared evaluation framework — the registry is where "which model" gets resolved.
- **Frozen-dataclass configs + JSON run configs.** Config objects are `@dataclass(frozen=True)`
  with computed properties for derived paths. Run parameters live in JSON files (native JSON
  types — `true`/`false`, not `"True"`), loaded into the dataclasses.
- **Reference-mode-first validation.** You cannot collect real Riot data here (it needs Henry's
  API key and days of crawling) and cannot train on a GPU. So build a **synthetic-match
  generator** that produces plausible fake matches in the real Parquet schema, and drive the
  *entire* pipeline — feature engineering, GBDT + TCN training, evaluation, plots — end-to-end
  on it. This is DRAKE's analogue of AlphaBlokus using TicTacToe to validate the framework
  before the hard game. The payoff: the code is proven correct and CI-green, and the moment a
  real Riot key appears, data flows and models train with **no code changes**.
- **Tests mirror `src/` one-to-one.** `tests/data/test_features.py` ↔ `src/drake/data/features.py`.
- Framework/pipeline code goes under the subpackage matching its phase; small cross-cutting
  modules (config, protocols) stay at the package root.

## Conventions

Follow `docs/guides/STYLE-GUIDE.md` for all code. Highlights:

- Full type annotations on every function signature — machine-enforced (`mypy` strict, in CI).
- `ruff` lint + `ruff format` (line length 120, Python 3.11+). Never hand-fight the formatter.
- Google-style docstrings on public classes/methods.
- `loguru` for logging (no `print()`), `{}` placeholders.
- `from __future__ import annotations`; modern builtin generics (`list[str]`, `int | None`).
- `pathlib.Path` for filesystem, `time.perf_counter()` for timing.
- Parquet for stored data (see docs/01 for the schema); DuckDB is a convenience layer only.

## Commands

```bash
uv sync --extra dev                 # Install package + dev tooling (pytest, ruff, mypy)
uv run pytest                       # Run tests
uv run pytest -m "not slow"         # Skip slow/integration tests (what CI runs)
uv run ruff check . && uv run ruff format --check .   # Lint + format check (matches CI)
uv run mypy                         # Typecheck (strict)
```

Scripts run as modules against the installed package — no `PYTHONPATH` needed. When you add a
console entry point, wire it through `[project.scripts]` in `pyproject.toml`.

All three gates (ruff, mypy, pytest) run in `.github/workflows/ci.yml` on every push/PR. Keep
them green as you build; a red main branch is a bug.

## About the dependencies

`pyproject.toml` ships a starting dependency set drawn from the documented tech stack
(torch, lightgbm, scikit-learn, pandas/pyarrow, httpx, duckdb, loguru, plotly, matplotlib).
Treat it as a starting point — `uv add`/`uv remove` freely and commit the updated `uv.lock`.

## Things NOT to do

Carried over from `docs/05-HANDOFF.md` — these are settled decisions, don't relitigate them:

- Don't add curriculum learning (low-elo-first) — rejected; low elo is noisier, not simpler.
- Don't add multi-task tier prediction — rejected; stable-anchor seeding gives clean labels.
- Don't build LSTM / Transformer / Mamba for v1 — rejected; TCN is primary, GRU optional.
- Don't add Champ2Vec pre-training for v1 — deferred; end-to-end training first.
- Don't scrape third-party sites for historical rank — fragile; forward collection only.
- Don't overcomplicate tier labelling — stable-anchor seeding is the answer.
- Don't reach for a database — this is a batch pipeline; Parquet on disk is the design.
- Don't pad options for learning value. Henry prefers "do the thing that works best" over
  "let's try everything." If one approach is clearly right, build it and say why.
