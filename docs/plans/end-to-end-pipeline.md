# End-to-End Pipeline — Synthetic-First Build

Take DRAKE from planning docs to a working, CI-green, end-to-end system: architecture,
Riot collection pipeline (tested against synthetic API responses — no key yet), a synthetic
match generator driving the full downstream path, feature engineering, the GBDT baseline
(Model A), the evaluation harness, and a CLI. Stretch: the TCN unified model (Model B)
validated on the GPU box. Implements `docs/00`–`03`; conventions per
`docs/guides/STYLE-GUIDE.md`.

**Design stance (the architecture decisions this plan commits to):**

- **Protocol seams:** `IRiotApi` (real client ↔ synthetic generator — the collector cannot
  tell them apart), `IWinProbabilityModel` (GBDT ↔ TCN ↔ later models), defined in
  `drake/protocols.py`. Implementations inherit explicitly.
- **Composition root:** `drake/registry.py` is the only module that names concrete model
  classes and wires them from config. The CLI calls the registry; everything else depends
  on protocols.
- **Config:** frozen dataclasses in `drake/config.py`, loaded from JSON files in
  `run_configurations/`. Derived paths are computed properties on the config.
- **Synthetic-first:** `SyntheticRiotApi` emits Match-v5 / Timeline-v5 / League-v4 shaped
  payloads from a latent team-strength simulator, so the *identical* collection, feature,
  training, and evaluation code runs today and against the real API later with zero changes.
- **Module layout:** pipeline code under `drake/data/`, `drake/models/`, `drake/evaluation/`;
  cross-cutting modules (`domain`, `config`, `protocols`, `registry`, `cli`) at package root.
  Tests mirror one-to-one.

---

## Checklist

| # | Item | Effort | Priority | Done |
|---|------|--------|----------|------|
| P1 | This plan | 30 min | High | ✅ |
| P2 | Core architecture: `domain.py`, `config.py`, `protocols.py`, `.env.example` | 1.5 h | High | ✅ |
| P3 | Rate limiter + Riot API client, tested via httpx MockTransport | 2 h | High | ✅ |
| P4 | Synthetic match generator (`SyntheticRiotApi` + latent-strength simulator) | 2 h | High | ✅ |
| P5 | Stable-anchor seeding + resumable collector (SQLite checkpoints, raw Parquet) | 2 h | High | ✅ |
| P6 | Feature engineering: raw → processed Parquet (draft, in-game, momentum, 30s resample) | 2 h | High | ✅ |
| P7 | Match-level time-based train/val/test splits | 45 min | High | ✅ |
| P8 | GBDT baseline (Model A): draft + in-game LightGBM behind the model protocol | 1.5 h | High | ✅ |
| P9 | Evaluation harness: metrics, per-timestamp matrix, reliability plot, report | 2 h | High | ✅ |
| P10 | Registry + CLI (`drake collect/features/split/train/evaluate`) + run configs | 1.5 h | High | ✅ |
| P11 | End-to-end slow test + README quickstart rewrite | 1 h | High | ✅ |
| P12 | TCN unified model (Model B) + training loop, tiny-config train on synthetic | 3 h | Medium | |
| P13 | Validate TCN training on the GPU box (CUDA), per REMOTE-TRAINING.md | 1 h | Medium | |
| P14 | Per-tier evaluation breakdown + Platt calibration | 1.5 h | Low | ✅ |

---

## P2. Core architecture

`drake/domain.py`: `Tier`, `Region`, `Role`, `Side` StrEnums with stable integer encodings
(`Tier.code`, etc. — the int8 codes in the Parquet schema), champion-count constants
(`NUM_CHAMPIONS = 165`, `UNKNOWN_CHAMPION_INDEX = 165`), queue id, and shared type aliases.

`drake/config.py`: frozen dataclasses — `PathsConfig` (data/models/results roots with
computed stage subpaths), `CollectionConfig`, `SyntheticConfig`, `FeatureConfig`,
`SplitConfig`, `GbdtConfig`, `TcnConfig`, `EvaluationConfig`, composed into `RunConfig`;
`RunConfig.from_json(path)` loads native-JSON run files. Defaults mirror the documented
hyperparameters (docs/02).

`drake/protocols.py`: `IRiotApi` (league entries, match ids, match, timeline) and
`IWinProbabilityModel` (fit / predict / save / load). `.env.example` documents `RIOT_API_KEY`.

## P3. Rate limiter + Riot API client

`drake/data/rate_limiter.py`: sliding-window limiter enforcing both dev-key windows
(20 req/s and 100 req/2 min) with injectable clock/sleep for fast tests.
`drake/data/riot_client.py`: `RiotApiClient(IRiotApi)` over httpx — regional vs platform
routing, retry with exponential backoff on 429 (honouring `Retry-After`) and 5xx, `None` on
404, API key from env. Tests use `httpx.MockTransport` with recorded-shape responses — the
live API is never called.

## P4. Synthetic match generator

`drake/data/synthetic.py`: `SyntheticRiotApi(IRiotApi)` — deterministic (seeded) generator
of League-v4 entry pages (a controllable fraction are stable anchors), per-player match id
lists, and Match-v5/Timeline-v5 payloads. Behind it, a latent-strength simulator: each draft
gets a hidden strength differential (champion/role effects + noise); the winner is sampled
from it, and gold/xp/kill/objective trajectories drift accordingly. This puts *learnable
signal* in the fake data so models produce real (non-trivial) metrics. Payloads carry the
exact fields the pipeline consumes, in Riot's shapes.

## P5. Seeding + resumable collector

`drake/data/seeding.py`: `is_stable_anchor` per docs/01 (veteran, not freshBlood, active,
≥150 games, 47–53% WR) + `collect_seed_players` writing per-tier anchor Parquet.
`drake/data/collector.py`: crawl anchors → recent match ids → match + timeline, labelled
with the anchor's tier/LP; SQLite checkpoint DB tracks seeded tiers and per-match download
state so restarts skip completed work; dedup by match id; quality filters (queue 420,
duration > 300 s); raw rows (JSON payload columns + key metadata) written as chunked part files under
`raw/matches/{region}/{tier}/` — a simplification of the docs/01 per-patch filenames;
patch stays available as a column. Works identically against `RiotApiClient` or `SyntheticRiotApi`.

## P6. Feature engineering

`drake/data/features.py`: `FeatureBuilder` parses raw match+timeline Parquet and emits the
processed schema from docs/01 — one row per `(match_id, timestep)`: identity, context
(tier/LP/region/patch/season), draft (10 champion id columns), in-game diffs (gold/xp/cs
total and per-lane, K/D/A, towers, dragons + soul, barons + active buff, herald, inhibitors,
vision, level, plates, normalized game time), momentum deltas (2 min / 5 min windows), and
label. Participant frames resampled 60 s → 30 s by linear interpolation; discrete events
assigned to their actual timestamps. T=0 is the draft row with zeroed game state.
Writes per-tier Parquet + `metadata.json` (feature schema, counts).

## P7. Splits

`drake/data/splits.py`: split by match id, never by row; newest ~10% of matches (by
`gameCreation`) become the time-based test set; the rest split 80/10/10
train/val/calibration by seeded shuffle. Written as id-list text files per docs/01 layout.

## P8. GBDT baseline (Model A)

`drake/models/gbdt.py`: `GbdtBaseline(IWinProbabilityModel)` wrapping two LightGBM binary
classifiers — draft (T=0 rows) and in-game (T>0 rows, with the draft model's P(win) as an
input feature), hyperparameters from docs/02 via `GbdtConfig`. Early stopping on the val
set. Save/load via LightGBM's native text format + a small JSON manifest.

> **Deviation from docs/01 (deliberate):** champion/tier/region inputs use LightGBM's native
> categorical-feature handling instead of a 1,674-wide one-hot expansion — same information,
> far cheaper, one line of code. Revisit only if per-slot one-hot measurably helps.

## P9. Evaluation harness

`drake/evaluation/metrics.py`: log loss, Brier, AUC, ECE (binned), accuracy — pure numpy.
`drake/evaluation/evaluator.py`: evaluates any `IWinProbabilityModel` on the test split at
the docs/03 timestamps (T=0, 5, 10, 15, 20, 25, 30 min), emits a metrics Parquet + a
markdown report with the evaluation matrix. `drake/evaluation/plots.py`: reliability
diagram and metric-vs-timestamp line plots (matplotlib, saved to results dir).

## P10. Registry + CLI

`drake/registry.py`: builds the configured `IRiotApi` (synthetic vs riot) and
`IWinProbabilityModel` (gbdt / tcn) — the only module naming concretions.
`drake/cli.py`: argparse CLI wired through `[project.scripts]` — `drake collect` (synthetic
or riot, per config.source), `drake features`, `drake split`, `drake train`, `drake evaluate`, each taking
`--config run_configurations/<file>.json`. Ships a default `synthetic_smoke.json`.

## P11. End-to-end slow test + README

`tests/test_end_to_end.py` (`@pytest.mark.slow`): synth → features → split → train GBDT →
evaluate on a small config in a tmp dir; asserts artifacts exist and test-set AUC beats 0.5
(the synthetic signal is learnable). README quickstart rewritten around `uv` + the real CLI.

## P12. TCN unified model (Model B)

`drake/models/tcn.py`: the docs/02 architecture — 10 role×side embedding tables (166×32,
UNKNOWN at 165), tier/region/patch embeddings + LP/season scalars, draft encoder MLP
(362→256→128), draft_vec concatenated to every timestep, input projection, 5 causal dilated
residual blocks (d=1..16, 128 ch), shared win head. `drake/training/trainer.py`: AdamW +
cosine schedule, multi-timestep BCE with uniform time weighting and padding masks, random
UNKNOWN masking of 1–3 champions, early stopping, device auto-select (cuda/mps/cpu).
`TcnModel(IWinProbabilityModel)` wraps net + trainer so the registry/CLI/evaluator treat it
exactly like the GBDT.

## P13. GPU validation

Push branch, clone/sync on the box per REMOTE-TRAINING.md, run a short synthetic TCN
training inside tmux on CUDA; confirm `torch.cuda.is_available()` and that loss decreases.
Record the actual result here when done.

## P14. Per-tier evaluation + calibration

Extend the evaluator with a per-tier metrics breakdown and optional Platt scaling
(fit on the calibration split, report raw vs calibrated ECE side by side).

> Landed as part of P9 — the evaluator was designed with per-tier slices and Platt
> calibration from the start, so a separate pass wasn't needed.
