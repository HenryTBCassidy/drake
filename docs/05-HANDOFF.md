# DRAKE — Handoff Context

> This document provides context for future Claude instances (or other AI assistants) continuing work on this project. Read this first.

---

## What Is This Project?

DRAKE (Draft and Real-time Knowledge Evaluator) is a League of Legends win probability prediction system. It uses a unified ML model to predict P(win) from champion select through endgame, conditioned on skill tier.

**One model, continuous P(win), draft through nexus.**

---

## Who Is Henry?

- **Current role:** Data scientist at Transak, specialising in trading and pricing
- **Previous experience:** Quantitative researcher at G-Research (London-based quant fund)
- **Technical skills:** Strong Python, software engineering, data analysis, statistics. Conceptual understanding of ML — has built an AlphaZero tic-tac-toe implementation and a mini Shakespeare GPT (character-level transformer)
- **ML training experience:** Limited hands-on experience training production ML models. This is his first major ML project beyond tutorials
- **Tools:** Has unlimited Claude and Codex access. Plans to use AI assistance heavily for implementation
- **Communication style:** Results-oriented. Dislikes overcomplication. Will push back if you're being too tutorial-like or adding unnecessary complexity. Prefers "do the thing that works best" over "let's try everything for learning"
- **League knowledge:** Mid-Emerald ADC player on EUW. Understands the game domain well from a player's perspective

---

## Current State

### What's Been Done
- [x] Full project planning and architecture design
- [x] Model architecture decisions (TCN primary, GBDT baseline)
- [x] Data pipeline design (stable-anchor seeding, Riot API strategy)
- [x] Feature engineering specification
- [x] Evaluation framework design
- [x] Competitive landscape research
- [x] Project documentation (this folder)

### What Has NOT Been Done
- [ ] Any code written
- [ ] Any data collected
- [ ] Any models trained
- [ ] Project environment setup (venv, requirements.txt, etc.)
- [ ] Riot API key obtained
- [ ] Git repository initialized

### Immediate Next Steps
1. Initialize git repo, set up Python environment
2. Get a Riot API development key (https://developer.riotgames.com/)
3. Build the data collection pipeline (API client, rate limiter, stable-anchor seeding)
4. Start data crawling (wall-clock bottleneck — start early)
5. While data collects: build GBDT baseline on initial data
6. Build TCN unified model
7. Evaluate and compare

---

## Key Architecture Decisions (and Why)

### Why TCN over GRU/LSTM/Transformer?
- **Parallel training:** 2-4x faster than RNN (processes all timesteps simultaneously)
- **No sequential bottleneck:** Each training batch processes complete games in parallel
- **Dilated convolutions:** Receptive field covers any game length without vanishing gradients
- **Simplicity:** Easier to implement correctly than attention mechanisms
- **Results:** Empirically matches or beats RNNs on most sequence tasks
- Rejected LSTM (marginal over GRU, more params), Transformer (overkill for ~60 timesteps), Mamba (too bleeding-edge, implementation risk)

### Why 10 embedding tables (not 1)?
- Same champion plays differently in different roles and on different sides
- "Caitlyn red bot" has different trap placement patterns than "Caitlyn blue bot"
- 10 separate tables (role x side) learn these contextual differences
- Enables inspecting "what does Caitlyn mean in the red ADC context specifically?"
- Only costs 53k params total — not expensive

### Why stable-anchor player seeding?
- Riot API Match-v5 returns NO rank/tier information in match data
- Only League-v4 gives current rank, and only current (no historical)
- Can't project current rank backwards reliably (players climb/demote)
- Solution: collect matches ONLY from players whose rank is demonstrably stable
  - veteran=True, freshBlood=False, ~50% WR, 150+ games
  - Their current rank IS their actual rank during recent games (by definition)
- Every match in our dataset has at least one stable-anchor player → reliable tier label

### Why not curriculum learning (train low elo first)?
- Low elo games are NOISIER, not simpler
- More throws, random outcomes, less adherence to win conditions
- The model doesn't benefit from learning patterns that don't exist
- Random shuffle with tier as an input feature is simpler and works

### Why not multi-task tier prediction?
- Originally considered predicting tier as auxiliary task alongside win probability
- Problem: circular dependency for draft prediction — tier heavily affects draft win%, and we'd be predicting both
- Stable-anchor seeding already gives us clean tier labels without needing to predict them
- Simpler is better

### Why region as a model input?
- KR, EUW, NA etc. have fundamentally different metas, game pacing, and draft priorities
- The same draft composition has different win rates across regions
- Region embedding (10 regions x 8 dims) is cheap and captures this cleanly
- Alternative was training separate per-region models — wasteful, less data per model

### Why both tier AND LP?
- Tier embedding captures categorical, tier-level patterns — Gold meta ≠ Diamond meta, and this is a discrete behavioral shift
- LP proxy adds continuous within-tier granularity — Gold 1 99LP ≠ Gold 4 0LP
- We always have both from League-v4 for our anchor players
- They encode different information and are not redundant

### Why not Champ2Vec pre-training (for v1)?
- End-to-end training is simpler and avoids alignment issues
- Pre-trained embeddings might not align with the downstream task
- Can always add pre-training later as an experiment
- Focus on getting the full pipeline working first

---

## Architecture Quick Reference

### TCN Unified Model (~640k params)

```
Champion IDs (10) → 10 Embedding Tables (166x32 each)
                         ↓
              Concat (320) + Tier (16) + LP (1) + Region (8) + Patch (16) + Season (1) = 362
                         ↓
              Draft Encoder MLP: 362 → 256 → 128 = draft_vec
                         ↓
              Concat draft_vec (128) to every timestep's game features (~100)
                         ↓
              Input Projection: ~228 → 128
                         ↓
              5 TCN Blocks (dilations 1,2,4,8,16, 128 channels, kernel 3)
                         ↓
              Win Head: 128 → 64 → 1 → sigmoid  (applied at every timestep)
                         ↓
              P(win) at T=0 (draft), T=1 (0:30), T=2 (1:00), ...
```

### GBDT Baseline (LightGBM)

```
Draft:    one-hot champions (1650) + tier + LP + region + patch + season + side → LightGBM → P(win|draft)
In-Game:  game features (~80) + P(win|draft) → LightGBM → P(win|game_state_t)
```

### Key Numbers
- Champion embedding dims: 32
- Tier embedding dims: 16
- Region embedding dims: 8
- Patch embedding dims: 8 (major) + 8 (minor)
- Draft encoder input: 362 dims
- Draft vector dims: 128
- TCN channels: 128
- TCN blocks: 5
- TCN kernel size: 3
- TCN dilations: [1, 2, 4, 8, 16]
- Timestep resolution: 30 seconds
- UNKNOWN token: index 165 in each embedding table
- Total champions: 165 (+ 1 UNKNOWN = 166 entries per table)

---

## Data Pipeline Quick Reference

### APIs Used
- **Match-v5:** Match details (post-game) — `/lol/match/v5/matches/{matchId}`
- **Timeline-v5:** Minute-by-minute game state — `/lol/match/v5/matches/{matchId}/timeline`
- **League-v4:** Current rank data — `/lol/league/v4/entries/by-summoner/{summonerId}`

### Rate Limits (Dev Key)
- 20 requests/second, 100 requests/2 minutes
- ~240 enriched matches/hour
- ~5,760 matches/day continuous

### Data Retention
- Match metadata: ~2 years
- Timeline data: ~1 year (rolling)

### Collection Strategy
1. Query League-v4 for stable-anchor players per tier
2. Get their recent match IDs (last 20 games, ranked solo queue)
3. Fetch match details + timeline for each
4. Deduplicate, filter (no remakes, no AFKs), store as Parquet

---

## Evaluation Quick Reference

### Metrics
- **Log Loss** (primary) — probability accuracy
- **Brier Score** — decomposable accuracy
- **AUC-ROC** — discrimination ability
- **ECE** — calibration quality

### Targets
- Draft log loss < 0.69 (beat coin flip)
- Draft accuracy > 56% (match LoLDraftAI)
- 15-min log loss < 0.55
- ECE < 0.03 after calibration

### Evaluation Matrix
- Models (GBDT, TCN, Hybrid, optional GRU) x Timestamps (draft, 5m, 10m, 15m, 20m, 25m, 30m) x Tiers (Iron through Challenger) x Regions (NA, EUW, KR, etc.)

### Timestep Importance Attribution
- **Gradient attribution (primary):** ∂P(win_final)/∂features_t — fast, native to PyTorch
- **Occlusion masking (validation):** replace timestep features with interpolated neutral values, measure delta
- Different from swing detection: swings = what changed prediction between steps; importance = how much each step influenced the final outcome

---

## File Layout

```
drake/
├── docs/                   # You are here — project documentation
│   ├── 00-PROJECT-OVERVIEW.md
│   ├── 01-DATA-PIPELINE.md
│   ├── 02-MODEL-ARCHITECTURES.md
│   ├── 03-EVALUATION-PLAN.md
│   ├── 04-COMPETITIVE-LANDSCAPE.md
│   └── 05-HANDOFF.md      # This file
├── configs/                # Model and pipeline configuration (TBD)
├── scripts/                # Data collection, training, evaluation (TBD)
├── drake/                  # Core Python package (TBD)
│   ├── data/               # Data loading, feature engineering
│   ├── models/             # Model definitions (TCN, GBDT, GRU)
│   ├── training/           # Training loops, loss functions
│   └── evaluation/         # Metrics, calibration, visualization
├── notebooks/              # Exploratory analysis (TBD)
├── data/                   # Raw and processed data (gitignored)
├── models/                 # Trained artifacts (gitignored)
├── results/                # Evaluation outputs (TBD)
├── README.md
└── .gitignore
```

---

## Gotchas and Lessons Learned

*(To be updated as development progresses)*

1. **Riot API has no rank in match data.** This is the single most annoying data limitation. Match-v5 used to have `highestAchievedSeasonTier` in the old v4 API, but it was removed. Stable-anchor seeding is the cleanest workaround.

2. **Timeline data is only ~1 year.** Don't plan on collecting historical data going back further than that. Collect forward.

3. **20 extra API calls per match for rank enrichment.** Each match has 10 players, each needing a League-v4 lookup. This is the throughput bottleneck.

4. **Spectator-v5 has been deactivated by Riot.** Don't waste time trying to use it for live game data.

5. **No third-party historical rank APIs exist.** Sites like OP.GG build their historical rank graphs from periodic polling. You can't query them for historical data.

6. **Henry prefers simplicity.** If you find yourself proposing three alternatives where one will clearly work, just recommend the one that works. Don't pad options for educational value.

---

## Things NOT to Do

- Don't suggest curriculum learning (training on low elo first) — already rejected
- Don't suggest multi-task tier prediction — already rejected
- Don't suggest building LSTM, Transformer, or Mamba for v1 — already rejected
- Don't suggest Champ2Vec pre-training for v1 — deferred to post-v1
- Don't suggest web scraping for historical rank data — too fragile, forward collection is better
- Don't overcomplicate the tier labelling — stable-anchor seeding is the answer
- Don't build a learning path — build the best model first

---

## Useful Links

- Riot Developer Portal: https://developer.riotgames.com/
- Match-v5 Docs: https://developer.riotgames.com/apis#match-v5
- League-v4 Docs: https://developer.riotgames.com/apis#league-v4
- LightGBM Docs: https://lightgbm.readthedocs.io/
- PyTorch TCN reference: https://github.com/locuslab/TCN
