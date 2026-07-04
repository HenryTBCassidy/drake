# DRAKE — Project Overview

## Mission

Build a unified ML system that predicts League of Legends win probability from champion select through endgame, conditioned on skill tier, with swing detection and per-champion attribution.

**One model, continuous P(win), draft through nexus.**

---

## Core Capabilities

### 1. Draft Prediction (T=0)
- P(win) given full 10-champion draft, side, tier, region, patch
- Per-champion "value add" — marginal contribution of each pick
- Partial draft evaluation — predict mid-champion-select (picks 1 through 10)
- UNKNOWN token masking for remaining picks

### 2. In-Game Prediction (T>0)
- P(win) updated at 30-second intervals using game state
- Features: gold differential, XP, kills, towers, dragons, baron, CS, vision
- Time-window momentum (rate of change over last 2 and 5 minutes)
- Same model, same architecture — draft sets initial state, game events update it

### 3. Timestep Importance Attribution
- **Gradient-based (primary):** Compute ∂P(win_final)/∂features_t to measure how much each timestep influenced the outcome. Native to TCN via PyTorch autograd — no retraining needed
- **Occlusion masking (validation):** Replace timestep t's features with neutral values (interpolated from t-1, t+1), re-run model, measure delta P(win). The timestep analogue of UNKNOWN token masking for champions
- **Swing detection:** Identify timestamps where |delta P(win)| exceeds threshold, attribute to specific game events (kills, objectives, item spikes)
- Classify as "game-winning" or "game-losing" moments
- Per-player swing attribution

### 4. Tier & Region Conditioning
- Predictions conditioned on skill bracket (Iron through Challenger) and server region (NA, EUW, KR, etc.)
- Same champion composition plays differently at different elos and in different regions (KR meta ≠ NA meta)
- Tier embedding (9 tiers x 16 dims) + continuous LP proxy for within-tier granularity
- Region embedding (10 regions x 8 dims)
- Trained on tier-labelled data from stable-anchor player seeding

---

## Build Phases

### Phase 1: Data Pipeline + GBDT Baseline
**Estimated effort:** 12-16 hours (+ wall-clock wait for data crawling)

| Task | Hours | Notes |
|------|-------|-------|
| API client with rate limiting | 2-3 | Match-v5, Timeline-v5, League-v4 |
| Stable-anchor player seeding | 2-3 | Query League-v4, filter stable players |
| Match collection pipeline | 3-4 | Parallel collection, resume support |
| Feature engineering | 2-3 | Draft features, timestep features, momentum |
| GBDT baseline training | 2-3 | LightGBM draft + in-game models |

**Deliverables:**
- Working data collection pipeline
- 50k-200k+ tier-labelled matches (depends on crawl duration)
- GBDT baseline with draft and in-game log loss numbers
- Feature importance analysis

### Phase 2: TCN Unified Model
**Estimated effort:** 10-14 hours

| Task | Hours | Notes |
|------|-------|-------|
| Champion embedding tables | 1-2 | 10 role x side tables, 165+1 x 32 |
| Draft encoder MLP | 1-2 | 362 -> 256 -> 128 |
| TCN architecture | 3-4 | 5 blocks, dilations, residual connections |
| Training loop + loss | 2-3 | Multi-timestep BCE, time weighting |
| Hyperparameter tuning | 2-3 | Learning rate, dropout, channels |

**Deliverables:**
- Trained TCN unified model
- Head-to-head comparison vs GBDT at all timestamps
- Champion embedding vectors for analysis

### Phase 3: Analysis Features
**Estimated effort:** 8-12 hours

| Task | Hours | Notes |
|------|-------|-------|
| Per-champion value add | 2-3 | UNKNOWN token masking, marginal P(win) |
| Timestep importance + swing detection | 2-3 | Gradient attribution, occlusion masking, event correlation |
| Embedding visualization | 1-2 | t-SNE/UMAP, similarity matrices |
| Calibration + reporting | 2-3 | Platt scaling, reliability plots, per-tier |
| Partial draft evaluation | 1-2 | Sequential pick prediction |

**Deliverables:**
- Value add rankings per champion per tier
- Timestep importance maps and swing detection on held-out games
- Embedding analysis (role clusters, synergy patterns)
- Full evaluation report with calibration

---

## Total Effort

| Phase | Hours |
|-------|-------|
| Phase 1: Data Pipeline + GBDT | 12-16 |
| Phase 2: TCN Unified Model | 10-14 |
| Phase 3: Analysis Features | 8-12 |
| **Total** | **30-42** |

**Note:** Data collection requires continuous crawling (wall-clock bottleneck independent of coding hours). Start Phase 1 data collection early and work on other tasks while it runs.

---

## Key Decisions Log

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Primary architecture | TCN (Temporal Convolutional Network) | Parallel training (2-4x faster than RNN), no sequential bottleneck, handles variable-length games well, dilated convolutions capture long-range dependencies |
| Benchmark | LightGBM (GBDT) | Hard to beat on tabular data, approximates AWS-level approach, fast to train, good feature importance |
| Champion encoding | 10 role x side embedding tables (d=32) | Captures "Caitlyn red bot" vs "Caitlyn blue bot" — same champion in different contexts has different value |
| Region encoding | Region embedding (10 regions x 8 dims) | KR, EUW, NA etc. have fundamentally different metas, game pacing, and draft priorities. Region is a first-class input, not a filter |
| Tier + LP encoding | Both: tier embedding (categorical) + LP proxy (continuous) | Tier embedding captures categorical tier-level patterns (different metas/playstyles per elo). LP proxy adds within-tier granularity (Gold 1 99LP ≠ Gold 4 0LP). They encode different information |
| Season timing | Days-since-season-start feature | Early season ranks are noisy (post-reset), late season ranks are stable. Simple continuous feature to capture this |
| Tier labelling | Stable-anchor player seeding | Riot API has no rank in match data. Seeding from stable-rank players gives reliable tier labels by construction |
| Partial drafts | UNKNOWN token (166th embedding entry) | Trained by randomly masking 1-3 champions during training. Enables mid-draft evaluation |
| Curriculum learning | Rejected | Low elo data is noisier, not simpler. Random shuffle with tier conditioning is better |
| Multi-task tier prediction | Rejected | Overcomplicated. Stable-anchor seeding solves tier labelling cleanly without circular dependencies |
| Champ2Vec pre-training | Deferred to post-v1 | End-to-end training is simpler and avoids alignment issues. Pre-training is a nice-to-have for embedding analysis |
| LSTM/Transformer/Mamba | Skipped for v1 | LSTM marginal over GRU with more params. Transformer overkill for ~60 timesteps. Mamba too bleeding-edge (implementation risk) |

---

## Open Questions / Future Work

- **Mamba/SSM:** Revisit after v1 if TCN underperforms expectations. Selective state space models are theoretically compelling for this use case
- **Live game integration:** Real-time prediction during spectated games
- **Web UI:** Dashboard for draft analysis and game review
- **Champ2Vec pre-training:** Train champion embeddings from co-occurrence patterns, compare to end-to-end
- **Patch adaptation:** Fast fine-tuning or patch delta layer when new patches drop
- **Item build integration:** Extend to predict optimal item builds (a la Gary Mialaret / Tolki)
- **Pro play model:** Separate model or fine-tuning for professional/competitive games
- **Duo queue detection:** Infer duo bot lanes from shared match history — duo lanes play very differently from two solos
- **Per-region analysis:** Compare embedding structures and value-add rankings across regions
