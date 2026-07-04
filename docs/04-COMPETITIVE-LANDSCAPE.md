# DRAKE — Competitive Landscape

## Overview

Win probability and draft prediction for League of Legends is an active area with several commercial tools, academic papers, and open-source projects. This document maps the landscape and identifies what DRAKE can learn from and improve upon.

---

## Commercial / Production Tools

### AWS Win Probability (Riot Broadcast)

The official win probability shown during professional League of Legends broadcasts.

- **Model:** XGBoost (gradient-boosted decision trees)
- **Features:** ~13 hand-crafted features (gold diff, tower diff, dragon/baron status, etc.)
- **Scope:** In-game only (no draft prediction), professional play only
- **Accuracy:** Not publicly disclosed, but considered the industry standard
- **Access:** Not queryable — embedded in broadcast production pipeline
- **Limitations:**
  - Pro play only — not applicable to solo queue
  - No draft prediction component
  - No per-champion attribution
  - Relatively few features (room for improvement with richer feature sets)

**What DRAKE learns from AWS:**
- XGBoost/GBDT is a proven baseline for in-game tabular features
- ~13 features is surprisingly effective — feature engineering matters more than model complexity
- Professional credibility comes from calibration, not just accuracy

### LoLDraftAI

Deep learning draft prediction tool focused on champion select.

- **Model:** Neural network with champion embeddings
- **Accuracy:** ~56% on full draft prediction
- **Key features:**
  - Per-champion "value add" — marginal contribution of each pick
  - UNKNOWN token for partial draft evaluation (mid-champion-select)
  - Role-aware embeddings
- **Scope:** Draft only (T=0), no in-game prediction
- **Access:** Web tool

**What DRAKE learns from LoLDraftAI:**
- **UNKNOWN token masking** — adopted directly for partial draft evaluation
- **Per-champion value add** — adopted as a core analysis feature
- ~56% is the accuracy bar to beat for draft prediction
- Champion embeddings are the right approach (vs one-hot)

### iTero

Hybrid draft analysis tool with the highest reported accuracy.

- **Model:** Hybrid linear + GBDT architecture
- **Accuracy:** ~67% reported (draft + early game context)
- **Key features:**
  - Combines draft composition analysis with time-window win rate data
  - Per-champion impact scores
- **Scope:** Draft-focused, some early game integration
- **Access:** Web tool

**What DRAKE learns from iTero:**
- 67% is an aggressive accuracy target (may include early game info, not pure draft)
- Hybrid approaches (combining model types) can outperform single architectures
- Time-window win rate data as a feature (community-level stats about champion performance)

---

## Academic Work

### DraftRec (2023)
- **Architecture:** Transformer-based model for champion recommendation during draft
- **Approach:** Treats draft as a sequential recommendation problem
- **Key insight:** Models the draft as a dialogue between two teams
- **Relevance:** Transformer approach for draft sequence modeling. Interesting but potentially over-engineered for our use case (TCN simpler, draft is only 10 picks).

### LoLytics / FC-Net Approaches (2022)
- **Architecture:** Fully-connected neural networks
- **Approach:** Standard MLP on champion composition features
- **Accuracy:** ~54-56% range
- **Relevance:** Baseline neural approach. Confirms that simple MLPs on draft data achieve ~55%. Our embedding-based approach should improve on this.

### Various Win Prediction Papers
- Multiple academic papers from 2018-2024 tackle LoL win prediction
- Common approaches: logistic regression, random forests, GBDTs, simple neural nets
- In-game accuracy ranges from 70-85% at 15+ minutes depending on features
- Few papers attempt unified draft + in-game models
- Most use match-level features rather than temporal sequences

---

## Notable Individual Projects

### Gary Mialaret / "Tolki" (Item Build Optimization)

- **Focus:** ML-optimized item build paths for League of Legends
- **Career path:** Personal project → Splyce (esports org) → T1 → Fnatic as data analyst
- **Approach:** Trained models to predict optimal item purchases and build orders
- **Relevance:** Proof of concept that LoL ML projects can lead to industry opportunities. Item build optimization is a potential future extension for DRAKE.

### Various GitHub Projects

Open-source LoL prediction models exist in various states:

- **Simple GBDT/RF models:** Most common. Use match-level features. ~55-60% draft accuracy.
- **Embedding-based approaches:** Less common. Some implement Champ2Vec-style pre-training.
- **Temporal models:** Rare. A few attempt LSTM/GRU on timeline data.
- **Production quality:** Generally low. Most are notebooks or scripts, not deployed systems.

---

## Dota 2 Comparisons

Dota 2's more complex draft (bans interleaved with picks, more heroes, stronger counters) has attracted significant ML work:

### OpenDota / STRATZ Models
- Mature win prediction systems with public APIs
- Generally achieve higher draft accuracy than LoL models (~60%+)
- Benefit from Dota 2's stronger draft impact (counter-picks matter more)

### Dota 2 Academic Papers
- More academic attention than LoL due to OpenDota's public data
- Approaches: GBDTs, neural embeddings, graph neural networks on hero interactions
- GNN approaches interesting for modeling champion synergies/counters

**Transferable ideas:**
- Hero/champion interaction graphs (GNN on draft)
- Draft sequence modeling (order of picks matters)
- Rich public APIs enabling larger datasets

---

## DRAKE's Differentiation

### What Nobody Else Does

| Feature | AWS | LoLDraftAI | iTero | DRAKE |
|---------|-----|-----------|-------|-------|
| Draft prediction | No | Yes | Yes | Yes |
| In-game prediction | Yes | No | Limited | Yes |
| **Unified model (draft→in-game)** | No | No | No | **Yes** |
| Per-champion value add | No | Yes | Yes | Yes |
| Partial draft (UNKNOWN) | No | Yes | No | Yes |
| Swing detection + attribution | Basic | No | No | **Yes** |
| Tier conditioning | N/A (pro only) | Unknown | Unknown | **Yes** |
| All ranked tiers | N/A | Yes | Yes | Yes |
| Temporal sequence modeling | No (snapshots) | N/A | N/A | **Yes (TCN)** |
| Champion embeddings | No | Yes | Unknown | **Yes (role x side)** |

### Core Advantages

1. **Unified architecture:** One model from draft through endgame. Draft doesn't just predict win — it initializes the state that tracks the entire game. No other tool does this.

2. **Temporal awareness:** TCN processes the game as a sequence, not independent snapshots. Momentum, trends, and context are built into the architecture.

3. **Tier conditioning:** The same draft has different win probabilities at different skill levels. DRAKE explicitly models this. Most tools either ignore tier or train separate models.

4. **Swing attribution:** Not just "this team is winning" but "this moment changed the game because of this event." Narrative-level analysis.

5. **Embedding richness:** 10 role x side tables capture contextual champion value. "Caitlyn red ADC" is a different entity from "Caitlyn blue mid." This level of granularity is unusual.

---

## Competitive Targets Summary

| Metric | Current Best | DRAKE Target | Stretch |
|--------|-------------|-------------|---------|
| Draft accuracy | ~56% (LoLDraftAI) | 56%+ | 60%+ |
| 10-min accuracy | ~67% (iTero) | 67%+ | 72%+ |
| 20-min accuracy | ~80% (estimated) | 82%+ | 88%+ |
| Calibration (ECE) | Unknown | <0.03 | <0.01 |
| Unified model | Nobody | Yes | With real-time |

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| GBDT beats TCN at everything | Primary model is worse than baseline | Hybrid model (C) captures neural draft + GBDT in-game |
| Riot API rate limits too restrictive | Insufficient training data | Focus on 2-3 tiers first, expand later. Apply for production key. |
| Champion embeddings don't learn meaningful structure | Draft prediction no better than one-hot | Fall back to GBDT draft baseline. Investigate pre-training. |
| Tier labels are noisy despite stable-anchor | Models don't learn tier-specific patterns | Tighten anchor criteria, increase minimum games threshold |
| Model doesn't generalize across patches | Accuracy drops on new patches | Patch delta layer (2-param recalibration per patch) |
| Overfitting on small tiers (Master+) | Poor high-elo predictions | Pool Master/GM/Challenger as "high elo" bucket |
