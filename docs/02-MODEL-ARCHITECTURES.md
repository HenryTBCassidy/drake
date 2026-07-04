# DRAKE — Model Architectures

## Overview

DRAKE evaluates four model variants to find the best approach for unified draft-to-in-game win probability prediction. All share the same data pipeline and evaluation framework.

| Model | Type | Draft | In-Game | Unified | Params | Priority |
|-------|------|-------|---------|---------|--------|----------|
| A — GBDT Baseline | LightGBM | Separate | Separate | No | N/A (trees) | Build first |
| B — TCN Unified | Neural (PyTorch) | Yes | Yes | Yes | ~640k | Primary |
| C — Hybrid | Neural + GBDT | Neural draft | GBDT in-game | Partial | Mixed | If TCN in-game < GBDT |
| D — GRU Unified | Neural (PyTorch) | Yes | Yes | Yes | ~460k | Optional |

---

## Shared Components (Neural Models)

### Champion Embedding Tables

**10 separate embedding tables** — one per role x side combination:

```
blue_top_embedding:  Embedding(166, 32)   # 165 champions + 1 UNKNOWN token
blue_jg_embedding:   Embedding(166, 32)
blue_mid_embedding:  Embedding(166, 32)
blue_adc_embedding:  Embedding(166, 32)
blue_sup_embedding:  Embedding(166, 32)
red_top_embedding:   Embedding(166, 32)
red_jg_embedding:    Embedding(166, 32)
red_mid_embedding:   Embedding(166, 32)
red_adc_embedding:   Embedding(166, 32)
red_sup_embedding:   Embedding(166, 32)
```

**Why 10 tables instead of 1?**
- "Caitlyn blue ADC" and "Caitlyn red ADC" have different values (blue side trap placements, etc.)
- "Caitlyn ADC" and "Caitlyn mid" have very different contexts
- Each table learns context-specific champion representations
- Enables inspecting role+side-specific champion similarity via dot products

**UNKNOWN token (index 165):**
- Used for partial draft evaluation (mid-champion-select)
- During training: randomly mask 1-3 champions per game, replace with UNKNOWN
- During inference: set unpicked champions to UNKNOWN for mid-draft P(win)

**Parameters:** 10 tables x 166 entries x 32 dims = **53,120 parameters**

### Tier & LP Conditioning

```
tier_embedding:  Embedding(9, 16)    # Iron=0, Bronze=1, ..., Challenger=8
lp_proxy:        float               # Normalized 0-1 within tier
```

**Why both tier AND LP?** They capture different information:
- **Tier embedding** learns categorical, tier-level patterns — Gold players have fundamentally different metas, champion pools, and macro decisions than Diamond players. This is a discrete behavioral shift, not a smooth gradient.
- **LP proxy** adds continuous granularity within a tier — Gold 1 99LP is about to promote and plays very differently from Gold 4 0LP. The embedding alone can't capture this.
- We always have both from League-v4 for our anchor players, so there's no missing-data concern.

**Parameters:** 9 x 16 = **144 parameters**

### Region Encoding

```
region_embedding:  Embedding(10, 8)   # NA=0, EUW=1, KR=2, EUNE=3, ...
```

**Why region matters:** Different servers have fundamentally different playstyles:
- KR: faster tempo, more aggressive early, shorter average game duration
- NA: more passive laning, slower objective trading
- EUW: somewhere between KR and NA
- The same draft composition has different win rates across regions

**Parameters:** 10 x 8 = **80 parameters**

### Patch Encoding

```
patch_major_embedding:  Embedding(20, 8)   # Major patch versions
patch_minor_embedding:  Embedding(30, 8)   # Minor patch versions
```

**Parameters:** 20 x 8 + 30 x 8 = **400 parameters**

### Season Progress

```
season_progress:  float    # Days since ranked season start, normalized 0-1
```

Early season ranks are noisy (post-reset climbing), late season ranks are settled. A simple continuous feature.

### Draft Encoder MLP

Shared across all neural models. Takes concatenated embeddings and produces a fixed-size draft vector.

```
Input:  10 champion embeddings (320) + tier (16) + LP (1) + region (8) + patch (16) + season (1) = 362
        ↓
Linear(362, 256) + LayerNorm + ReLU + Dropout(0.2)
        ↓
Linear(256, 128) + LayerNorm + ReLU + Dropout(0.2)
        ↓
Output: draft_vec (128 dims)
```

**Parameters:** 362x256 + 256 + 256x128 + 128 + norms = **~126,000 parameters**

### Win Probability Head

Shared across all timesteps (including T=0 for draft-only prediction).

```
Input:  hidden state (128 dims)
        ↓
Linear(128, 64) + ReLU + Dropout(0.1)
        ↓
Linear(64, 1) + Sigmoid
        ↓
Output: P(win) ∈ [0, 1]
```

**Parameters:** (128x64 + 64) + (64x1 + 1) = **8,321 parameters**

---

## Model A — GBDT Baseline (LightGBM)

### Architecture

Two separate models (no shared architecture):

**Draft Model:**
```
Input: champion_one_hot (1650) + tier_one_hot (9) + region_one_hot (10) + patch (2) + LP (1) + season (1) + side (1)
       = 1674 sparse features
       ↓
LightGBM Binary Classifier
       ↓
Output: P(win|draft)
```

**In-Game Model:**
```
Input: all in-game features (~80) + draft_probability (1)
       = ~81 features per snapshot
       ↓
LightGBM Binary Classifier (one model, all timestamps)
       ↓
Output: P(win|game_state_t)
```

### Hyperparameters (starting point)

```python
lgb_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 127,
    'max_depth': 8,
    'learning_rate': 0.05,
    'n_estimators': 1500,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_samples': 50,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'early_stopping_rounds': 50,
}
```

### Strengths
- Fast to train (<1 hour on 200k matches)
- Excellent on tabular data — hard to beat
- Feature importance built in
- No GPU required
- Approximates AWS-level approach (XGBoost with ~13 features)

### Weaknesses
- Can't learn champion interactions through embeddings
- Draft model uses sparse one-hot (1650 dims) — less expressive
- Two separate models — no unified architecture
- Can't do partial draft evaluation naturally

### Estimated Training Time
- Draft: 5-15 minutes
- In-Game: 15-45 minutes
- Total: ~1 hour including hyperparameter search

---

## Model B — TCN Unified (Primary)

### Full Architecture

```
                         DRAKE TCN Unified Architecture
                         ==============================

INPUTS
──────
Champion IDs (10)              Tier  LP  Region  Patch  Season
[23,64,157,51,89,             [4] [0.7]  [1]   [14,10] [0.6]
 67,238,7,145,412]
    │                          │    │      │      │      │
    ▼                          ▼    ▼      ▼      ▼      ▼
┌──────────────────┐    ┌────────┐ │ ┌────────┐┌──────┐ │
│ 10 Embedding     │    │Tier Emb│ │ │Reg Emb ││Patch │ │
│ Tables           │    │ 9→16   │ │ │ 10→8   ││Emb   │ │
│ 166×32 each      │    └───┬────┘ │ └───┬────┘│20→8  │ │
│                  │        │      │     │     │30→8  │ │
│ Output: 10×32    │        │      │     │     └──┬───┘ │
│ = 320 dims       │        │      │     │        │     │
└────────┬─────────┘        │      │     │        │     │
         │                  │      │     │        │     │
         └──────┬───────────┴──────┴─────┴────────┴─────┘
                │
                ▼
         ┌──────────────┐
         │   CONCAT     │
         │ 320+16+1+8+  │
         │  16+1 = 362  │
         └──────┬───────┘
                │
                ▼
    ┌───────────────────────┐
    │   DRAFT ENCODER MLP   │
    │                       │
    │  Linear(362→256)      │
    │  LayerNorm + ReLU     │
    │  Dropout(0.2)         │
    │                       │
    │  Linear(256→128)      │
    │  LayerNorm + ReLU     │
    │  Dropout(0.2)         │
    │                       │
    │  Output: draft_vec    │
    │          (128 dims)   │
    └───────────┬───────────┘
                │
                │ draft_vec is CONCATENATED to every timestep
                │
                ▼
    ┌───────────────────────────────────────────────┐
    │          TIMESTEP INPUT CONSTRUCTION          │
    │                                               │
    │  T=0: [draft_vec(128) | zeros(~100)]          │
    │  T=1: [draft_vec(128) | game_features_1(~100)]│
    │  T=2: [draft_vec(128) | game_features_2(~100)]│
    │   ...                                         │
    │  T=N: [draft_vec(128) | game_features_N(~100)]│
    │                                               │
    │  Shape: (batch, seq_len, ~228)                │
    │                                               │
    │  NOTE: draft_vec already encodes region,      │
    │  tier, LP, patch, and season information      │
    └───────────────────┬───────────────────────────┘
                        │
                        ▼
    ┌───────────────────────────────────────────────┐
    │              INPUT PROJECTION                 │
    │         Linear(~228 → 128)                    │
    └───────────────────┬───────────────────────────┘
                        │
                        ▼
    ┌───────────────────────────────────────────────┐
    │              TCN BACKBONE                     │
    │                                               │
    │  Block 1: dilation=1,  kernel=3, 128 channels │
    │    CausalConv1d → LayerNorm → ReLU → Dropout  │
    │    CausalConv1d → LayerNorm → ReLU → Dropout  │
    │    + Residual Connection                      │
    │                                               │
    │  Block 2: dilation=2,  kernel=3, 128 channels │
    │    (same structure + residual)                 │
    │                                               │
    │  Block 3: dilation=4,  kernel=3, 128 channels │
    │    (same structure + residual)                 │
    │                                               │
    │  Block 4: dilation=8,  kernel=3, 128 channels │
    │    (same structure + residual)                 │
    │                                               │
    │  Block 5: dilation=16, kernel=3, 128 channels │
    │    (same structure + residual)                 │
    │                                               │
    │  Receptive field:                               │
    │  1 + 2×(3-1)×(1+2+4+8+16) = 125 timesteps    │
    │  = 62 min at 30s res (more than any game)     │
    │                                               │
    │  Output: (batch, seq_len, 128)                │
    └───────────────────┬───────────────────────────┘
                        │
                        ▼
    ┌───────────────────────────────────────────────┐
    │              WIN HEAD                         │
    │    Applied independently at every timestep    │
    │                                               │
    │    Linear(128→64) + ReLU + Dropout(0.1)       │
    │    Linear(64→1) + Sigmoid                     │
    │                                               │
    │    Output: P(win) at each timestep            │
    │    Shape: (batch, seq_len, 1)                 │
    └───────────────────────────────────────────────┘

    T=0         T=1         T=2         ...    T=N
    P(win)=0.52 P(win)=0.55 P(win)=0.48 ...   P(win)=0.91
    (draft)     (1:00)      (1:30)             (end)
```

### TCN Block Detail

Each TCN block consists of two causal dilated convolutions with a residual shortcut:

```
Input x
  │
  ├──────────────────────────────────────┐
  │                                      │ (residual, 1x1 conv if dims differ)
  ▼                                      │
CausalConv1d(128, 128, kernel=3, dil=d)  │
  ▼                                      │
LayerNorm                                │
  ▼                                      │
ReLU                                     │
  ▼                                      │
Dropout(0.2)                             │
  ▼                                      │
CausalConv1d(128, 128, kernel=3, dil=d)  │
  ▼                                      │
LayerNorm                                │
  ▼                                      │
ReLU                                     │
  ▼                                      │
Dropout(0.2)                             │
  ▼                                      │
  + ◄────────────────────────────────────┘
  ▼
Output
```

**Causal padding:** Left-pad input by `(kernel_size - 1) * dilation` so that position t only sees positions <= t. No information leakage from future timesteps.

### Parameter Count Breakdown

| Component | Parameters |
|-----------|-----------|
| 10 Champion Embedding Tables (166 x 32 each) | 53,120 |
| Tier Embedding (9 x 16) | 144 |
| Region Embedding (10 x 8) | 80 |
| Patch Embeddings (20x8 + 30x8) | 400 |
| Draft Encoder MLP | ~126,000 |
| Input Projection Linear(~228, 128) | ~29,300 |
| TCN Block 1 (dilation=1) | ~66,000 |
| TCN Block 2 (dilation=2) | ~66,000 |
| TCN Block 3 (dilation=4) | ~66,000 |
| TCN Block 4 (dilation=8) | ~66,000 |
| TCN Block 5 (dilation=16) | ~66,000 |
| Win Head MLP | ~8,321 |
| LayerNorms (throughout) | ~5,000 |
| **Total** | **~550,000-640,000** |

### Training Configuration

```python
training_config = {
    'optimizer': 'AdamW',
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'scheduler': 'CosineAnnealingLR',
    'warmup_steps': 500,
    'batch_size': 64,            # games per batch
    'max_epochs': 50,
    'early_stopping_patience': 5,
    'gradient_clip_norm': 1.0,
}
```

### Loss Function

Multi-timestep Binary Cross-Entropy with time weighting:

```
L = (1/T) * Σ_t  w(t) * BCE(P_hat(t), y)

where:
  P_hat(t) = model's predicted P(win) at timestep t
  y         = actual outcome (0 or 1, same for all timesteps)
  w(t)      = time weight (options below)
  T         = number of timesteps in the game
```

**Time weighting options (to evaluate):**
1. **Uniform:** w(t) = 1 for all t (simplest)
2. **Draft-boosted:** w(0) = 2.0, w(t>0) = 1.0 (emphasize draft accuracy)
3. **Late-game boosted:** w(t) = 1 + t/T (emphasize late-game when outcome is clearer)
4. **Curriculum:** Start uniform, gradually increase late-game weight

Start with uniform, experiment with draft-boosted if draft accuracy lags GBDT.

### Forward Pass Example

```python
def forward(self, champion_ids, tier, lp, region, patch, season, game_features, seq_lengths):
    """
    champion_ids: (batch, 10) — int64, champion IDs per role slot
    tier: (batch,) — int64, tier index 0-8
    lp: (batch, 1) — float32, normalized LP 0-1
    region: (batch,) — int64, region index 0-9
    patch: (batch, 2) — int64, [major, minor]
    season: (batch, 1) — float32, normalized days since season start
    game_features: (batch, max_seq_len, ~100) — float32, padded
    seq_lengths: (batch,) — int64, actual sequence lengths
    """
    # 1. Embed champions (10 separate tables)
    champ_embeds = []
    for i, table in enumerate(self.champion_embeddings):
        champ_embeds.append(table(champion_ids[:, i]))  # (batch, 32)
    champ_concat = torch.cat(champ_embeds, dim=-1)  # (batch, 320)

    # 2. Embed tier, region, patch
    tier_emb = self.tier_embedding(tier)        # (batch, 16)
    region_emb = self.region_embedding(region)   # (batch, 8)
    patch_emb = torch.cat([
        self.patch_major_emb(patch[:, 0]),      # (batch, 8)
        self.patch_minor_emb(patch[:, 1]),       # (batch, 8)
    ], dim=-1)

    # 3. Draft encoder
    draft_input = torch.cat([
        champ_concat, tier_emb, lp, region_emb, patch_emb, season
    ], dim=-1)  # (batch, 362)
    draft_vec = self.draft_encoder(draft_input)  # (batch, 128)

    # 4. Construct timestep inputs
    # Repeat draft_vec for each timestep
    draft_expanded = draft_vec.unsqueeze(1).expand(-1, max_seq_len, -1)  # (batch, T, 128)
    timestep_input = torch.cat([draft_expanded, game_features], dim=-1)  # (batch, T, ~228)

    # 5. Project to TCN channel size
    x = self.input_projection(timestep_input)  # (batch, T, 128)
    x = x.transpose(1, 2)  # (batch, 128, T) — Conv1d expects channels first

    # 6. TCN backbone
    x = self.tcn(x)  # (batch, 128, T)
    x = x.transpose(1, 2)  # (batch, T, 128)

    # 7. Win head (applied at every timestep)
    p_win = self.win_head(x).squeeze(-1)  # (batch, T)

    return p_win
```

### Why TCN Over Alternatives

| Property | TCN | GRU/LSTM | Transformer |
|----------|-----|----------|-------------|
| Training speed | Fast (parallel) | Slow (sequential) | Fast (parallel) |
| Long-range deps | Dilated convolutions | Vanishing gradient risk | Full attention |
| Param efficiency | Moderate | Good | Heavy |
| Inference | O(1) per new step | O(1) per new step | O(T) recompute |
| Implementation | Simple | Simple | Complex |
| Causal by design | Yes (left padding) | Yes (autoregressive) | Needs causal mask |
| Game-length scale | Great (receptive field > any game) | OK | OK (but quadratic) |

### Estimated Training Time

| Dataset Size | GPU | Estimated Time |
|-------------|-----|----------------|
| 100k matches | RTX 3080 / A100 | 2-4 hours |
| 200k matches | RTX 3080 / A100 | 4-8 hours |
| 500k matches | RTX 3080 / A100 | 8-16 hours |
| 100k matches | CPU only (M1/M2) | 8-16 hours |

---

## Model C — Hybrid (Neural Draft + GBDT In-Game)

### Architecture

Uses the best of both worlds: neural champion embeddings for draft, GBDT for tabular in-game data.

```
                DRAFT (T=0)                    IN-GAME (T>0)
                ─────────                      ──────────────
    Champion embeddings (10 tables)
                │
    Draft Encoder MLP (362→256→128)
                │
    Win Head (128→64→1→sigmoid)          GBDT (LightGBM)
                │                              │
           P(win|draft)  ──────────────> input feature
                                               │
                                     game_features (~80)
                                               │
                                        P(win|game_state)
```

### When to Use
- If TCN in-game prediction underperforms GBDT baseline
- GBDT is notoriously hard to beat on pure tabular data
- But neural embeddings will likely beat one-hot for draft
- This hybrid captures the best of each paradigm

### Training
1. Train the neural draft model (same as Model B's encoder + head)
2. Generate P(win|draft) for all training matches
3. Train GBDT in-game model with P(win|draft) as an input feature
4. At inference: run draft model first, feed output to GBDT

---

## Model D — GRU Unified (Optional)

### Architecture

Same shared components as Model B, but replaces TCN with a GRU recurrent network. The key advantage: draft_vec naturally initializes the hidden state.

```
Draft Encoder → draft_vec (128)
                    │
                    ▼
                h₀ = draft_vec
                    │
    ┌───────────────┼───────────────────────┐
    │               │                       │
    ▼               ▼                       ▼
 ┌──────┐      ┌──────┐               ┌──────┐
 │ GRU  │──h₁──│ GRU  │──h₂── ... ──│ GRU  │──h_N
 │ Cell │      │ Cell │               │ Cell │
 └──┬───┘      └──┬───┘               └──┬───┘
    │             │                       │
    ▼             ▼                       ▼
Win Head      Win Head                Win Head
    │             │                       │
P(win|T=0)  P(win|T=1)              P(win|T=N)
```

### GRU Configuration

```python
gru_config = {
    'input_size': ~100,     # game features per timestep
    'hidden_size': 128,
    'num_layers': 2,
    'dropout': 0.2,
    'bidirectional': False,  # causal — can't look ahead
}
```

### Parameter Count

| Component | Parameters |
|-----------|-----------|
| Shared embeddings + encoder | ~177,664 |
| GRU Layer 1 (input=100, hidden=128) | ~88,320 |
| GRU Layer 2 (input=128, hidden=128) | ~99,072 |
| Win Head | ~8,321 |
| **Total** | **~373,000-460,000** |

### Draft Initialization
- h₀ = draft_vec directly (since hidden_size matches encoder output = 128)
- At T=0 with zero game features: GRU processes draft_vec through one step, win head outputs P(win|draft)
- At T>0: GRU sequentially updates hidden state with game features

### Why Optional
- Sequential processing means ~2-4x slower training than TCN
- Marginal accuracy difference in practice
- Interesting comparison point if we have time
- Natural hidden state initialization is more elegant than TCN's concatenation approach

### Estimated Training Time
- 2-4x slower than TCN due to sequential bottleneck
- 100k matches: ~6-12 hours on GPU

---

## Patch Delta Layer (All Models, Post-Training)

After training on multi-patch data, add a lightweight recalibration layer for new patches:

```
p_patch = sigmoid(a_patch * logit(p_base) + b_patch)

where:
  p_base   = base model output
  a_patch  = learned scale (initialized to 1.0)
  b_patch  = learned bias (initialized to 0.0)
  logit(p) = log(p / (1-p))
```

**Only 2 parameters per patch.** Fine-tune on a small sample (~1000 games) from the new patch while freezing the rest of the model. This handles meta shifts without full retraining.

---

## Model Comparison Summary

| Aspect | GBDT (A) | TCN (B) | Hybrid (C) | GRU (D) |
|--------|----------|---------|------------|---------|
| Draft accuracy | Moderate | Best | Best | Good |
| In-game accuracy | Likely best | Good-Best | Likely best | Good |
| Training time | <1 hr | 4-8 hrs | <2 hrs total | 8-16 hrs |
| Unified model | No | Yes | No | Yes |
| Champion insights | Feature importance | Embedding analysis | Mixed | Embedding analysis |
| Partial draft | No | Yes (UNKNOWN) | Partial | Yes (UNKNOWN) |
| Swing detection | Manual thresholds | Natural (delta P) | Mixed | Natural (delta P) |
| Implementation complexity | Low | Medium | Medium | Medium |
| GPU required | No | Yes (recommended) | Partial | Yes (recommended) |

### Build Order
1. **Model A (GBDT)** — First. Fast to build, establishes baseline.
2. **Model B (TCN)** — Primary. Build immediately after GBDT while data crawls.
3. **Model C (Hybrid)** — Only if TCN in-game < GBDT in-game.
4. **Model D (GRU)** — Only if time permits and we want the comparison.
