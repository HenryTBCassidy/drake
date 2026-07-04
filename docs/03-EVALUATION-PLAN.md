# DRAKE — Evaluation Plan

## Overview

Evaluation covers four dimensions:
1. **Prediction accuracy** — How well does each model predict win probability?
2. **Calibration** — When the model says 70%, do teams win ~70% of the time?
3. **Analysis features** — Per-champion value, swing detection, embedding quality
4. **Cross-model comparison** — Which model wins at which timestamps?

---

## Primary Metrics

### Log Loss (Primary Metric)
```
LogLoss = -(1/N) * Σ [y * log(p) + (1-y) * log(1-p)]
```
- Lower is better
- Penalizes confident wrong predictions heavily
- Standard for probability estimation tasks
- **Target:** <0.68 for draft, <0.50 for late game (15+ min)

### Brier Score
```
BrierScore = (1/N) * Σ (p - y)²
```
- Lower is better (0 = perfect)
- Decomposes into calibration + resolution + uncertainty
- More interpretable than log loss
- **Target:** <0.24 for draft, <0.15 for late game

### AUC-ROC
- Discrimination ability (can the model distinguish wins from losses?)
- Less sensitive to calibration than log loss
- **Target:** >0.58 for draft, >0.80 for late game

### Expected Calibration Error (ECE)
```
ECE = Σ_b (n_b / N) * |accuracy_b - confidence_b|
```
- Bin predictions into 10-20 buckets by predicted probability
- Compare average prediction vs actual win rate in each bucket
- Lower is better (0 = perfectly calibrated)
- **Target:** <0.02 after calibration

---

## Evaluation Matrix

Every model is evaluated at multiple timestamps to compare draft vs in-game performance:

| Timestamp | Description | GBDT (A) | TCN (B) | Hybrid (C) | GRU (D) |
|-----------|-------------|----------|---------|------------|---------|
| T=0 (draft) | Champion select only | LogLoss | LogLoss | LogLoss | LogLoss |
| T=5m | Early laning phase | LogLoss | LogLoss | LogLoss | LogLoss |
| T=10m | Mid laning | LogLoss | LogLoss | LogLoss | LogLoss |
| T=15m | Transition to mid-game | LogLoss | LogLoss | LogLoss | LogLoss |
| T=20m | Mid-game team fights | LogLoss | LogLoss | LogLoss | LogLoss |
| T=25m | Late mid-game | LogLoss | LogLoss | LogLoss | LogLoss |
| T=30m | Late game | LogLoss | LogLoss | LogLoss | LogLoss |
| T=35m+ | Very late game | LogLoss | LogLoss | LogLoss | LogLoss |

Same table replicated for Brier Score, AUC, and ECE.

**Expected pattern:** All models converge as game progresses (more information = easier prediction). Draft-only should be the hardest. The interesting question is how fast each model improves with game state data.

---

## Per-Tier Evaluation

All metrics broken down by skill bracket:

| Tier | N matches | Draft LogLoss | 15m LogLoss | 30m LogLoss | Notes |
|------|-----------|---------------|-------------|-------------|-------|
| Iron | | | | | Expect noisier (more throws) |
| Bronze | | | | | |
| Silver | | | | | Largest sample likely |
| Gold | | | | | |
| Platinum | | | | | |
| Emerald | | | | | |
| Diamond | | | | | |
| Master | | | | | Smallest sample |
| Grandmaster+ | | | | | May combine with Master |

---

## Per-Region Evaluation

All metrics broken down by server region:

| Region | N matches | Draft LogLoss | 15m LogLoss | 30m LogLoss | Avg Game Duration | Notes |
|--------|-----------|---------------|-------------|-------------|-------------------|-------|
| NA | | | | | | |
| EUW | | | | | | |
| KR | | | | | | Expect fastest games, most predictable |
| EUNE | | | | | | |
| Others | | | | | | If collected |

**Hypotheses:**
- KR games should be most predictable (more decisive play, fewer throws)
- KR games should have shorter average duration
- Draft accuracy may vary by region (different meta adherence)
- The region embedding should capture these differences automatically

---

**Hypothesis (Tier):** Higher tiers should have:
- Better draft prediction (meta is more respected)
- Faster in-game convergence (fewer throws/comebacks)
- Lower late-game log loss (games close out more cleanly)

---

## Calibration Analysis

### Reliability Diagrams
- X-axis: predicted probability (binned into 20 buckets)
- Y-axis: actual win rate in each bucket
- Perfect calibration = diagonal line
- Generate per-model, per-tier, per-timestamp

### Post-Hoc Calibration

If raw model outputs are miscalibrated:

1. **Platt Scaling:** Logistic regression on validation set predictions
   ```
   p_calibrated = sigmoid(a * logit(p_raw) + b)
   ```
   Simple, 2 parameters, usually sufficient.

2. **Isotonic Regression:** Non-parametric, fits a stepwise non-decreasing function
   - More flexible than Platt
   - Needs more validation data
   - Use if Platt scaling shows systematic pattern

3. **Temperature Scaling:** Divide logits by learned temperature T
   ```
   p_calibrated = sigmoid(logit(p_raw) / T)
   ```
   Even simpler than Platt (1 parameter).

**Protocol:**
- Train on train set, calibrate on calibration set (held out from validation), evaluate on test set
- Compare calibrated vs uncalibrated ECE
- Report both raw and calibrated metrics

---

## Analysis Features

### Per-Champion Value Add

**Method:** UNKNOWN token masking

For each champion C in a draft:
1. Compute P(win | full draft) with all 10 champions
2. Replace C with UNKNOWN token → compute P(win | draft \ C)
3. Value add of C = P(win | full draft) - P(win | draft \ C)

**Interpretation:**
- Positive value → champion C is helping the team
- Negative value → champion C is hurting (bad matchup, bad synergy)
- Sum of all 10 value-adds ≈ 0 (zero-sum-ish, modulo interaction effects)

**Aggregation:**
- Average value add per champion across all games
- Value add per champion per tier (meta varies by elo)
- Value add per champion per patch
- Value add per champion per role (is Caitlyn better ADC or mid?)

**Visualization:**
- Bar chart: top 20 / bottom 20 champions by value add per tier
- Heatmap: champion x tier matrix of value add
- Scatter plot: pick rate vs value add (find underrated/overrated champions)

### Timestep Importance Attribution

Measures how much each moment in the game influenced the final outcome — the temporal analogue of per-champion value add.

**Method 1 — Gradient Attribution (Primary):**
```python
def timestep_importance_gradient(model, inputs, target_timestep=-1):
    """
    Compute ∂P(win_final) / ∂features_t for all timesteps t.
    High gradient magnitude = that timestep was influential.
    """
    inputs.requires_grad_(True)
    p_win = model(inputs)
    p_final = p_win[:, target_timestep]  # final prediction
    p_final.backward()
    # Gradient magnitude per timestep = importance score
    importance = inputs.grad.norm(dim=-1)  # (batch, seq_len)
    return importance
```
- **Fast:** Single backward pass, native PyTorch autograd
- **Interpretable:** Tells you "if the features at timestep t were slightly different, how much would the final prediction change?"
- Can use Integrated Gradients for more robust attribution (interpolate from baseline to actual input)

**Method 2 — Occlusion Masking (Validation):**
```python
def timestep_importance_occlusion(model, inputs, t):
    """
    Replace timestep t's features with interpolated neutral values,
    re-run model, measure delta P(win).
    """
    masked_inputs = inputs.clone()
    if t == 0:
        masked_inputs[:, t] = 0  # zero out draft timestep
    else:
        # Interpolate from t-1 and t+1
        masked_inputs[:, t] = (inputs[:, t-1] + inputs[:, min(t+1, T-1)]) / 2
    p_original = model(inputs)[:, -1]
    p_masked = model(masked_inputs)[:, -1]
    return (p_original - p_masked).abs()
```
- **Expensive:** One forward pass per masked timestep per game
- **More intuitive:** "What if this moment hadn't happened?"
- Use as validation that gradient attribution is working correctly

**Aggregation:**
- Per-game heatmap: timestep importance across the game
- Average importance curve: which game phases matter most across many games
- Per-tier comparison: do late-game moments matter more in low elo (more throws)?

**Visualization:**
- Heatmap overlay on P(win) timeline: importance intensity at each timestep
- "Most important moments" ranking per game

### Swing Detection

**Definition:** A "swing" is a timestep where |delta P(win)| exceeds a threshold.

```python
def detect_swings(p_win_sequence, threshold=0.05):
    """
    p_win_sequence: array of P(win) at each timestep
    threshold: minimum |delta P| to count as swing
    """
    deltas = np.diff(p_win_sequence)
    swing_indices = np.where(np.abs(deltas) > threshold)[0]
    swing_magnitudes = deltas[swing_indices]
    return swing_indices, swing_magnitudes
```

**Attribution:**
For each detected swing, correlate with game events at that timestamp:
- Champion kills (which players involved)
- Objective takes (dragon, baron, tower)
- Item power spikes
- Level advantages

**Relationship to Timestep Importance:** Swing detection measures *what changed the model's prediction between consecutive timesteps* (forward-looking delta). Timestep importance measures *how much each timestep influenced the final outcome* (backward-looking gradient). A swing might not be "important" if the game was already decided. An important timestep might not be a swing if the features changed gradually.

**Visualization:**
- Win probability timeline plot (like AWS broadcast overlay)
- Annotated swings with event descriptions
- Per-player swing contribution (who causes the most game-changing moments)

### Partial Draft Evaluation

Predict P(win) at each stage of champion select:

| Stage | Known | Unknown | Prediction |
|-------|-------|---------|------------|
| Pick 1 | 1 champion | 9 UNKNOWN | P(win) after first pick |
| Pick 2-3 | 3 champions | 7 UNKNOWN | P(win) after second team picks |
| Pick 4-5 | 5 champions | 5 UNKNOWN | P(win) at half-draft |
| Pick 6-7 | 7 champions | 3 UNKNOWN | P(win) entering second rotation |
| Pick 8-9 | 9 champions | 1 UNKNOWN | P(win) with one pick remaining |
| Pick 10 | 10 champions | 0 UNKNOWN | Full draft P(win) |

**Visualization:**
- Draft timeline: P(win) evolution pick by pick
- Decision support: "if you pick champion X here, P(win) = ..."
- Best/worst remaining picks at each stage

### Embedding Analysis

**t-SNE / UMAP Visualization:**
- Plot champion embeddings in 2D
- Color by champion class (assassin, mage, tank, etc.)
- Expect clustering by role/class
- Compare across role tables (is "Yasuo top" near or far from "Yasuo mid"?)

**Cosine Similarity Matrix:**
- 165 x 165 heatmap of champion similarity per role table
- Find "functionally similar" champions
- Identify surprising similarities/dissimilarities

**Synergy/Counter Detection:**
- For each (champ_A, champ_B) pair on the same team: average P(win) when both present
- For each (champ_A, champ_B) pair on opposite teams: average P(win) for team with champ_A
- Rank synergies and counters
- Compare to known community knowledge (sanity check)

---

## Data Splits

### Split Strategy
- **Split by match, not by player** — prevents data leakage (same player's games in train and test)
- **Time-based holdout for final evaluation** — last N days of collected data as test set
- **Random split for development** — 80/10/10 train/val/test on remaining data

```
All matches (sorted by date)
├── Development set (older matches, ~90%)
│   ├── Train (80% of dev)
│   ├── Validation (10% of dev)
│   └── Calibration (10% of dev)  ← used for post-hoc calibration
└── Test set (newest ~10%)        ← final evaluation only
```

### Why Time-Based Holdout
- Simulates real deployment: model trained on past data, predicting future games
- Catches overfitting to time-specific patterns
- More realistic than random split

### Cross-Validation (Optional)
- 5-fold CV on development set for hyperparameter tuning
- Time-series aware: each fold uses earlier data for training, later for validation
- Expensive for neural models — use for GBDT, skip for TCN/GRU

---

## Visualization Plan

### Required Plots

| Plot | Purpose | Tools |
|------|---------|-------|
| Log loss vs timestamp (per model) | Compare model quality over game time | Matplotlib line plot |
| Reliability diagram (per model) | Check calibration | Matplotlib scatter + diagonal |
| AUC vs timestamp (per model) | Discrimination over game time | Matplotlib line plot |
| Per-tier metrics table | Tier-specific performance | Pandas + Matplotlib table |
| P(win) timeline for example games | Showcase swing detection | Plotly interactive |
| Champion value add bar chart | Top/bottom champions per tier | Matplotlib horizontal bar |
| Timestep importance heatmap | Which moments mattered most | Matplotlib/Seaborn heatmap |
| Avg importance by game phase | When do games get decided? | Matplotlib area plot |
| Champion embedding t-SNE | Embedding quality | Matplotlib scatter with annotations |
| Cosine similarity heatmap | Champion relationships | Seaborn heatmap |
| Draft evolution (pick-by-pick P) | Partial draft evaluation | Matplotlib step plot |
| Per-region metrics comparison | Region-specific performance | Matplotlib grouped bar |
| Training curves (loss vs epoch) | Convergence monitoring | Matplotlib line plot |
| Feature importance (GBDT) | What drives GBDT predictions | LightGBM built-in + Matplotlib |
| Confusion matrix at p=0.5 | Simple win/loss classification | Seaborn heatmap |

### Interactive Dashboard (Stretch Goal)
- Streamlit or Plotly Dash
- Select a match → see P(win) timeline with events
- Select a champion → see value add across tiers/patches
- Embedding explorer with hover info
- Draft simulator with live P(win) updates

---

## Success Criteria

### Minimum Viable Model
- [ ] Draft log loss < 0.69 (better than coin flip)
- [ ] 15-minute log loss < 0.55
- [ ] 30-minute log loss < 0.40
- [ ] ECE < 0.03 after calibration
- [ ] Per-champion value add rankings pass sanity check (known strong picks ranked high)

### Competitive Targets
- [ ] Draft accuracy approaching 56% (LoLDraftAI level)
- [ ] 10-minute accuracy approaching 67% (iTero-reported level)
- [ ] Late-game AUC > 0.90
- [ ] Embedding clusters match champion classes

### Stretch Goals
- [ ] Draft accuracy > 58%
- [ ] Swing detection correlates with community-recognized "throws"
- [ ] Partial draft evaluation useful for pick/ban strategy
- [ ] Model generalizes across patches without retraining (patch delta layer)
