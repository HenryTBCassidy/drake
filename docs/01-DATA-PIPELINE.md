# DRAKE — Data Pipeline

## Overview

DRAKE's data pipeline collects League of Legends match data from Riot's API, labels it with reliable tier information using a stable-anchor player strategy, and transforms it into training-ready features for both draft and in-game models.

```
League-v4 API          Match-v5 API          Timeline-v5 API
(rank data)            (match details)       (minute-by-minute)
     |                      |                      |
     v                      v                      v
Stable-Anchor ──────> Match Collection ──────> Feature Engineering
Player Seeding         + Enrichment              + Storage
     |                      |                      |
     v                      v                      v
Tier-labelled         Raw match data           Training-ready
player pool           (Parquet)                features (Parquet)
```

---

## Riot API Overview

### Match-v5 (`/lol/match/v5/matches/{matchId}`)
Returns complete match details after game ends.

**Key fields per participant (10 per match):**
- `championId`, `championName` — champion played
- `teamPosition` — role (TOP, JUNGLE, MIDDLE, BOTTOM, UTILITY)
- `teamId` — 100 (blue) or 200 (red)
- `win` — boolean, the label
- `kills`, `deaths`, `assists`
- `totalMinionsKilled`, `neutralMinionsKilled`
- `goldEarned`, `goldSpent`
- `totalDamageDealtToChampions`, `totalDamageTaken`
- `wardsPlaced`, `wardsKilled`, `visionScore`
- `turretKills`, `inhibitorKills`
- `dragonKills`, `baronKills`
- `item0` through `item6` — final build
- `summoner1Id`, `summoner2Id` — summoner spells
- `puuid` — unique player identifier (for rank lookup)

**Key fields per team (2 per match):**
- `win` — boolean
- `objectives.baron.kills`, `objectives.dragon.kills`, etc.
- `objectives.tower.kills`, `objectives.inhibitor.kills`
- `objectives.riftHerald.kills`
- `bans` — list of banned champions (for draft context)

**Match metadata:**
- `gameDuration` — in seconds
- `gameVersion` — patch (e.g., "14.10.1")
- `queueId` — 420 = Ranked Solo/Duo (our target)
- `platformId` — server region
- `gameCreation` — timestamp

**Important:** Match-v5 does NOT contain any rank/tier information. This is why we need the stable-anchor strategy.

### Timeline-v5 (`/lol/match/v5/matches/{matchId}/timeline`)
Returns minute-by-minute game state snapshots plus discrete events.

**participantFrames (per player, per minute):**
- `totalGold`, `currentGold`
- `xp`, `level`
- `minionsKilled`, `jungleMinionsKilled`
- `position.x`, `position.y`
- `damageStats.totalDamageDoneToChampions`
- `damageStats.totalDamageTaken`

**events (discrete, timestamped):**
- `CHAMPION_KILL` — killer, victim, assistants, position, bounty
- `BUILDING_KILL` — tower/inhibitor destroyed, team
- `ELITE_MONSTER_KILL` — dragon/baron/herald, killer, team
- `ITEM_PURCHASED`, `ITEM_SOLD`, `ITEM_UNDO`
- `WARD_PLACED`, `WARD_KILL`
- `SKILL_LEVEL_UP`
- `TURRET_PLATE_DESTROYED`

**Data retention:** Timeline data is available for approximately 1 year (rolling). Match metadata persists ~2 years.

### League-v4 (`/lol/league/v4/entries/by-summoner/{encryptedSummonerId}`)
Returns current ranked standing for a player.

**Key fields:**
- `tier` — IRON, BRONZE, SILVER, GOLD, PLATINUM, EMERALD, DIAMOND, MASTER, GRANDMASTER, CHALLENGER
- `rank` — I, II, III, IV (within tier)
- `leaguePoints` — 0-100 LP within division
- `wins`, `losses` — total ranked games
- `veteran` — boolean, long-time player in this division
- `freshBlood` — boolean, recently promoted
- `hotStreak` — boolean, on a winning streak
- `inactive` — boolean

**Important:** This is CURRENT rank only. No historical rank data is available through the API.

### Rate Limits

| Key Type | Limit |
|----------|-------|
| Development | 20 requests/second, 100 requests/2 minutes |
| Production | Higher (requires Riot approval) |

**Throughput math (dev key):**
- 1 match = 1 match detail + 1 timeline + 10 rank lookups = 12 API calls
- At ~50 requests/minute sustained (respecting both limits): ~4 matches/minute
- **~240 enriched matches/hour**
- **~5,760 matches/day** continuous crawling

### Regional Routing

| Region | Platform | Routing |
|--------|----------|---------|
| NA | na1 | americas |
| EUW | euw1 | europe |
| KR | kr | asia |
| EUNE | eun1 | europe |

Match-v5 and Timeline use regional routing (americas/europe/asia). League-v4 uses platform routing (na1/euw1/kr).

---

## Stable-Anchor Player Seeding

### The Problem
Riot's Match-v5 API returns no rank information. We need tier labels for training data, but we can't look up historical rank (only current rank), and players' ranks fluctuate.

### The Solution
Seed data collection from players whose current rank is reliably stable — by construction, their recent matches are accurately labelled with their current tier.

### Anchor Player Criteria

```python
def is_stable_anchor(entry):
    return (
        entry['veteran'] == True          # Long-time resident of this division
        and entry['freshBlood'] == False   # Not recently promoted
        and entry['inactive'] == False     # Still actively playing
        and (entry['wins'] + entry['losses']) >= 150  # Substantial game count
        and 0.47 <= win_rate(entry) <= 0.53  # Near 50% WR = settled at true skill
    )
```

### Why This Works
- `veteran=True` + `freshBlood=False` = player has been in this tier for a while
- ~50% win rate = player is at equilibrium, not climbing or falling
- 150+ games = not a smurf or placement variance
- Their last 1-2 weeks of matches are accurately labelled as their current tier

### Collection Flow

```
For each tier (IRON through CHALLENGER):
  1. Query League-v4 for all players in that tier
  2. Filter for stable anchors
  3. For each anchor player:
     a. Get their recent match IDs (last 20 games, ranked solo/duo only)
     b. For each match:
        - Fetch match details (Match-v5)
        - Fetch timeline (Timeline-v5)
        - Label with anchor player's tier
     c. Rate-limit appropriately
  4. Deduplicate matches (same match may appear for multiple anchor players)
  5. Store as Parquet, partitioned by tier and patch
```

### Estimated Data Volumes

| Tier | Anchor Players (est.) | Matches per Player | Unique Matches (est.) |
|------|----------------------|--------------------|-----------------------|
| Iron | 5,000-10,000 | 20 | 30,000-60,000 |
| Bronze | 15,000-30,000 | 20 | 80,000-150,000 |
| Silver | 20,000-40,000 | 20 | 100,000-200,000 |
| Gold | 15,000-30,000 | 20 | 80,000-150,000 |
| Platinum | 10,000-20,000 | 20 | 50,000-100,000 |
| Emerald | 5,000-15,000 | 20 | 30,000-80,000 |
| Diamond | 3,000-8,000 | 20 | 15,000-40,000 |
| Master+ | 1,000-3,000 | 20 | 5,000-15,000 |

**Total estimated:** 400k-800k unique matches across all tiers (NA alone). More with EUW/KR.

### Tier Label Confidence
Since every match in our dataset contains at least one stable-anchor player, we know the match tier with high confidence. For mixed-tier lobbies (which are rare in solo queue), the anchor player's tier represents the lobby's approximate skill level.

---

## Feature Engineering

### Draft Features (T=0)

| Feature | Type | Dimensions | Description |
|---------|------|------------|-------------|
| blue_top_champ | categorical | 1 (→ 32 via embedding) | Champion ID for blue top laner |
| blue_jg_champ | categorical | 1 (→ 32 via embedding) | Champion ID for blue jungler |
| blue_mid_champ | categorical | 1 (→ 32 via embedding) | Champion ID for blue mid laner |
| blue_adc_champ | categorical | 1 (→ 32 via embedding) | Champion ID for blue ADC |
| blue_sup_champ | categorical | 1 (→ 32 via embedding) | Champion ID for blue support |
| red_top_champ | categorical | 1 (→ 32 via embedding) | Champion ID for red top laner |
| red_jg_champ | categorical | 1 (→ 32 via embedding) | Champion ID for red jungler |
| red_mid_champ | categorical | 1 (→ 32 via embedding) | Champion ID for red mid laner |
| red_adc_champ | categorical | 1 (→ 32 via embedding) | Champion ID for red ADC |
| red_sup_champ | categorical | 1 (→ 32 via embedding) | Champion ID for red support |
| tier | categorical | 1 (→ 16 via embedding) | Skill tier (Iron-Challenger, 9 values) |
| lp_proxy | continuous | 1 | Normalized LP (0-1 within tier). Tier embedding captures categorical tier-level patterns; LP adds continuous within-tier granularity (Gold 1 99LP ≠ Gold 4 0LP) |
| region | categorical | 1 (→ 8 via embedding) | Server region (NA, EUW, KR, etc. — 10 values). Different regions have fundamentally different metas, game pacing, and draft priorities |
| patch_major | categorical | 1 (→ 8 via embedding) | Major patch version |
| patch_minor | categorical | 1 (→ 8 via embedding) | Minor patch version |
| season_progress | continuous | 1 | Days since season start, normalized. Early season ranks are noisy (post-reset), late season ranks are stable |
| side | binary | 1 | Already implicit in role x side embeddings |

**Total draft input (after embeddings):** 10 x 32 + 16 + 1 + 8 + 8 + 8 + 1 = 362 dimensions

### In-Game Features (per timestep)

Computed from Timeline-v5 participantFrames, expressed as blue-minus-red differentials where applicable.

| Feature | Description |
|---------|-------------|
| gold_diff | Total gold differential (blue - red team sum) |
| gold_diff_top | Gold diff for top lane matchup |
| gold_diff_jg | Gold diff for jungle matchup |
| gold_diff_mid | Gold diff for mid lane matchup |
| gold_diff_bot | Gold diff for bot lane (ADC+SUP combined) |
| xp_diff | Total XP differential |
| xp_diff_per_lane | XP diff per lane matchup (5 features) |
| cs_diff | Total CS differential |
| cs_diff_per_lane | CS diff per lane (5 features) |
| kill_diff | Team kill differential |
| death_diff | Team death differential |
| assist_diff | Team assist differential |
| tower_diff | Tower kill differential |
| dragon_diff | Dragon kill differential |
| dragon_soul_blue | Binary: blue has dragon soul |
| dragon_soul_red | Binary: red has dragon soul |
| baron_diff | Baron kill differential |
| baron_active_blue | Binary: blue has active baron buff |
| baron_active_red | Binary: red has active baron buff |
| herald_diff | Rift Herald kill differential |
| inhibitor_diff | Inhibitor kill differential |
| vision_diff | Vision score differential |
| level_diff | Average champion level differential |
| plate_gold_diff | Turret plate gold differential |
| game_time | Normalized game time (0-1, capped at 45 min) |

**Momentum features (time windows):**

| Feature | Description |
|---------|-------------|
| delta_gold_2min | Gold diff change over last 2 minutes |
| delta_gold_5min | Gold diff change over last 5 minutes |
| delta_kills_2min | Kill diff change over last 2 minutes |
| delta_kills_5min | Kill diff change over last 5 minutes |
| delta_towers_2min | Tower diff change over last 2 minutes |
| delta_objectives_5min | Objective diff change over last 5 minutes |

**Estimated total in-game features per timestep:** ~50-80 (exact count determined during implementation)

### Timestep Resampling

Timeline-v5 provides data at 1-minute intervals. We resample to 30-second intervals via linear interpolation for smoother predictions. Discrete events (kills, objectives) are assigned to their actual timestamps.

For a typical 30-minute game:
- Raw frames: 30 timestamps
- After 30s resampling: ~60 timestamps
- With draft as T=0: 61 total timesteps per game

### GBDT-Specific Features

The GBDT baseline uses hand-crafted features since it can't use embeddings:

| Feature | Dimensions | Description |
|---------|-----------|-------------|
| champion_one_hot | 1650 (165 x 10 roles) | Sparse one-hot per role slot |
| tier_one_hot | 9 | One-hot tier encoding |
| lp_proxy | 1 | Normalized LP (0-1 within tier) |
| region_one_hot | 10 | One-hot region encoding |
| patch_numeric | 2 | Major and minor patch as numbers |
| season_progress | 1 | Normalized days since season start |
| side | 1 | Blue=0, Red=1 |
| all in-game features above | ~80 | Same as neural model |

---

## Storage Format

### Why Parquet (Not a Database)

This is a batch ML pipeline: collect → process → train → evaluate. There's no concurrent access, no real-time writes, no multi-user requirement. Parquet is purpose-built for this pattern:

- **Columnar:** fast to read subsets of features (e.g. only gold_diff and xp_diff for a quick analysis)
- **Compression:** numeric game data compresses 5-10x. Repeated draft features (same 10 champion IDs at every timestep) compress nearly for free
- **Native ML support:** pandas, PyArrow, and PyTorch DataLoaders read Parquet directly — no serialisation step
- **Zero infrastructure:** just files on disk. No server to manage, no connection strings, no schema migrations

**Why not Postgres/MySQL:** Row-oriented storage is slower for analytical reads across many columns (which is every training batch). Adds operational overhead (server, connections) with no benefit for a single-user pipeline.

**Why not Snowflake:** Cloud data warehouse designed for enterprise analytics at scale. Overkill by several orders of magnitude.

### DuckDB for Exploration (Optional)

DuckDB queries Parquet files directly with SQL, in-process, zero config. Useful for ad-hoc exploration in notebooks:

```sql
-- How many Gold matches do we have from patch 14.10 on EUW?
SELECT COUNT(DISTINCT match_id) FROM 'data/processed/game_features/*.parquet'
WHERE tier = 3 AND patch_major = 14 AND patch_minor = 10 AND region = 1;

-- Average game duration by tier
SELECT tier, AVG(MAX(game_time_sec)) as avg_duration
FROM 'data/processed/game_features/*.parquet'
GROUP BY match_id, tier
GROUP BY tier ORDER BY tier;
```

DuckDB is a convenience layer for poking around data, not load-bearing infrastructure. The training pipeline reads Parquet directly.

### Two-Stage Storage Design

**Stage 1 — Raw:** Data arrives from three separate API endpoints, stored as-is.

**Stage 2 — Processed:** Feature engineering joins everything into a single fully denormalized dataset. Each row is one `(match_id, timestep)` pair containing ALL features needed for training — draft info, game state, tier, region, patch, label. No joins at training time.

Draft features (champion IDs, tier, region, etc.) repeat identically at every timestep within a match. This looks wasteful but Parquet's columnar compression handles it efficiently — a column of 61 identical int16 values compresses to nearly nothing.

### Directory Layout

```
data/
├── raw/                            # Stage 1: as-collected from API
│   ├── seed_players/
│   │   ├── na1_GOLD.parquet        # Stable-anchor player lists per tier
│   │   ├── na1_PLATINUM.parquet
│   │   └── ...
│   ├── matches/
│   │   ├── na1/
│   │   │   ├── patch_14.10/
│   │   │   │   ├── GOLD_matches.parquet    # Match-v5 responses
│   │   │   │   ├── GOLD_timelines.parquet  # Timeline-v5 responses
│   │   │   │   └── ...
│   │   │   └── ...
│   │   └── ...
│   └── ranks/
│       └── na1_ranks.parquet       # League-v4 rank lookups
├── processed/                      # Stage 2: fully denormalized, training-ready
│   ├── game_features/
│   │   ├── GOLD_games.parquet      # One row per (match_id, timestep)
│   │   └── ...                     #   with ALL features + label
│   └── metadata.json               # Collection stats, feature schema, column dtypes
├── checkpoints/
│   └── collection.db               # SQLite — tracks crawl progress (resume support)
└── splits/
    ├── train_match_ids.txt
    ├── val_match_ids.txt
    └── test_match_ids.txt
```

### Parquet Schema (processed game features)

Each row is one `(match_id, timestep)` pair. A 30-minute game produces ~61 rows.

```
# ── Identity ──
match_id:       string    # Unique match identifier
timestep:       int32     # 0 = draft, 1-N = in-game (30s intervals)
game_time_sec:  int32     # Actual seconds into the game

# ── Context (same for every timestep in a match) ──
tier:           int8      # 0=Iron, 1=Bronze, ..., 8=Challenger
lp_proxy:       float32   # Normalized LP within tier (0-1)
region:         int8      # 0=NA, 1=EUW, 2=KR, 3=EUNE, ...
patch_major:    int8      # e.g., 14
patch_minor:    int8      # e.g., 10
season_progress: float32  # Days since season start, normalized

# ── Draft (same for every timestep in a match) ──
blue_top:       int16     # Champion ID
blue_jg:        int16
blue_mid:       int16
blue_adc:       int16
blue_sup:       int16
red_top:        int16
red_jg:         int16
red_mid:        int16
red_adc:        int16
red_sup:        int16

# ── Game state (zeros at T=0, populated at T>0) ──
gold_diff:      float32
xp_diff:        float32
... (all game features)

# ── Label (same for every timestep in a match) ──
label:          int8      # 1 = blue win, 0 = red win
```

---

## Data Quality Filters

| Filter | Criteria | Reason |
|--------|----------|--------|
| Queue type | queueId == 420 (Ranked Solo/Duo) | Only competitive games |
| Game duration | > 300 seconds (5 min) | Filter remakes |
| AFK detection | All players have >50% game time activity | Remove AFK/disconnect games |
| Patch boundaries | Separate by major patch | Avoid cross-patch training contamination |
| Duplicate removal | Deduplicate by matchId | Same match found via multiple anchor players |

---

## Pipeline Robustness

### Resume Support
- Track collected match IDs in a SQLite checkpoint database
- On restart, skip already-collected matches
- Separate checkpoints for: seed players collected, match IDs found, matches downloaded, timelines downloaded

### Error Handling
- Retry with exponential backoff on 429 (rate limit) and 5xx errors
- Skip and log on 404 (match not found — very old or deleted)
- Log and continue on malformed data (missing fields, unexpected values)

### Monitoring
- Progress bar per tier with ETA
- Log: matches/hour, API calls/minute, errors/hour
- Alert if rate limit hit rate exceeds threshold
