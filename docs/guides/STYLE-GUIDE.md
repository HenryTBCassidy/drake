# DRAKE — Style Guide

Living document. Append as we discover new conventions. All code in this repo should follow
these rules. Reference this before reviewing or writing code.

> Adapted from the AlphaBlokus style guide — DRAKE inherits the same engineering conventions.

---

## Philosophy

- **Consistent over clever.** One way to do things, everywhere.
- **Type everything.** If Python can type-check it, annotate it.
- **No boilerplate.** If a framework eliminates ceremony, use it (loguru over stdlib logging,
  dataclasses over manual `__init__`, etc.).
- **Mathematical where appropriate.** ML has standard notation. Use it when it aids
  understanding, but not at the cost of readability for non-specialists.
- **Flat is better than nested.** Prefer early returns, guard clauses, and flat loops over
  deep nesting.
- **Delete dead code.** Don't comment it out, don't leave it behind a flag. Git has history.
- **Correct first, fast second.** Get the pipeline right, then optimise the parts that measure slow.

---

## Project Layout

The code is an installable `src/`-layout package (`src/drake/`, hatchling build). Tests and
scripts import the installed package — never rely on the repo root being on `sys.path`, and
never add `PYTHONPATH` incantations.

The concrete module layout is the implementer's decision (see `AGENTS.md` → "You own the
architecture"). Whatever the layout:

- Pipeline code goes in the subpackage matching its phase (e.g. data collection, feature
  engineering, model definitions, training, evaluation). Small cross-cutting modules (config,
  protocol definitions, the composition root) stay at the package root.
- **One composition root.** A single module is the only place allowed to import and name
  concrete model/backend classes and wire them from config. Everything else depends on
  protocol interfaces, not concretions.
- Tests mirror the source tree one-to-one under `tests/` (`tests/data/test_features.py` ↔
  `data/features.py`). Shared test utilities that scripts also need ship inside the package,
  not in `tests/`.
- Operational scripts go at `scripts/` top level; measurement tooling under
  `scripts/benchmarks/` or `scripts/profiling/`.

---

## Naming

| Thing | Convention | Example |
|-------|-----------|---------|
| Classes | PascalCase | `TcnUnifiedModel`, `RiotApiClient`, `FeatureBuilder` |
| Methods / functions | snake_case | `collect_matches`, `build_draft_features` |
| Variables | snake_case | `match_id`, `gold_diff`, `stable_anchors` |
| Constants | SCREAMING_SNAKE | `RANKED_SOLO_QUEUE_ID`, `UNKNOWN_CHAMPION_INDEX` |
| Type aliases | PascalCase | `FeatureMatrix`, `ChampionId`, `Tier` |
| Files / modules | snake_case | `features.py`, `riot_client.py` |
| Directories | snake_case | `run_configurations/`, `evaluation/` |
| Test files | `test_` prefix | `test_features.py`, `test_riot_client.py` |

### Specific conventions

- **Config parameters:** always `config` (not `args`, not `run_config`). Exception: when the
  type disambiguates (e.g. `model_config: ModelConfig`).
- **Neural networks:** `net` for the PyTorch module, `wrapper`/`model` for what wraps it.
- **No single-letter variables** except in comprehensions and trivial loops (`i` in `range(n)`).
- **Boolean variables and parameters** read as assertions: `is_stable`, `has_timeline`,
  `should_resample`. Not `stable`, `timeline`, `resample`.
- **Acronyms in names:** well-known standalone acronyms stay uppercase (`GBDT`, `TCN`, `GRU`,
  `API`, `LP`, `AUC`, `ECE`). Lowercase them only when embedded in a longer camelCase word.
- **Be specific over generic.** `draft_log_loss` not `loss`. `champion_embedding` not `emb`.
- **Encode units in names when ambiguous.** `game_duration_seconds` not `duration`.
  `timeout_seconds` not `timeout`.
- **Use domain vocabulary.** LoL has precise terms — use them: `tier`, `division`, `LP`,
  `participant`, `timeline`, `objective`, `dragon_soul`, `champion_select`, `swing`.
- **Avoid abbreviations** except universal ones: `config`, `init`, `num`, `idx`, `dir`, `fn`.

---

## Type System

### Always annotate

Every function signature must have full type annotations — parameters and return type. No
exceptions. This is machine-enforced by `mypy` (`disallow_untyped_defs`), which runs in CI.

```python
# Good
def build_draft_features(self, match: MatchRecord) -> FeatureRow: ...

# Bad
def build_draft_features(self, match): ...
```

### Use modern syntax (Python 3.10+)

```python
# Good
tuple[NDArray, int]        list[MatchRecord]        dict[str, float]        int | None

# Bad
from typing import Tuple, List, Dict, Optional
Tuple[NDArray, int]        Optional[int]
```

### TypeAlias for domain types

Define at the top of each module, after imports:

```python
from typing import TypeAlias

FeatureMatrix: TypeAlias = NDArray[np.float32]
ChampionId: TypeAlias = int
MatchId: TypeAlias = str
```

### Interfaces: Protocol with explicit subclassing

Define interfaces as `typing.Protocol`; implementations **explicitly inherit** from the
Protocol (like `class Foo : IFoo` in C#). This gives explicit intent, static enforcement of
missing methods, and structural subtyping as a fallback.

```python
class IWinProbabilityModel(Protocol):
    def fit(self, dataset: TrainingData) -> None: ...
    def predict(self, features: FeatureMatrix) -> NDArray: ...

class GbdtBaseline(IWinProbabilityModel):
    def fit(self, dataset: TrainingData) -> None: ...
    def predict(self, features: FeatureMatrix) -> NDArray: ...
```

- **Do not** use `@runtime_checkable` — it only checks method existence, not signatures.
- **Do not** put `__init__` on Protocols — implementations have their own constructors.

### No quoted type annotations

Never use string-quoted forward references (`model: "IWinProbabilityModel"`). Add
`from __future__ import annotations` at the top of the file to make annotations lazy, or order
definitions so the type is already defined.

### Frozen dataclasses for value objects

Configuration, DTOs, and immutable domain objects are `@dataclass(frozen=True)` — immutable,
hashable, and clearly "a value, not a mutable entity." Use computed `@property` for derived
values (paths, dimensions).

---

## Documentation

### Docstrings: Google style

```python
def is_stable_anchor(entry: LeagueEntry) -> bool:
    """Decide whether a ranked player is a reliable tier anchor.

    A stable anchor's current rank reliably labels their recent matches: a
    long-time resident of the division (veteran, not fresh blood), still
    active, with enough games and a near-50% win rate to be at equilibrium.

    Args:
        entry: A League-v4 ranked entry for one player.

    Returns:
        True if the player qualifies as a stable anchor.
    """
```

- **Always** docstring public classes, public methods, module-level functions.
- **Skip** private helpers where the name + type hints are self-documenting, and `__init__`
  when the class docstring covers construction.

### Comments

- Don't state the obvious. `i += 1  # increment i` is noise.
- **Do** explain *why*, not *what*: `# Blue-minus-red so positive always means blue is ahead`.
- **Do** flag non-obvious correctness constraints: `# queueId 420 = ranked solo/duo only`.
- Use `# TODO:` for known work items, with a doc/ticket reference when possible.

---

## Imports

Order (enforced by ruff): stdlib → third-party → local, blank line between groups.

```python
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from drake.config import RunConfig
```

- **No `*` imports.** Ever.
- **No relative imports.** Always absolute: `from drake.config import RunConfig`.
- **Import modules, not many names**, when you'll use several from one module
  (`import numpy as np`, not `from numpy import array, zeros, where`).

---

## Code Structure

### One primary class per file

One main class plus the helpers only it uses. A helper shared across classes gets its own file.
Multiple classes in one file is fine when they share one cohesive concern.

### Module layout (newspaper order — public first)

1. Module docstring → 2. Imports → 3. Constants → 4. Type aliases → 5. Public classes
(public methods first, then private) → 6. Module-level code (rare — avoid).

**Method ordering:** callees before callers — the reader looks *up* to find an implementation,
never *down*. Private helpers live inside the class that uses them, not floating at module level.

### Early returns

```python
# Good
def collect(self, match_id: MatchId) -> MatchRecord | None:
    if self.already_have(match_id):
        return None
    return self.fetch(match_id)
```

### f-strings for formatting

Use f-strings everywhere except logging, where loguru's lazy `{}` placeholders are used:

```python
logger.info("Collected {} matches for tier {}", count, tier)   # logging: lazy {}
path = f"matches_{tier}_{patch}.parquet"                         # everything else: f-string
```

---

## Configuration

- **Frozen dataclasses** for all config objects.
- **JSON files** in `run_configurations/` for run parameters; native JSON types
  (`true`/`false`, not `"True"`).
- **Computed properties** for derived values (paths, feature dimensions).
- **No hardcoded config values in source.** If it might change between runs, it goes in config.

---

## Logging

- **loguru** for all *application* logging — progress, warnings, errors, diagnostic state.
  Never `print()` for logging.
- **`print()` is fine for deliberate human-facing console output**: CLI-rendered summaries and
  tables, interactive displays (e.g. a reliability-diagram dump, a value-add ranking), and
  standalone `__main__` dev utilities. The distinction is intent: a *diagnostic stream* uses
  loguru; the *stdout a user explicitly asked to see* can use `print()`.
- Structured data (metrics, feature stats) goes to Parquet/results, not log files.
- Levels: `DEBUG` (detailed state) · `INFO` (progress: collection/training milestones) ·
  `WARNING` (recoverable: rate-limit backoff, dropped malformed match) · `ERROR` (operation-stopping).

---

## Testing

- **pytest** — function-based tests, no test classes.
- **Fixtures** in `conftest.py` for shared setup (configs, synthetic matches).
- **`@pytest.mark.parametrize`** for combinatorial tests.
- **`@pytest.mark.slow`** for integration tests (>5 seconds); CI runs `-m "not slow"`.
- **No mocks for pipeline/feature logic** — use real objects and synthetic match fixtures.
  Mock only true external boundaries (the Riot HTTP API), and prefer recorded/synthetic
  responses over hand-built mocks.
- **Test file naming:** `test_{module}.py` mirroring source structure.
- **Assertions:** plain `assert` with descriptive messages.

---

## Git Workflow

### Branch naming: `{type}/{short-kebab-description}`

| Type | When |
|------|------|
| `feat/` | New features |
| `fix/` | Bug fixes |
| `refactor/` | Structural changes, no behaviour change |
| `chore/` | Tooling, CI, docs, config |
| `test/` | Adding or fixing tests |

### Commit messages

**One sentence, imperative mood.** Aim ≤72 chars, hard cap ~100. No body, no bullets — the
diff shows *what*, the PR description carries the longer *why*. If you want a body, the commit
is doing too much; split it.

```
Add stable-anchor seeding to the Riot collection pipeline
```

### PRs

One concern per PR. Title matches the commit subject if single-commit. Link the doc section
being implemented (e.g. "Implements docs/01-DATA-PIPELINE.md § Stable-Anchor Player Seeding").

---

## Tooling

All three gates run in CI (`.github/workflows/ci.yml`) on every push/PR; keep them green locally.

- **mypy — strict typing is machine-enforced.** `uv run mypy` with `disallow_untyped_defs`,
  `disallow_incomplete_defs`, `no_implicit_optional`. For genuinely untypeable third-party
  corners, prefer a narrow `# type: ignore[<code>]  # <reason>` over module-wide overrides.
- **ruff — lint and format.** `uv run ruff check .` plus `uv run ruff format --check .`
  (line length 120). Run `ruff format` before committing; never hand-fight the formatter.
- **pytest.** `uv run pytest -m "not slow"` in CI; run the full suite including `slow` before
  merging anything that touches training or the data pipeline.

---

## Anti-Patterns — What We Don't Do

- ❌ `print()` for logging/diagnostics — use loguru (`print()` is fine for deliberate console display)
- ❌ Magic numbers — extract to named constants
- ❌ Code duplication — extract shared code
- ❌ Mutable global state — pass state explicitly
- ❌ Commented-out code — delete it, git has history
- ❌ `from module import *` — explicit imports only
- ❌ Bare `except:` — always catch specific exceptions
- ❌ `os.path` — use `pathlib.Path`
- ❌ `typing.Tuple/List/Dict/Optional` — use builtins + `X | None`
- ❌ Quoted type annotations — use `from __future__ import annotations`
- ❌ Docstrings on obvious one-liners — the type hints are enough
- ❌ Deep nesting — refactor with early returns / guard clauses

---

## Patterns We Preserve

- ✅ Frozen dataclasses for config and DTOs
- ✅ TypeAlias definitions at the top of files
- ✅ Protocol-based interfaces with explicit subclassing
- ✅ StrEnum for enumerations (tiers, regions, roles)
- ✅ Computed properties on config objects
- ✅ One composition root; the rest of the code depends on protocols
- ✅ Google-style docstrings with Args/Returns
- ✅ `time.perf_counter()` for timing (not `time.time()`)
- ✅ `pathlib.Path` for all filesystem operations
- ✅ Explicit `__init__.py` re-exports for public APIs
