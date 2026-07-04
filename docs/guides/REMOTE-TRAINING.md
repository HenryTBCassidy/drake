# DRAKE — Remote Training Runbook

How to use the home GPU box to train and validate DRAKE models from this Mac. Companion to
`docs/guides/STYLE-GUIDE.md` (code conventions).

DRAKE's models are small — the TCN is ~640k params, the GRU ~460k, and the GBDT baseline runs on
CPU. **VRAM is not a constraint for this project**; the box's 8 GB is ample. The box matters for
*speed* (parallel training, real-data runs) and for validating the CUDA path, not for fitting a
big model into memory.

---

## The box

| | |
|---|---|
| OS | Ubuntu (native, kernel 6.17) |
| GPU | NVIDIA GeForce RTX 3060 Ti, 8 GB VRAM (CUDA 13.2, driver 595.71) |
| CPU / RAM | 20 cores / 31 GB |
| Tooling | `uv` (`~/.local/bin/uv`), `tmux`, `git` all present |

It's a **shared personal machine**, not disposable cloud. Be a good tenant: run real work inside
`tmux`, don't leave runaway processes, and check with Henry before launching anything that will
occupy the GPU for hours.

---

## How to reach it

Two SSH aliases are configured in this Mac's `~/.ssh/config`. The concrete host mapping (IP,
user, dev-tunnel details) lives in `local/MY-SETUP.md` (gitignored) — it's kept out of this
public repo deliberately.

| Alias | Works from | Use |
|-------|-----------|-----|
| `gpu-linux` | Mac on the home LAN | Direct, fastest at home. |
| `gpu-anywhere` | Mac on **any** network | Rides a Microsoft dev tunnel over HTTPS — works off-LAN. Default when unsure. |

Quick liveness check:

```bash
ssh gpu-anywhere 'echo ok; nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader'
```

**Box invariants (don't undo these):** sleep is masked (the box never dozes), `sshd` runs
always-on, and user services linger across logins so it survives reboots. Real runs always go
inside `tmux` so a dropped connection can't kill them.

> **Footgun:** never restart `sshd` from inside an SSH session with a bare `systemctl restart` —
> it can kill the issuing session and leave sshd down. Detach it:
> `sudo systemd-run --on-active=2 systemctl restart ssh.service`.

---

## First-time DRAKE setup on the box

The box has `AlphaBlokus` cloned but **not `drake`**. Once:

```bash
ssh gpu-anywhere 'bash -lc "
  cd ~ &&
  git clone https://github.com/HenryTBCassidy/drake.git &&
  cd drake &&
  source ~/.local/bin/env &&
  uv sync --extra dev &&
  uv run pytest -q -m \"not slow\"
"'
```

On Linux, `uv sync` installs the **CUDA-enabled** torch wheel from PyPI by default; the box's
driver is backward-compatible with it. Verify CUDA is live before trusting a training run:

```bash
ssh gpu-anywhere 'bash -lc "cd ~/drake && source ~/.local/bin/env && uv run python -c \"import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)\""'
```

Expect `True NVIDIA GeForce RTX 3060 Ti`. If it prints `False`, torch fell back to CPU — reinstall
torch from the CUDA index before continuing.

---

## Running work on the box

The workflow mirrors local dev: code moves via git, results come back via `rsync`.

```bash
# 1. Push your branch from the Mac, pull it on the box
git push -u origin <branch>
ssh gpu-anywhere 'bash -lc "cd ~/drake && git fetch origin && git checkout <branch> && git pull"'

# 2. Start work inside tmux so it survives disconnects
ssh gpu-anywhere
tmux new -s train
source ~/.local/bin/env && cd ~/drake
uv run <the drake training command>          # e.g. `uv run drake train ...` once the CLI exists
#   detach with Ctrl-b d; reattach later with: tmux attach -t train

# 3. Pull results/artifacts back to the Mac when done
rsync -avz gpu-anywhere:~/drake/<results-dir>/ ./<results-dir>/
```

While a run is going, from the Mac:

```bash
ssh gpu-anywhere 'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader'   # GPU load
ssh gpu-anywhere 'bash -lc "cd ~/drake && git pull && git log --oneline -3"'                   # sync new code (won't affect an in-flight run)
```

An in-flight run won't pick up pulled code changes (Python already imported its modules) — the
next launch does.

---

## Sanity-check commands to keep handy

```bash
ssh gpu-anywhere 'echo ok'                                                              # box alive?
ssh gpu-anywhere 'nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader'  # GPU state
ssh gpu-anywhere 'bash -lc "cd ~/drake && source ~/.local/bin/env && uv run python -c \"import torch; print(torch.cuda.is_available())\""'  # CUDA works?
ssh gpu-anywhere 'tmux ls'                                                              # any runs going?
```

---

## When something this guide doesn't cover comes up

Update this guide. Operational knowledge erodes fast — if you debug the same box oddity twice,
add it here so the next session doesn't relearn it.
