#!/usr/bin/env bash
# Reproduce Track A end to end. Usage: bash run_all.sh [--dry-run]
set -euo pipefail
cd "$(dirname "$0")"
DRY=0; if [ "${1:-}" = "--dry-run" ]; then DRY=1; fi
export HF_HOME="$PWD/.cache/huggingface" TORCH_HOME="$PWD/.cache/torch" XDG_CACHE_HOME="$PWD/.cache" MPLCONFIGDIR="$PWD/.cache/mpl" WANDB_DISABLED=true PIP_REQUIRE_VIRTUALENV=1
if [ -d .venv/Scripts ]; then PY=.venv/Scripts/python.exe; PIP=.venv/Scripts/pip.exe; else PY=.venv/bin/python; PIP=.venv/bin/pip; fi
run() { echo "+ $*"; if [ "$DRY" = 0 ]; then "$@"; fi; }

if [ ! -d .venv ]; then
  echo "+ create .venv and install requirements.txt (torch from the cu128 index)"
  if [ "$DRY" = 0 ]; then
    (python3 -m venv .venv 2>/dev/null || py -3 -m venv .venv)
    if [ -d .venv/Scripts ]; then PY=.venv/Scripts/python.exe; PIP=.venv/Scripts/pip.exe; else PY=.venv/bin/python; PIP=.venv/bin/pip; fi
    $PIP install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements.txt
  fi
fi
run $PY -m gd.data                                   # data/{train,dev,test}.jsonl
run $PY -m pytest tests -q                           # verifier tests
for c in student_none student_prompt teacher_prompt; do run $PY -m gd.eval --cond $c --n_latency 20; done
run $PY -m gd.opcd --run opcd    --guided_frac 0.0 --steps 150 --eval_every 25   # M2
run $PY -m gd.eval --cond opcd --n_latency 20
run $PY -m gd.opcd --run opcd_gr --guided_frac 0.3 --steps 150 --eval_every 25   # M3
run $PY -m gd.eval --cond opcd_gr --n_latency 20
run $PY -m gd.dpo_baseline                                                        # M4
run $PY -m gd.eval --cond dpo --n_latency 20
run $PY -m gd.eval --cond summary
run $PY -m gd.plots
run $PY -m gd.stats                                   # bootstrap CIs + McNemar -> results/ci.md
