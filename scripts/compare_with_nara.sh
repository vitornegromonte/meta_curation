#!/usr/bin/env bash
# Full DataRater-vs-NARA test on prepared tabular datasets.
#
# Phase 2: synthetic pool with known junk -> DataRater vs NARA
#   Data-IQ (dataiq-lr, dataiq-mlp) vs random, same downstream eval.
# Phase 1: curated-vs-full-vs-random on the real train split.
# Both phases run by default; --no-phase1 runs phase 2 only.
#
# Usage:
#   scripts/compare_with_nara.sh [--dataset parkinson] [--all]
#     [--meta-steps 600] [--n-synth 8000] [--junk-frac 0.3]
#     [--out-dir reports/full_test] [--no-phase1] [--setup-env] [--python python3]
#
# --all runs every dataset under nara/datasets/prepared and writes a
# combined comparison table (combined_summary.txt).
# --setup-env creates .venv-fulltest (torch CPU + numpy + scikit-learn).
set -euo pipefail

DATASET="parkinson"
ALL="0"
META_STEPS="600"
N_SYNTH="8000"
JUNK_FRAC="0.3"
OUT_DIR=""
PHASE1="1"
SETUP_ENV="0"
PYTHON="python3"

while [ $# -gt 0 ]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2;;
    --all) ALL="1"; shift;;
    --meta-steps) META_STEPS="$2"; shift 2;;
    --n-synth) N_SYNTH="$2"; shift 2;;
    --junk-frac) JUNK_FRAC="$2"; shift 2;;
    --out-dir) OUT_DIR="$2"; shift 2;;
    --no-phase1) PHASE1="0"; shift;;
    --setup-env) SETUP_ENV="1"; shift;;
    --python) PYTHON="$2"; shift 2;;
    -h|--help) sed -n '2,17p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 1;;
  esac
done

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ "$SETUP_ENV" = "1" ] && [ ! -x ".venv-fulltest/bin/python" ]; then
  echo "-> creating .venv-fulltest (torch CPU + numpy + scikit-learn)..."
  python3 -m venv .venv-fulltest
  .venv-fulltest/bin/pip install --quiet --index-url https://download.pytorch.org/whl/cpu torch
  .venv-fulltest/bin/pip install --quiet numpy scikit-learn
fi
if [ -x ".venv-fulltest/bin/python" ]; then PYTHON=".venv-fulltest/bin/python"; fi
"$PYTHON" -c "import torch, sklearn, numpy" 2>/dev/null \
  || { echo "missing deps (torch/sklearn/numpy). Re-run with --setup-env." >&2; exit 1; }

if [ "$ALL" = "1" ]; then
  DATASETS="$(for d in nara/datasets/prepared/*/; do
    n=$(basename "$d")
    [ -f "$d/${n}_train.csv" ] && [ -f "$d/${n}_test.csv" ] && echo "$n"
  done | sort | tr '\n' ' ')"
  [ -z "$DATASETS" ] && { echo "no datasets found under nara/datasets/prepared" >&2; exit 1; }
  echo "datasets: $DATASETS"
else
  DATASETS="$DATASET"
fi
[ -z "$OUT_DIR" ] && OUT_DIR="reports/full_test"
mkdir -p "$OUT_DIR"
echo "out: $OUT_DIR"

run_one() {
  local ds="$1" dest="$2"
  local data_root="nara/datasets/prepared/${ds}"
  local n_train
  n_train=$(($(wc -l < "${data_root}/${ds}_train.csv") - 1))
  echo "=== $ds ($n_train train rows) ==="
  [ "$n_train" -lt 500 ] && echo "WARNING: <500 rows - meta-learning will be noisy."
  mkdir -p "$dest"
  if [ "$PHASE1" = "1" ]; then
    echo "--- $ds phase 1: real-data curation ---"
    "$PYTHON" -m meta_curation.nara_adapter --dataset "$ds" --data-root "$data_root" \
      --meta-steps "$META_STEPS" --out-dir "$dest/phase1" > "$dest/phase1.log" 2>&1 \
      || { echo "$ds phase 1 FAILED (see $dest/phase1.log)"; return 1; }
    tail -4 "$dest/phase1.log"
  fi
  echo "--- $ds phase 2: DataRater vs NARA Data-IQ ---"
  "$PYTHON" -m meta_curation.nara_phase2 --dataset "$ds" --data-root "$data_root" \
    --meta-steps "$META_STEPS" --n-synth "$N_SYNTH" --junk-frac "$JUNK_FRAC" \
    --out-dir "$dest/phase2" > "$dest/phase2.log" 2>&1 \
    || { echo "$ds phase 2 FAILED (see $dest/phase2.log)"; return 1; }
  tail -12 "$dest/phase2.log"
}

FAILED=""
for ds in $DATASETS; do
  run_one "$ds" "$OUT_DIR/$ds" || FAILED="$FAILED $ds"
done

echo "=== combined summary (phase 2) ==="
"$PYTHON" - "$OUT_DIR" $DATASETS <<'EOF' | tee "$OUT_DIR/combined_summary.txt"
import os
import sys

import torch

out, datasets = sys.argv[1], sys.argv[2:]
print(f"{'dataset':12s} {'arm':12s} {'kept':>6s} {'junk':>6s} {'test MSE':>9s}")
for ds in datasets:
    p = os.path.join(out, ds, "phase2", "run.pt")
    if not os.path.exists(p):
        print(f"{ds:12s} FAILED (no run.pt)")
        continue
    r = torch.load(p, weights_only=False)
    first = True
    for name, idx in r["arms"].items():
        import numpy as np

        idx = np.asarray(idx)
        junk = float(r["is_junk"].numpy()[idx].mean())
        tag = ds if first else ""
        print(f"{tag:12s} {name:12s} {len(idx):6d} {junk:6.3f} {r['test_mse'][name]:9.4f}")
        first = False
EOF
[ -n "$FAILED" ] && echo "FAILED datasets:$FAILED" && exit 1
if [ "$ALL" = "0" ]; then cp "$OUT_DIR/combined_summary.txt" "$OUT_DIR/summary.txt"; fi
echo "saved: $OUT_DIR/combined_summary.txt"
