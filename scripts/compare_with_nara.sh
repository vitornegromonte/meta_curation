#!/usr/bin/env bash
# Full DataRater-vs-NARA test on a prepared tabular dataset.
#
# Phase 2 (always): synthetic pool with known junk -> DataRater vs NARA
#   Data-IQ (dataiq-lr, dataiq-mlp) vs random, same downstream eval.
# Phase 1 (opt-in): curated-vs-full-vs-random on the real train split.
#
# Usage:
#   scripts/compare_with_nara.sh [--dataset parkinson] [--meta-steps 600]
#     [--n-synth 8000] [--junk-frac 0.3] [--out-dir reports/full_test]
#     [--phase1] [--setup-env] [--python python3]
#
# --setup-env creates .venv-fulltest (torch CPU + numpy + scikit-learn).
set -euo pipefail

DATASET="parkinson"
META_STEPS="600"
N_SYNTH="8000"
JUNK_FRAC="0.3"
OUT_DIR=""
PHASE1="0"
SETUP_ENV="0"
PYTHON="python3"

while [ $# -gt 0 ]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2;;
    --meta-steps) META_STEPS="$2"; shift 2;;
    --n-synth) N_SYNTH="$2"; shift 2;;
    --junk-frac) JUNK_FRAC="$2"; shift 2;;
    --out-dir) OUT_DIR="$2"; shift 2;;
    --phase1) PHASE1="1"; shift;;
    --setup-env) SETUP_ENV="1"; shift;;
    --python) PYTHON="$2"; shift 2;;
    -h|--help) sed -n '2,14p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 1;;
  esac
done

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
[ -z "$OUT_DIR" ] && OUT_DIR="reports/full_test_${DATASET}"
DATA_ROOT="nara/datasets/prepared/${DATASET}"
TRAIN_CSV="${DATA_ROOT}/${DATASET}_train.csv"
[ -f "$TRAIN_CSV" ] || { echo "missing dataset: $TRAIN_CSV" >&2; exit 1; }

if [ "$SETUP_ENV" = "1" ] && [ ! -x ".venv-fulltest/bin/python" ]; then
  echo "-> creating .venv-fulltest (torch CPU + numpy + scikit-learn)..."
  python3 -m venv .venv-fulltest
  .venv-fulltest/bin/pip install --quiet --index-url https://download.pytorch.org/whl/cpu torch
  .venv-fulltest/bin/pip install --quiet numpy scikit-learn
fi
if [ -x ".venv-fulltest/bin/python" ]; then PYTHON=".venv-fulltest/bin/python"; fi
"$PYTHON" -c "import torch, sklearn, numpy" 2>/dev/null \
  || { echo "missing deps (torch/sklearn/numpy). Re-run with --setup-env." >&2; exit 1; }

N_TRAIN=$(($(wc -l < "$TRAIN_CSV") - 1))
echo "dataset: $DATASET ($N_TRAIN train rows)"
[ "$N_TRAIN" -lt 500 ] && echo "WARNING: <500 rows - meta-learning will be noisy."

mkdir -p "$OUT_DIR"
echo "out: $OUT_DIR"

if [ "$PHASE1" = "1" ]; then
  echo "=== phase 1: real-data curation ==="
  "$PYTHON" -m meta_curation.nara_adapter --dataset "$DATASET" --data-root "$DATA_ROOT" \
    --meta-steps "$META_STEPS" --out-dir "$OUT_DIR/phase1" 2>&1 | tee "$OUT_DIR/phase1.log" | tail -5
fi

echo "=== phase 2: DataRater vs NARA Data-IQ ==="
"$PYTHON" -m meta_curation.nara_phase2 --dataset "$DATASET" --data-root "$DATA_ROOT" \
  --meta-steps "$META_STEPS" --n-synth "$N_SYNTH" --junk-frac "$JUNK_FRAC" \
  --out-dir "$OUT_DIR/phase2" 2>&1 | tee "$OUT_DIR/phase2.log" | tail -20

echo "=== summary ==="
"$PYTHON" - "$OUT_DIR/phase2/run.pt" <<'EOF' 2>/dev/null | tee "$OUT_DIR/summary.txt"
import sys
import torch

r = torch.load(sys.argv[1], weights_only=False)
a = r["args"]
print(f"dataset meta_steps={a['meta_steps']} n_synth={a['n_synth']} junk_frac={a['junk_frac']}")
print(f"{'arm':12s} {'kept':>6s} {'junk':>6s} {'test MSE':>9s}")
for name, idx in r["arms"].items():
    import numpy as np

    idx = np.asarray(idx)
    junk = float(r["is_junk"].numpy()[idx].mean())
    print(f"{name:12s} {len(idx):6d} {junk:6.3f} {r['test_mse'][name]:9.4f}")
import torch as _t

print("corr(score, junk) =", round(float(_t.corrcoef(_t.stack([r["scores"], r["is_junk"].float()]))[0, 1]), 3))
EOF
echo "saved: $OUT_DIR/summary.txt"
