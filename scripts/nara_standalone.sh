#!/usr/bin/env bash
# Run the NARA (profile2gen) pipeline standalone.
#
# The upstream nara/ tree is a nested repo with placeholder paths, so this
# harness NEVER edits it: `localize` copies code into $WORK/code with paths
# resolved, symlinks the real datasets, and every stage runs from there.
# Functional deviations from upstream are marked NARA-DEV in the localizer
# (dataset args for get_parameters, absolute path2save, csv delimiter sniff,
#  sculpt repointed at traditional outputs: the profiled-stage generator
#  module `generating_sculped` does not exist upstream).
#
# Usage: scripts/nara_standalone.sh <cmd> [args]
#   setup [core|eval]      create .venv-nara (+deps; eval adds autogluon)
#   localize               (re)build $WORK from nara/
#   params D M             optuna hparams for model M on dataset D
#   choosing D             best profiler (pre: real data)
#   generate D M           traditional synthetic data
#   sculpt D M             profile traditional output -> indices (easy/hard)
#   final D M              drop hard -> FinalData
#   evaluate-orig D        AutoGluon on real data (needs setup eval)
#   evaluate-synth D M     AutoGluon on FinalData (needs setup eval)
#   pipeline D M           params -> choosing -> generate -> sculpt -> final
#   status                 show what exists in $WORK
#
# Env: NARA_WORK (default <repo>/nara-standalone), NARA_TRIALS (default 10),
#   NARA_VENV (default <repo>/.venv-nara).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="${NARA_WORK:-$ROOT/nara-standalone}"
VENV="${NARA_VENV:-$ROOT/.venv-nara}"
PY="$VENV/bin/python"

cmd="${1:-help}"; shift || true

setup_core() {
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install --quiet --index-url https://download.pytorch.org/whl/cpu torch
  "$VENV/bin/pip" install --quiet numpy pandas scikit-learn synthcity optuna \
    xgboost cleanlab pydantic tqdm ipython
  echo "core env ready: $VENV"
}

case "$cmd" in
  setup) setup_core;;
  setup-eval)
    [ -x "$PY" ] || setup_core
    "$VENV/bin/pip" install --quiet autogluon.tabular
    echo "eval env ready";;
  localize)
    python3 "$ROOT/scripts/nara_localize.py" "$ROOT" "$WORK"
    mkdir -p "$WORK"/{models,generated,indices,FinalData,model_status,results_evaluation,outs}
    ln -sfn "$ROOT/nara/datasets" "$WORK/datasets"
    echo "work tree: $WORK";;
  status)
    for d in code datasets models generated indices FinalData model_status results_evaluation outs; do
      [ -e "$WORK/$d" ] && echo "OK   $d" || echo "MISS $d"
    done
    du -sh "$WORK" 2>/dev/null;;
  params)
    D="${1:-fat}"; M="${2:-ctgan}"
    (cd "$WORK/code/find_best_param_generative_models" &&
      PYTHONPATH="$WORK/code/find_best_param_generative_models/utils" \
      "$PY" get_parameters.py --model_name "$M" --dataset "$D" \
        --data-root "$WORK/datasets/prepared" --path2save "$WORK/models");;
  choosing)
    D="${1:-fat}"
    (cd "$WORK/code/choosing_framework" &&
      PYTHONPATH="$WORK/code/choosing_framework/utils" \
      "$PY" executing_chosing_framework.py \
        --data_path_real "$WORK/datasets/prepared" \
        --path2save "$WORK/models/frameworks/pre/$D" \
        --noise_level "[0, 0.02, 0.08, 0.1, 0.25, 0.20]" \
        --dataset_name "$D" \
        --datacentric_treshold "[0.1, 0.125, 0.15, 0.175, 0.2, 0.25]" \
        --stage pre);;
  generate)
    D="${1:-fat}"; M="${2:-ctgan}"
    (cd "$WORK/code/generating_synth_data/traditional" &&
      PYTHONPATH="$WORK/code/generating_synth_data/utils" \
      "$PY" executing_generating.py --model_name "$M" --dataset_name "$D");;
  sculpt)
    D="${1:-fat}"; M="${2:-ctgan}"
    mkdir -p "$WORK/models/frameworks/post/$D/$M"
    ln -sfn "$WORK/models/frameworks/pre/$D/best_framework.npy" \
      "$WORK/models/frameworks/post/$D/$M/best_framework.npy"
    (cd "$WORK/code/profiling_synth_data" &&
      PYTHONPATH="$WORK/code/profiling_synth_data/utils" \
      "$PY" after_profile_synth_data.py --dataset_name "$D" --model_name "$M");;
  final)
    D="${1:-fat}"; M="${2:-ctgan}"
    (cd "$WORK/code/profiling_synth_data" &&
      PYTHONPATH="$WORK/code/profiling_synth_data/utils" \
      "$PY" final_data.py --dataset_name "$D" --model_name "$M");;
  evaluate-orig)
    D="${1:-fat}"
    (cd "$WORK/code/evaluation" &&
      PYTHONPATH="$WORK/code/evaluation/utils" \
      "$PY" execute_evaluate.py --data_path "$WORK/datasets/prepared" \
        --dataset_name "$D" --output_path "$WORK/model_status/originais/$D" \
        --performs_path "$WORK/results_evaluation/$D");;
  evaluate-synth)
    D="${1:-fat}"; M="${2:-ctgan}"
    (cd "$WORK/code/evaluation" &&
      PYTHONPATH="$WORK/code/evaluation/utils" \
      "$PY" execute_evaluate_synthetic.py \
        --data_path_synt "$WORK/generated" --dataset_name "$D" \
        --output_path "$WORK/model_status/synthetic/$D" \
        --performs_path "$WORK/results_evaluation/synthetic/$D/$M" \
        --model_name "$M" --data_path_real "$WORK/datasets/prepared" --stage final);;
  pipeline)
    D="${1:-fat}"; M="${2:-ctgan}"
    "$0" params "$D" "$M" && "$0" choosing "$D" && "$0" generate "$D" "$M" \
      && "$0" sculpt "$D" "$M" && "$0" final "$D" "$M";;
  *) sed -n '2,24p' "$0"; exit 1;;
esac
