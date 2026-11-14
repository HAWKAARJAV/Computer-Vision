#!/usr/bin/env bash
# run_experiments.sh — every experiment behind every figure, one step at a time.
#
#   bash run_experiments.sh data      # download + verify datasets, create split files
#   bash run_experiments.sh e1        # PlantVillage ablation (4 variants)        → Fig. E1
#   bash run_experiments.sh e2        # PlantDoc ablation (4 variants)            → Fig. E2
#   bash run_experiments.sh e3        # PlantVillage → PlantDoc zero-shot          → Fig. E3  (needs e1)
#   bash run_experiments.sh e4        # PlantDoc → PlantVillage zero-shot          → Fig. E4  (needs e2)
#   bash run_experiments.sh e5        # PlantDoc training improvements S0–S6      → Fig. E5  (needs e1)
#   bash run_experiments.sh figures   # rebuild all figures + all_results.csv from results/
#   bash run_experiments.sh all       # everything above, in order
#
# Pipeline check on CPU in minutes (random-pixel data, 1 epoch, no ImageNet download):
#   SMOKE=1 bash run_experiments.sh all
#
# Overridable settings (environment variables):
#   PYTHON=python3  EPOCHS=30  BATCH_PV=64  BATCH_PD=32  LR=1e-3  PROPOSED=multiscale_attention
set -euo pipefail
cd "$(dirname "$0")"

STEP="${1:-help}"
VARIANTS=(baseline multiscale attention multiscale_attention)
PROPOSED="${PROPOSED:-multiscale_attention}"

if [[ "${SMOKE:-0}" == "1" ]]; then
  DATA_DIR=data_smoke; RESULTS=results_smoke
  EPOCHS="${EPOCHS:-1}"; BATCH_PV="${BATCH_PV:-8}"; BATCH_PD="${BATCH_PD:-8}"
  EXTRA=(--no_pretrained)
  echo ">>> SMOKE MODE: random-pixel data in $DATA_DIR, results in $RESULTS (numbers are meaningless)"
else
  DATA_DIR=data; RESULTS=runs   # fresh runs; commit summaries to results/ for the paper figures
  EPOCHS="${EPOCHS:-30}"; BATCH_PV="${BATCH_PV:-64}"; BATCH_PD="${BATCH_PD:-32}"
  EXTRA=()
fi
LR="${LR:-1e-3}"
PY="${PYTHON:-$(command -v python3 || command -v python)}"

need() { [[ -f "$1" ]] || { echo "Missing $1 — run step '$2' first."; exit 1; }; }

step_data() {
  if [[ "${SMOKE:-0}" == "1" ]]; then
    [[ -d data_smoke ]] || "$PY" prepare_datasets.py --make_smoke
    "$PY" prepare_datasets.py --verify --data_dir data_smoke
  else
    [[ -d data/PlantVillage ]] || "$PY" prepare_datasets.py --plantvillage
    [[ -d data/PlantDoc ]]     || "$PY" prepare_datasets.py --plantdoc
    "$PY" prepare_datasets.py --verify
  fi
}

step_e1() {
  "$PY" ablation_study.py --dataset plantvillage --data_dir "$DATA_DIR" \
    --results_dir "$RESULTS/plantvillage" --epochs "$EPOCHS" --batch_size "$BATCH_PV" --lr "$LR" ${EXTRA[@]+"${EXTRA[@]}"}
}

step_e2() {
  "$PY" ablation_study.py --dataset plantdoc --data_dir "$DATA_DIR" \
    --results_dir "$RESULTS/plantdoc" --epochs "$EPOCHS" --batch_size "$BATCH_PD" --lr "$LR" ${EXTRA[@]+"${EXTRA[@]}"}
}

cross() {  # $1 source  $2 target  $3 results subfolder holding source checkpoints  $4 step name
  for v in "${VARIANTS[@]}"; do
    ckpt="$RESULTS/$3/$v/best_model.pth"
    need "$ckpt" "$4"
    "$PY" cross_corpus.py --source "$1" --target "$2" --variant "$v" --protocol direct \
      --source_checkpoint "$ckpt" --data_dir "$DATA_DIR" --results_dir "$RESULTS" --batch_size "$BATCH_PD"
  done
}
step_e3() { cross plantvillage plantdoc plantvillage e1; }
step_e4() { cross plantdoc plantvillage plantdoc e2; }

step_e5() {
  ckpt="$RESULTS/plantvillage/$PROPOSED/best_model.pth"
  need "$ckpt" e1
  "$PY" improvement_study.py --dataset plantdoc --variant "$PROPOSED" --pv_checkpoint "$ckpt" \
    --data_dir "$DATA_DIR" --results_dir "$RESULTS" --epochs "$EPOCHS" --batch_size "$BATCH_PD" --lr "$LR" ${EXTRA[@]+"${EXTRA[@]}"}
}

step_figures() { "$PY" make_figures.py --results_dir "$RESULTS"; }

case "$STEP" in
  data) step_data ;;
  e1) step_e1 ;;
  e2) step_e2 ;;
  e3) step_e3 ;;
  e4) step_e4 ;;
  e5) step_e5 ;;
  figures) step_figures ;;
  all) step_data; step_e1; step_e2; step_e3; step_e4; step_e5; step_figures ;;
  *) sed -n '2,19p' "$0"; exit 0 ;;
esac
echo ">>> step '$STEP' finished"
