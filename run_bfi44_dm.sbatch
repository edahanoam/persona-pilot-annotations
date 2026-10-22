#!/usr/bin/env bash
#SBATCH --job-name=bfi44-dm
#SBATCH --output=/data/user_data/ndahan/logs/bfi44-dm-%j.out
#SBATCH --error=/data/user_data/ndahan/logs/bfi44-dm-%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=preempt
#SBATCH --qos=preempt_cpu_qos
#SBATCH --requeue
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --export=ALL

set -euo pipefail

PROVIDER="${PROVIDER:-openai}"
MODEL="${MODEL:-gpt-6-luna}"
N="${N:-100}"
SEED="${SEED:-42}"
DATASET_DIR="${DATASET_DIR:-data/twin_2k_500_short_personas}"
RESULTS_DIR="${RESULTS_DIR:-results}"

cd "$SLURM_SUBMIT_DIR"
mkdir -p "$RESULTS_DIR"

PYTHON="$SLURM_SUBMIT_DIR/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  echo "Virtual-environment Python is not executable: $PYTHON" >&2
  exit 1
fi

echo "Provider: $PROVIDER"
echo "Model: $MODEL"
echo "Personas: $N"
echo "Persona version: dm"
echo "Seed: $SEED"
echo "Python: $PYTHON"

"$PYTHON" --version
"$PYTHON" -c 'import sys, datasets, openai; print("executable:", sys.executable); print("datasets:", datasets.__version__)'

"$PYTHON" -c 'import sys; from datasets import load_from_disk; d=load_from_disk(sys.argv[1]); assert "persona_demographics_and_qualitative" in d.column_names; assert len(d) >= int(sys.argv[2]); print("dm column validated; rows:", len(d))' "$DATASET_DIR" "$N"

"$PYTHON" run_bfi44_inference.py \
  "$PROVIDER" \
  "$MODEL" \
  "$N" \
  dm \
  --seed "$SEED" \
  --dataset-dir "$DATASET_DIR" \
  --output "$RESULTS_DIR/bfi44_demographics_qualitative_${N}.csv"
