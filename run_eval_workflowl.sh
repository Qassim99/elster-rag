#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

source venv/bin/activate

python app/evaluation/eval_workflow.py \
  --model qwen/qwen3-32b \
  --dataset app/evaluation/dataset-de.json \
  --output app/evaluation/results/eval_results_qwen3_workflow_de.json
