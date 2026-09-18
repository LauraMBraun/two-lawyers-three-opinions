#!/usr/bin/env bash
# Full pipeline on the dummy corpus: inference -> metrics -> figures.
#   ./run_all.sh hf:Qwen/Qwen3-0.6B     # local model via transformers

set -euo pipefail

MODEL="${1:?usage: ./run_all.sh <hf-repo-id>, e.g. Qwen/Qwen3-0.6B}"
MAX_TOKENS="${MAX_TOKENS:-2048}" # Remove this limit when running wiht actual text data, not dummy corpus to avoid endless thinking loops.

echo "=== inference: $MODEL (reasoning on and off)==="
for THINKING in on off; do
  uv run src/run_inference.py --model "$MODEL" --thinking "$THINKING" \
  --max-tokens "$MAX_TOKENS"
done

echo "=== metrics ==="
uv run src/evaluate.py

echo "=== figures ==="
uv run src/figures.py

echo
echo "results/summary.csv"    
echo "results/*/metrics.json" 
echo "figures/"               
