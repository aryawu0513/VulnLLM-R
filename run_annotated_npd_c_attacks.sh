#!/bin/bash
# Run VulnLLM-R evaluations: C/NPD annotated × {annotated_clean,annotated_dpi,annotated_context_aware} × policy
# Variants split across GPU 0 (creatend, mkbuf) and GPU 1 (findrec, allocate) in parallel.

DATASET_BASE="/mnt/ssd/aryawu/redteaming_repoaudit/VulnLLM-R/datasets/C/NPD"
RESULTS_BASE="/mnt/ssd/aryawu/redteaming_repoaudit/VulnLLM-R/results/C/NPD"
MODEL="UCSB-SURFI/VulnLLM-R-7B"

VARIANTS_GPU0="creatend mkbuf"
VARIANTS_GPU1="findrec allocate"
CATEGORIES="annotated_clean annotated_dpi annotated_context_aware"

cd /mnt/ssd/aryawu/redteaming_repoaudit/VulnLLM-R
source .venv/bin/activate

run_variants() {
    local GPU="$1"
    local VARIANTS="$2"

    for CATEGORY in $CATEGORIES; do
        for VARIANT in $VARIANTS; do
            DATASET_PATH="${DATASET_BASE}/${CATEGORY}/${VARIANT}"
            OUTPUT_DIR="${RESULTS_BASE}/policy/${CATEGORY}"
            mkdir -p "$OUTPUT_DIR"

            echo "===== GPU${GPU} policy/${CATEGORY}/${VARIANT} ====="
            CUDA_VISIBLE_DEVICES=$GPU python -m vulscan.test.test \
                --output_dir "$OUTPUT_DIR" \
                --dataset_path "$DATASET_PATH" \
                --language c \
                --model "$MODEL" \
                --use_cot --use_policy \
                --vllm --tp 1 --max_tokens 4096 --save \
                >> "/tmp/vulnllm_annotated_${CATEGORY}_${VARIANT}.log" 2>&1

            echo "Done: GPU${GPU} policy/${CATEGORY}/${VARIANT}"
        done
    done
}

run_variants 0 "$VARIANTS_GPU0" &
PID0=$!
run_variants 1 "$VARIANTS_GPU1" &
PID1=$!

wait $PID0
wait $PID1

echo "All annotated runs complete."
