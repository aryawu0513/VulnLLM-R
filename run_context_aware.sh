#!/bin/bash
# Re-run context_aware only, with COT_Tool files now included.
# Variants split: GPU 2 (creatend, mkbuf), GPU 3 (findrec, allocate)

DATASET_BASE="/mnt/ssd/aryawu/redteaming_repoaudit/VulnLLM-R/datasets/C/NPD"
RESULTS_BASE="/mnt/ssd/aryawu/redteaming_repoaudit/VulnLLM-R/results/C/NPD"
MODEL="UCSB-SURFI/VulnLLM-R-7B"

cd /mnt/ssd/aryawu/redteaming_repoaudit/VulnLLM-R
source .venv/bin/activate

run_variants() {
    local GPU="$1"
    local VARIANTS="$2"

    for VARIANT in $VARIANTS; do
        DATASET_PATH="${DATASET_BASE}/context_aware/${VARIANT}"
        OUTPUT_DIR="${RESULTS_BASE}/policy/context_aware"
        mkdir -p "$OUTPUT_DIR"

        echo "===== GPU${GPU} context_aware/${VARIANT} ====="
        CUDA_VISIBLE_DEVICES=$GPU python -m vulscan.test.test \
            --output_dir "$OUTPUT_DIR" \
            --dataset_path "$DATASET_PATH" \
            --language c \
            --model "$MODEL" \
            --use_cot --use_policy \
            --vllm --tp 1 --max_tokens 4096 --save \
            >> "/tmp/vulnllm_context_aware_${VARIANT}.log" 2>&1

        echo "Done: GPU${GPU} context_aware/${VARIANT}"
    done
}

run_variants 2 "creatend mkbuf" &
PID2=$!
run_variants 3 "findrec allocate" &
PID3=$!

wait $PID2
wait $PID3

echo "All context_aware runs complete."
