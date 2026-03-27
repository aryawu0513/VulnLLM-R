#!/bin/bash
# Run VulnLLM-R evaluations: Python/NPD × {clean,dpi,context_aware} × policy
# 4 variants split across 2 GPUs in parallel (2 variants per GPU).

DATASET_BASE="/mnt/ssd/aryawu/redteaming_repoaudit/VulnLLM-R/datasets/Python/NPD"
RESULTS_BASE="/mnt/ssd/aryawu/redteaming_repoaudit/VulnLLM-R/results/Python/NPD"
MODEL="UCSB-SURFI/VulnLLM-R-7B"

CATEGORIES="clean dpi context_aware"

cd /mnt/ssd/aryawu/redteaming_repoaudit/VulnLLM-R
source .venv/bin/activate

run_variant() {
    local GPU="$1"
    local VARIANT="$2"

    for CATEGORY in $CATEGORIES; do
        DATASET_PATH="${DATASET_BASE}/${CATEGORY}/${VARIANT}"
        OUTPUT_DIR="${RESULTS_BASE}/policy/${CATEGORY}"
        mkdir -p "$OUTPUT_DIR"

        echo "===== GPU${GPU} Python policy/${CATEGORY}/${VARIANT} ====="
        CUDA_VISIBLE_DEVICES=$GPU python -m vulscan.test.test \
            --output_dir "$OUTPUT_DIR" \
            --dataset_path "$DATASET_PATH" \
            --language python \
            --model "$MODEL" \
            --use_cot --use_policy \
            --vllm --tp 1 --max_tokens 4096 --save \
            >> "/tmp/vulnllm_python_${CATEGORY}_${VARIANT}.log" 2>&1

        echo "Done: GPU${GPU} Python policy/${CATEGORY}/${VARIANT}"
    done
}

# GPU 2: finduser + parseitem (sequential within GPU)
run_variant_pair() {
    local GPU="$1"
    local V1="$2"
    local V2="$3"
    run_variant "$GPU" "$V1"
    run_variant "$GPU" "$V2"
}

run_variant_pair 2 "finduser" "parseitem" &
PID2=$!
run_variant_pair 3 "makeconn" "loadconf" &
PID3=$!

wait $PID2
wait $PID3

echo "All Python runs complete."
