#!/bin/bash
# Run all VulnLLM-R evaluations: C/UAF x {clean,dpi,context_aware} x policy
# Variants split across GPU 2 (freeitem, dropconn) and GPU 3 (relogger, rmentry) in parallel.

_DATASET_ROOT="${VL_DATASET_PREFIX:-/mnt/ssd/aryawu/redteaming_repoaudit/VulnLLM-R/datasets}"
DATASET_BASE="${_DATASET_ROOT}/C/UAF"
_RESULTS_ROOT="${VL_RESULT_PREFIX:-results}"
RESULTS_BASE="/mnt/ssd/aryawu/redteaming_repoaudit/VulnLLM-R/${_RESULTS_ROOT}/C/UAF"
MODEL="UCSB-SURFI/VulnLLM-R-7B"

VARIANTS_GPU2="freeitem dropconn"
VARIANTS_GPU3="relogger rmentry"
CATEGORIES="${CATEGORIES:-safe buggy context_aware}"

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
                >> "/tmp/vulnllm_uaf_run_policy_${CATEGORY}_${VARIANT}.log" 2>&1

            echo "Done: GPU${GPU} policy/${CATEGORY}/${VARIANT}"
        done
    done
}

if [ -n "${SINGLE_GPU:-}" ]; then
    run_variants "$SINGLE_GPU" "freeitem dropconn relogger rmentry"
    echo "All UAF runs complete."
    exit 0
fi

# Run both GPU groups in parallel
run_variants 2 "$VARIANTS_GPU2" &
PID2=$!
run_variants 3 "$VARIANTS_GPU3" &
PID3=$!

wait $PID2
wait $PID3

echo "All UAF runs complete."
