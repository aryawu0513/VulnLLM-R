#!/bin/bash
# Run VulnLLM-R on clean MLK baselines only (verify CWE-401 detection).

DATASET_BASE="/mnt/ssd/aryawu/redteaming_repoaudit/VulnLLM-R/datasets/C/MLK"
RESULTS_BASE="/mnt/ssd/aryawu/redteaming_repoaudit/VulnLLM-R/results/C/MLK"
MODEL="UCSB-SURFI/VulnLLM-R-7B"

cd /mnt/ssd/aryawu/redteaming_repoaudit/VulnLLM-R
source .venv/bin/activate

mkdir -p "${RESULTS_BASE}/policy/clean"

for VARIANT in allocbuf getrecord makeconn loadcfg; do
    DATASET_PATH="${DATASET_BASE}/clean/${VARIANT}"
    OUTPUT_DIR="${RESULTS_BASE}/policy/clean"

    echo "===== clean/${VARIANT} ====="
    CUDA_VISIBLE_DEVICES=2 python -m vulscan.test.test \
        --output_dir "$OUTPUT_DIR" \
        --dataset_path "$DATASET_PATH" \
        --language c \
        --model "$MODEL" \
        --use_cot --use_policy \
        --vllm --tp 1 --max_tokens 4096 --save \
        2>&1 | tee "/tmp/vulnllm_mlk_clean_${VARIANT}.log"

    echo "Done: clean/${VARIANT}"
done

echo "Clean baseline runs complete."
