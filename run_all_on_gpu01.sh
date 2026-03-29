#!/bin/bash
# Run ALL VulnLLM-R evaluations sequentially on GPUs 0 and 1.
# Order: annotated_C_NPD → standard_C_NPD → C_UAF → Python_NPD
# Each phase uses GPU 0 for first 2 variants, GPU 1 for last 2 variants (parallel within phase).

set -euo pipefail

MODEL="UCSB-SURFI/VulnLLM-R-7B"

cd /mnt/ssd/aryawu/redteaming_repoaudit/VulnLLM-R
source .venv/bin/activate

run_c_variants() {
    local GPU="$1"
    local VARIANTS="$2"
    local DATASET_BASE="$3"
    local RESULTS_BASE="$4"
    local CATEGORIES="$5"
    local LANGUAGE="$6"

    for CATEGORY in $CATEGORIES; do
        for VARIANT in $VARIANTS; do
            DATASET_PATH="${DATASET_BASE}/${CATEGORY}/${VARIANT}"
            OUTPUT_DIR="${RESULTS_BASE}/policy/${CATEGORY}"
            mkdir -p "$OUTPUT_DIR"

            echo "===== GPU${GPU} ${CATEGORY}/${VARIANT} ====="
            CUDA_VISIBLE_DEVICES=$GPU python -m vulscan.test.test \
                --output_dir "$OUTPUT_DIR" \
                --dataset_path "$DATASET_PATH" \
                --language "$LANGUAGE" \
                --model "$MODEL" \
                --use_cot --use_policy \
                --vllm --tp 1 --max_tokens 4096 --save \
                >> "/tmp/vulnllm_all_gpu01_${CATEGORY}_${VARIANT}.log" 2>&1
            echo "Done: GPU${GPU} ${CATEGORY}/${VARIANT}"
        done
    done
}

run_python_variant() {
    local GPU="$1"
    local V1="$2"
    local V2="$3"
    local DATASET_BASE="$4"
    local RESULTS_BASE="$5"
    local CATEGORIES="$6"

    for CATEGORY in $CATEGORIES; do
        for VARIANT in $V1 $V2; do
            DATASET_PATH="${DATASET_BASE}/${CATEGORY}/${VARIANT}"
            OUTPUT_DIR="${RESULTS_BASE}/policy/${CATEGORY}"
            mkdir -p "$OUTPUT_DIR"
            echo "===== GPU${GPU} ${CATEGORY}/${VARIANT} ====="
            CUDA_VISIBLE_DEVICES=$GPU python -m vulscan.test.test \
                --output_dir "$OUTPUT_DIR" \
                --dataset_path "$DATASET_PATH" \
                --language python \
                --model "$MODEL" \
                --use_cot --use_policy \
                --vllm --tp 1 --max_tokens 4096 --save \
                >> "/tmp/vulnllm_all_gpu01_${CATEGORY}_${VARIANT}.log" 2>&1
            echo "Done: GPU${GPU} ${CATEGORY}/${VARIANT}"
        done
    done
}

# --- Phase 1: Annotated C/NPD ---
echo "==============================="
echo "Phase 1: Annotated C/NPD"
echo "==============================="
DATASET_BASE_ANNOTATED="/mnt/ssd/aryawu/redteaming_repoaudit/VulnLLM-R/datasets/C/NPD"
RESULTS_BASE_ANNOTATED="/mnt/ssd/aryawu/redteaming_repoaudit/VulnLLM-R/results/C/NPD"
ANNOTATED_CATS="annotated_clean annotated_dpi annotated_context_aware"

run_c_variants 0 "creatend mkbuf" "$DATASET_BASE_ANNOTATED" "$RESULTS_BASE_ANNOTATED" "$ANNOTATED_CATS" "c" &
PID0=$!
run_c_variants 1 "findrec allocate" "$DATASET_BASE_ANNOTATED" "$RESULTS_BASE_ANNOTATED" "$ANNOTATED_CATS" "c" &
PID1=$!
wait $PID0; wait $PID1
echo "Phase 1 complete."

# --- Phase 2: Standard C/NPD ---
echo "==============================="
echo "Phase 2: Standard C/NPD"
echo "==============================="
DATASET_BASE_NPD="/mnt/ssd/aryawu/redteaming_repoaudit/VulnLLM-R/datasets/C/NPD"
RESULTS_BASE_NPD="/mnt/ssd/aryawu/redteaming_repoaudit/VulnLLM-R/results/C/NPD"
STD_CATS="clean dpi context_aware"

run_c_variants 0 "creatend mkbuf" "$DATASET_BASE_NPD" "$RESULTS_BASE_NPD" "$STD_CATS" "c" &
PID0=$!
run_c_variants 1 "findrec allocate" "$DATASET_BASE_NPD" "$RESULTS_BASE_NPD" "$STD_CATS" "c" &
PID1=$!
wait $PID0; wait $PID1
echo "Phase 2 complete."

# --- Phase 3: C/UAF ---
echo "==============================="
echo "Phase 3: C/UAF"
echo "==============================="
DATASET_BASE_UAF="/mnt/ssd/aryawu/redteaming_repoaudit/VulnLLM-R/datasets/C/UAF"
RESULTS_BASE_UAF="/mnt/ssd/aryawu/redteaming_repoaudit/VulnLLM-R/results/C/UAF"
UAF_CATS="clean dpi context_aware"

run_c_variants 0 "freeitem dropconn" "$DATASET_BASE_UAF" "$RESULTS_BASE_UAF" "$UAF_CATS" "c" &
PID0=$!
run_c_variants 1 "relogger rmentry" "$DATASET_BASE_UAF" "$RESULTS_BASE_UAF" "$UAF_CATS" "c" &
PID1=$!
wait $PID0; wait $PID1
echo "Phase 3 complete."

# --- Phase 4: Python/NPD ---
echo "==============================="
echo "Phase 4: Python/NPD"
echo "==============================="
DATASET_BASE_PY="/mnt/ssd/aryawu/redteaming_repoaudit/VulnLLM-R/datasets/Python/NPD"
RESULTS_BASE_PY="/mnt/ssd/aryawu/redteaming_repoaudit/VulnLLM-R/results/Python/NPD"
PY_CATS="clean dpi context_aware"

run_python_variant 0 "finduser" "parseitem" "$DATASET_BASE_PY" "$RESULTS_BASE_PY" "$PY_CATS" &
PID0=$!
run_python_variant 1 "makeconn" "loadconf" "$DATASET_BASE_PY" "$RESULTS_BASE_PY" "$PY_CATS" &
PID1=$!
wait $PID0; wait $PID1
echo "Phase 4 complete."

echo "==============================="
echo "ALL VulnLLM-R runs complete."
echo "==============================="
