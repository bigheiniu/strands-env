#!/bin/bash
# Resubmit 17 failed Qwen3-1.7B sonnet45 jobs
# 6 fresh (no results): 6, 32, 39, 48, 49, 50
# 11 resume (CancelledError): 8, 11, 12, 13, 14, 16, 22, 26, 28, 31, 44

set -e
cd /Users/lyichuan/PycharmProjects/strands-env

COMMON_ARGS=(
    greenland
    --artifact-path s3://shopqa-users/lyichuan/artifacts/
    --model-id hosted_vllm/local-model
    --backend sglang
    --model-path s3://shopqa-users/lyichuan/models/Qwen3-1.7B
    --serving-framework sglang
    --tensor-parallel-size 8
    --gpu-memory-utilization 0.7
    --max-tokens 16384
    --context-length 40960
    --temperature 0.6
    --top-p 0.95
    --top-k 20
    --max-tool-iters 15
    --max-tool-calls 30
    --n-samples-per-prompt 8
    --max-concurrency 10
    --save-interval 10
    --initiative-id SFAI-shared-p5
)

# --- 6 fresh jobs (no previous results) ---
for PART in 6 32 39 48 49 50; do
    echo "=== Submitting fresh job for part_${PART} ==="
    python examples/eval/agentworldmodel/launcher.py \
        "${COMMON_ARGS[@]}" \
        --job-name "lyichuan-qwen_1-7b-sonnet45_part${PART}" \
        --output-path "s3://shopqa-users/lyichuan/awm_results/sonnet45/Qwen3-1.7B/part_${PART}/" \
        --awm-data-s3 "s3://shopqa-users/lyichuan/data/awm/sonnet45_default_0325/part_${PART}"
    echo ""
done

# --- 11 resume jobs (CancelledError, have partial results) ---
declare -A RESUME_PATHS
RESUME_PATHS[8]="s3://shopqa-users/lyichuan/awm_results/sonnet45/Qwen3-1.7B/part_8/lyichuan-qwen_1-7b-sonnet45_part8_Cs5pgr3I9r/"
RESUME_PATHS[11]="s3://shopqa-users/lyichuan/awm_results/sonnet45/Qwen3-1.7B/part_11/lyichuan-qwen_1-7b-sonnet45_part11_c2pjuxHPBR/"
RESUME_PATHS[12]="s3://shopqa-users/lyichuan/awm_results/sonnet45/Qwen3-1.7B/part_12/lyichuan-qwen_1-7b-sonnet45_part12_y0pyefTNOi/"
RESUME_PATHS[13]="s3://shopqa-users/lyichuan/awm_results/sonnet45/Qwen3-1.7B/part_13/lyichuan-qwen_1-7b-sonnet45_part13_sPwWHOpJAU/"
RESUME_PATHS[14]="s3://shopqa-users/lyichuan/awm_results/sonnet45/Qwen3-1.7B/part_14/lyichuan-qwen_1-7b-sonnet45_part14_XJ7og3L7xZ/"
RESUME_PATHS[16]="s3://shopqa-users/lyichuan/awm_results/sonnet45/Qwen3-1.7B/part_16/lyichuan-qwen_1-7b-sonnet45_part16_T47UnDESxs/"
RESUME_PATHS[22]="s3://shopqa-users/lyichuan/awm_results/sonnet45/Qwen3-1.7B/part_22/lyichuan-qwen_1-7b-sonnet45_part22_BgnHjkyD9K/"
RESUME_PATHS[26]="s3://shopqa-users/lyichuan/awm_results/sonnet45/Qwen3-1.7B/part_26/lyichuan-qwen_1-7b-sonnet45_part26_De2Uz6vJVk/"
RESUME_PATHS[28]="s3://shopqa-users/lyichuan/awm_results/sonnet45/Qwen3-1.7B/part_28/lyichuan-qwen_1-7b-sonnet45_part28_Od4tIgaHgW/"
RESUME_PATHS[31]="s3://shopqa-users/lyichuan/awm_results/sonnet45/Qwen3-1.7B/part_31/lyichuan-qwen_1-7b-sonnet45_part31_BZ3bNstc7L/"
RESUME_PATHS[44]="s3://shopqa-users/lyichuan/awm_results/sonnet45/Qwen3-1.7B/part_44/lyichuan-qwen_1-7b-sonnet45_part44_qf4h8C11kv/"

for PART in 8 11 12 13 14 16 22 26 28 31 44; do
    echo "=== Submitting resume job for part_${PART} ==="
    python examples/eval/agentworldmodel/launcher.py \
        "${COMMON_ARGS[@]}" \
        --job-name "lyichuan-qwen_1-7b-sonnet45_part${PART}" \
        --output-path "s3://shopqa-users/lyichuan/awm_results/sonnet45/Qwen3-1.7B/part_${PART}/" \
        --awm-data-s3 "s3://shopqa-users/lyichuan/data/awm/sonnet45_default_0325/part_${PART}" \
        --resume-from "${RESUME_PATHS[$PART]}"
    echo ""
done

echo "All 17 jobs submitted!"
