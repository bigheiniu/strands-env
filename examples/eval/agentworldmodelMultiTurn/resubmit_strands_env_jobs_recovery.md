# Strands-Env AWM Evaluation on Greenland - Full Lifecycle

Complete guide for submitting, monitoring, diagnosing, and recovering AWM evaluation jobs on Greenland.

## Prerequisites

- AWS CLI configured with access to the S3 results bucket
- `awscurl` installed (for Greenland job submission)
- The `launcher.py` script at `examples/eval/agentworldmodel/launcher.py`
- The Greenland submission script `aws_greenland_common.sh` at `/Volumes/workplace/TauGL/scripts/` or alongside the launcher
- Access to strands-env source code for applying fixes

---

## Phase 1: Prepare Data and Models

### Upload dataset to S3

AWM evaluation data should be partitioned and uploaded to S3. Each partition contains `gen_tasks.jsonl`, `gen_verifier.jsonl`, `gen_envs.jsonl`, and a `databases/` directory.

```bash
# Upload a dataset with multiple partitions
for PART in $(seq 1 50); do
    aws s3 cp /path/to/data/part_${PART} \
        s3://shopqa-users/<user>/data/awm/<dataset>/part_${PART} \
        --recursive --region us-east-1
done
```

### Upload model weights to S3 (for local serving)

```bash
# Upload model from local directory
aws s3 cp /path/to/model s3://shopqa-users/<user>/models/<model-name> \
    --recursive --region us-west-2

# Or use a HuggingFace model ID directly (downloaded at job start)
# Set --model-path to the HF ID, e.g., Qwen/Qwen3-1.7B
```

---

## Phase 2: Submit Fresh Jobs

### Launcher overview

The launcher (`launcher.py`) handles:
1. Packing strands-env source code into a tarball and uploading to S3
2. Building the full shell command (model download, server start, eval run, S3 sync)
3. Submitting the job to Greenland via `aws_greenland_common.sh`

### Running evals on cloud desktop (no GPU, API-based models)

For API-based model evaluation (Bedrock, Kimi), use a cloud desktop as a long-running eval host. The cloud desktop has no GPU but provides a persistent environment for running the eval process.

**Cloud desktop:** `dev-dsk-lyichuan-2a-fad6ea46.us-west-2.amazon.com`

**Step 1: SSH into the cloud desktop**

```bash
ssh dev-dsk-lyichuan-2a-fad6ea46.us-west-2.amazon.com
```

**Step 2: Set up strands-env**

```bash
cd /home/<user>/strands-env  # or clone fresh
```

**Step 3a: Run eval with Bedrock (API-based)**

```bash
# Download AWM data
aws s3 cp s3://shopqa-users/<user>/data/awm/<dataset>/part_1 /tmp/awm_data --recursive --region us-east-1

AWM_DATA_DIR=/tmp/awm_data strands-env eval run \
    --evaluator examples.eval.agentworldmodel.awm_evaluator \
    --env examples.eval.agentworldmodel.awm_env \
    --backend bedrock \
    --model-id bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0 \
    --region us-west-2 \
    --max-tokens 16384 \
    --n-samples-per-prompt 8 \
    --max-concurrency 10 \
    --save-interval 10 \
    --env-config '{"max_tool_iters": 15, "max_tool_calls": 30}' \
    -o /tmp/awm_results/
```

**Step 4: Sync results to S3**

```bash
aws s3 cp /tmp/awm_results/ s3://shopqa-users/<user>/awm_results/<dataset>/<model>/part_1/ \
    --recursive --region us-east-1
```

### Single job submission (Greenland)

```bash
python examples/eval/agentworldmodel/launcher.py \
    greenland \
    --job-name "<user>-<model>-<dataset>_part1" \
    --output-path "s3://shopqa-users/<user>/awm_results/<dataset>/<model>/part_1/" \
    --awm-data-s3 "s3://shopqa-users/<user>/data/awm/<dataset>/part_1" \
    --artifact-path s3://shopqa-users/<user>/artifacts/ \
    --model-id hosted_vllm/local-model \
    --backend sglang \
    --model-path s3://shopqa-users/<user>/models/<model> \
    --serving-framework sglang \
    --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.7 \
    --max-tokens 16384 \
    --context-length 40960 \
    --temperature 0.6 \
    --top-p 0.95 \
    --top-k 20 \
    --max-tool-iters 15 \
    --max-tool-calls 30 \
    --n-samples-per-prompt 8 \
    --max-concurrency 10 \
    --save-interval 10 \
    --initiative-id SFAI-shared-p5
```

### Batch job submission (multiple parts)

```bash
#!/bin/bash
# Submit N parts for a single model/dataset combination
MODEL="Qwen3-1.7B"
MODEL_S3="s3://shopqa-users/<user>/models/Qwen3-1.7B"
DATASET="sonnet45"
DATA_S3_BASE="s3://shopqa-users/<user>/data/awm/sonnet45_default_0325"
OUTPUT_BASE="s3://shopqa-users/<user>/awm_results/sonnet45/Qwen3-1.7B"

for PART in $(seq 1 50); do
    echo "=== Submitting part_${PART} ==="
    python examples/eval/agentworldmodel/launcher.py \
        greenland \
        --job-name "<user>-qwen_1-7b-${DATASET}_part${PART}" \
        --output-path "${OUTPUT_BASE}/part_${PART}/" \
        --awm-data-s3 "${DATA_S3_BASE}/part_${PART}" \
        --artifact-path s3://shopqa-users/<user>/artifacts/ \
        --model-id hosted_vllm/local-model \
        --backend sglang \
        --model-path "${MODEL_S3}" \
        --serving-framework sglang \
        --tensor-parallel-size 8 \
        --gpu-memory-utilization 0.7 \
        --max-tokens 16384 \
        --context-length 40960 \
        --temperature 0.6 --top-p 0.95 --top-k 20 \
        --max-tool-iters 15 --max-tool-calls 30 \
        --n-samples-per-prompt 8 \
        --max-concurrency 10 --save-interval 10 \
        --initiative-id SFAI-shared-p5
done
```

### Multi-model batch submission

```bash
#!/bin/bash
# Submit jobs for multiple models x datasets x parts

MODELS=("Qwen3-1.7B" "Qwen3-8B" "Qwen3-14B")
MODEL_SUFFIXES=("1-7b" "8b" "14b")
DATASETS=("sonnet45" "original1000")
DATA_PATHS=("sonnet45_default_0325" "original1000")

for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    SUFFIX="${MODEL_SUFFIXES[$i]}"
    for j in "${!DATASETS[@]}"; do
        DATASET="${DATASETS[$j]}"
        DATA_PATH="${DATA_PATHS[$j]}"
        for PART in $(seq 1 5); do
            echo "=== ${MODEL} / ${DATASET} / part_${PART} ==="
            python examples/eval/agentworldmodel/launcher.py \
                greenland \
                --job-name "<user>-qwen_${SUFFIX}-${DATASET}_part${PART}" \
                --output-path "s3://shopqa-users/<user>/awm_results/${DATASET}/${MODEL}/part_${PART}/" \
                --awm-data-s3 "s3://shopqa-users/<user>/data/awm/${DATA_PATH}/part_${PART}" \
                --artifact-path s3://shopqa-users/<user>/artifacts/ \
                --model-id hosted_vllm/local-model \
                --backend sglang \
                --model-path "s3://shopqa-users/<user>/models/${MODEL}" \
                --serving-framework sglang \
                --tensor-parallel-size 8 \
                --gpu-memory-utilization 0.7 \
                --max-tokens 16384 \
                --context-length 40960 \
                --temperature 0.6 --top-p 0.95 --top-k 20 \
                --max-tool-iters 15 --max-tool-calls 30 \
                --n-samples-per-prompt 8 \
                --max-concurrency 10 --save-interval 10 \
                --initiative-id SFAI-shared-p5
        done
    done
done
```

### Bedrock API mode (no local serving)

```bash
python examples/eval/agentworldmodel/launcher.py \
    greenland \
    --job-name "<user>-sonnet-awm" \
    --output-path "s3://shopqa-users/<user>/awm_results/<dataset>/sonnet/" \
    --awm-data-s3 "s3://shopqa-users/<user>/data/awm/<dataset>/part_1" \
    --artifact-path s3://shopqa-users/<user>/artifacts/ \
    --model-id bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0 \
    --backend bedrock \
    --aws-region us-west-2 \
    --max-tokens 16384 \
    --n-samples-per-prompt 8 \
    --max-concurrency 10 --save-interval 10 \
    --initiative-id SFAI-shared-p5
```

### Key launcher arguments reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--job-name` | (required) | Job name prefix. Random suffix added for Greenland uniqueness; deterministic output dir uses this as-is. |
| `--output-path` | (required) | S3 destination for results |
| `--awm-data-s3` | (required) | S3 path to AWM dataset partition |
| `--artifact-path` | (required) | S3 path for code snapshot upload |
| `--backend` | `sglang` | Model backend: `sglang`, `bedrock`, or `kimi` |
| `--model-path` | None | S3 URI or HuggingFace ID for local model serving |
| `--model-id` | None | Model ID for the backend. Use `hosted_vllm/local-model` for local SGLang. |
| `--serving-framework` | `sglang` | `sglang` or `vllm` for local serving |
| `--tensor-parallel-size` | 8 | TP size for local model serving |
| `--gpu-memory-utilization` | 0.8 | SGLang `--mem-fraction-static`. Use 0.7 for large-vocab models. |
| `--context-length` | None | Override model's default context length |
| `--max-tokens` | 16384 | Max new tokens per generation |
| `--temperature` | 0.6 | Sampling temperature |
| `--top-p` | 0.95 | Top-p sampling |
| `--top-k` | 20 | Top-k sampling |
| `--n-samples-per-prompt` | 3 | Rollouts per task (for pass@k) |
| `--max-concurrency` | 10 | Concurrent eval episodes |
| `--max-tool-iters` | 15 | Max tool iterations per episode |
| `--max-tool-calls` | 30 | Max tool calls per episode |
| `--save-interval` | 10 | Checkpoint every N samples |
| `--sync-interval` | 60 | S3 sync interval in seconds |
| `--initiative-id` | `SFAI-shared-p5` | Greenland initiative (determines instance type and region) |
| `--resume-from` | None | S3 path to previous results for checkpoint resume |
| `--dry-run` | False | Print commands without submitting |

### Initiative to instance mapping

| Initiative | Instance Type | Region | GPUs |
|-----------|---------------|--------|------|
| `SFAI-shared-p5` | p5.48xlarge | us-west-2 | 8x H100 80GB |
| `SFAI-shared-p4de` | p4de.24xlarge | us-east-1 | 8x A100 80GB |
| `EvaluationP5NodesBOM` | p5.48xlarge | ap-south-1 | 8x H100 80GB |
| `Neo-post-training` | p5.48xlarge | ap-south-1 | 8x H100 80GB |
| `Rufus-post-training` | p5en.48xlarge | us-west-2 | 8x H200 |

### Proven working configs

| Model | TP | mem-fraction | context-length | max-tokens | Instance | Throughput | Notes |
|-------|-----|-------------|----------------|------------|----------|-----------|-------|
| Qwen3-1.7B | 8 | 0.7 | 40960 | 16384 | p5.48xlarge | ~647 tok/s | Safe, no OOM |
| Qwen3-8B | 8 | 0.7 | 40960 | 16384 | p5.48xlarge | ~647 tok/s | Safe, no OOM |
| Qwen3-14B | 8 | 0.7 | 40960 | 16384 | p5.48xlarge | - | Expected safe |

**Configs that cause OOM** (do not use):
- Any Qwen3 model with `mem-fraction-static=0.9` (logits all_gather needs ~4.64 GiB/GPU for vocab_size=151936)
- `max-tokens=30000` with `context-length=40960` (causes context_window_overflow)

### Verify job submission

After submitting, verify the job was accepted:

```bash
# The launcher prints a JSON response with jobId on success
# Example: {"jobId": "b173d3e0-d0ee-41cb-989e-2416071e48c2", "environment": "General"}

# If no jobId appears, check:
# 1. awscurl is installed and configured
# 2. The Greenland submission script path is correct
# 3. The initiative-id and instance-type are valid
```

---

## Phase 3: Monitor Running Jobs

### Check S3 for results

```bash
# List result files for a specific part
PART=1
aws s3 ls "s3://shopqa-users/<user>/awm_results/<dataset>/<model>/part_${PART}/" --recursive --region us-east-1

# Quick progress check across all parts
for PART in $(seq 1 50); do
    COUNT=$(aws s3 cp "s3://shopqa-users/<user>/awm_results/<dataset>/<model>/part_${PART}/<run-dir>/results.jsonl" - --region us-east-1 2>/dev/null | wc -l)
    echo "part_${PART}: ${COUNT} samples"
done
```

### Check eval.log for progress

```bash
aws s3 cp "s3://shopqa-users/<user>/awm_results/<dataset>/<model>/part_${PART}/<run-dir>/eval.log" /tmp/eval_progress.log --region us-east-1
tail -20 /tmp/eval_progress.log
```

### Check SGLang server health

```bash
aws s3 cp "s3://shopqa-users/<user>/awm_results/<dataset>/<model>/part_${PART}/<run-dir>/sglang_server.log" /tmp/sglang_progress.log --region us-east-1
tail -20 /tmp/sglang_progress.log
# Look for throughput: "gen throughput (token/s): XXX"
```

---

## Phase 4: Diagnose Failed Jobs

### Step 4.1: List all result directories

```bash
for PART in $(seq <start> <end>); do
    echo -n "part_${PART}: "
    aws s3 ls "s3://shopqa-users/<user>/awm_results/<dataset>/<model>/part_${PART}/" --region us-east-1 2>/dev/null | head -1 || echo "NO RESULTS"
done
```

### Step 4.2: Download and inspect logs

For each failed job, download **all available logs** before drawing conclusions:

```bash
PART=8
S3_BASE="s3://shopqa-users/<user>/awm_results/<dataset>/<model>/part_${PART}"

# Find the run subdirectory
aws s3 ls "${S3_BASE}/" --region us-east-1

# Download all logs
RUN_DIR="<run-dir>"
aws s3 cp "${S3_BASE}/${RUN_DIR}/eval.log" /tmp/eval_part${PART}.log --region us-east-1 2>/dev/null
aws s3 cp "${S3_BASE}/${RUN_DIR}/sglang_server.log" /tmp/sglang_part${PART}.log --region us-east-1 2>/dev/null
aws s3 cp "${S3_BASE}/${RUN_DIR}/results.jsonl" /tmp/results_part${PART}.jsonl --region us-east-1 2>/dev/null

# Check sample count (expected = n_tasks * n_samples_per_prompt)
wc -l /tmp/results_part${PART}.jsonl
```

### Step 4.3: Diagnose the failure

Read the logs to identify the **root cause**. Different errors require different fixes.

**Check eval.log for Python errors:**
```bash
grep -A 5 "Traceback\|Error\|Exception\|FATAL\|Crashed" /tmp/eval_part${PART}.log
tail -50 /tmp/eval_part${PART}.log
```

**Check sglang_server.log for serving errors:**
```bash
grep -i "OOM\|OutOfMemory\|CUDA\|error\|killed\|Segfault" /tmp/sglang_part${PART}.log
tail -50 /tmp/sglang_part${PART}.log
```

**Check results.jsonl for termination patterns:**
```bash
python3 -c "
import json
from collections import Counter
reasons = Counter()
with open('/tmp/results_part${PART}.jsonl') as f:
    for line in f:
        r = json.loads(line)
        reasons[r.get('termination_reason', 'unknown')] += 1
print(dict(reasons))
"
```

### Step 4.4: Identify error type and propose fix

Based on the log analysis, categorize the failure and determine the appropriate fix:

#### Error: `asyncio.CancelledError`

**Symptoms in logs:**
- `eval.log`: Traceback ending with `asyncio.exceptions.CancelledError`
- `results.jsonl`: Exists with partial results (fewer lines than expected)

**Root cause:** `CancelledError` is a `BaseException` in Python 3.9+, not caught by `except Exception`. When the concurrent tool executor cancels pending MCP tool calls (e.g., on timeout or shutdown), the uncaught error propagates up and crashes the evaluator.

**Fix:** Add `except asyncio.CancelledError` before `except Exception` in `src/strands_env/tools/mcp_tool.py`:
```python
async def stream(self, tool_use, invocation_state, **kwargs):
    try:
        content, status = await self.call_tool(self._mcp_tool.name, tool_use["input"])
    except asyncio.CancelledError:
        content = [ToolResultContent(text="Tool call cancelled")]
        status = "error"
    except Exception as e:
        content = [ToolResultContent(text=f"Tool call failed: {type(e).__name__}: {e}")]
        status = "error"
    yield ToolResultEvent(ToolResult(status=status, toolUseId=tool_use["toolUseId"], content=content))
```

**Resubmit:** With `--resume-from` (partial results exist).

---

#### Error: CUDA Out of Memory (OOM)

**Symptoms in logs:**
- `sglang_server.log`: `torch.OutOfMemoryError: Tried to allocate X GiB` or `CUDA out of memory`
- `eval.log`: Empty or shows "Server failed to start" / connection refused errors
- `results.jsonl`: Missing or empty

**Root cause:** The SGLang server's KV cache allocation (`--mem-fraction-static`) leaves insufficient GPU memory for the logits tensor. For models with large vocabularies (e.g., Qwen3 with vocab_size=151936), the logits all_gather needs ~4.64 GiB per GPU regardless of TP size.

**Fix:** Reduce `--gpu-memory-utilization`:
- `0.9` is too aggressive for large-vocab models on 80GB GPUs
- `0.7` is safe (leaves ~22 GiB free per GPU on H100/A100 80GB)
- `0.8` may work for smaller-vocab models

**Resubmit:** Fresh (no results to resume from, server never started).

---

#### Error: Context Window Overflow

**Symptoms in logs:**
- `eval.log`: Runs normally, no crash
- `results.jsonl`: Full result count, but all/most samples show `context_window_overflow` termination reason

**Root cause:** `max_new_tokens` + input token count exceeds the server's context length. E.g., `max_new_tokens=30000` + ~13K input > 40960 context.

**Fix:** Either:
1. Reduce `--max-tokens` (e.g., from 30000 to 16384)
2. Increase `--context-length` (if the model supports it and GPU memory allows)

**Resubmit:** Fresh (results are all invalid due to overflow, not worth resuming).

---

#### Error: Server Failed to Start (non-OOM)

**Symptoms in logs:**
- `eval.log`: "ERROR: Server failed to start. Last 100 lines of log:" followed by server log excerpt
- `sglang_server.log`: Various errors (model download failure, missing dependencies, config errors)

**Diagnosis:** Read the server log carefully for the specific error:
```bash
grep -i "error\|failed\|exception" /tmp/sglang_part${PART}.log | head -20
```

**Common sub-causes and fixes:**
- **Model download failure** (S3 access denied, network timeout): Check IAM permissions, S3 path correctness
- **Missing CUDA libraries**: Use a Docker image with matching CUDA version
- **Model config error** (unsupported dtype, TP size incompatible): Adjust `--dtype`, `--tensor-parallel-size`
- **Port conflict**: Shouldn't happen on fresh Greenland instances

**Resubmit:** Fresh.

---

#### Error: Job Preempted / Never Scheduled

**Symptoms in logs:**
- S3 directory is completely empty (no files at all)
- No `eval.log`, no `sglang_server.log`, no `results.jsonl`

**Root cause:** The Greenland job was either:
- Never scheduled (capacity issue)
- Preempted before any S3 sync happened
- The instance died before the first sync interval (default 60s)

**Diagnosis:** Check Greenland web UI or CloudWatch for the job status.

**Fix:** No code fix needed. Just resubmit.

**Resubmit:** Fresh.

---

#### Error: Eval Crash with No Logs

**Symptoms in logs:**
- `sglang_server.log`: Server started successfully (shows "Server is ready" or healthy requests)
- `eval.log`: Missing or empty
- `results.jsonl`: Missing or empty

**Root cause:** The eval process crashed before producing any output, and stderr was not captured. This was fixed by adding `2>&1 | tee /workspace/eval.log` to the eval command in `launcher.py`.

**Diagnosis:** If `eval.log` is missing, the launcher code may be outdated. Check that `build_s3_sync_commands()` includes the `tee` for eval output.

**Fix:** Ensure the launcher includes eval log capture:
```python
logged_run_cmd = f"{run_cmd} 2>&1 | tee {WORKDIR}/eval.log"
```

**Resubmit:** Fresh (with updated launcher code).

---

#### Error: Connection/Timeout Errors During Eval

**Symptoms in logs:**
- `eval.log`: Repeated `ConnectionError`, `TimeoutError`, or `httpx.ReadTimeout` for SGLang/MCP calls
- `sglang_server.log`: May show high load, request queue buildup, or crashes
- `results.jsonl`: Partial results

**Root cause:** Possible causes:
- SGLang server crashed mid-eval (check server log for OOM or segfault)
- Too many concurrent requests overwhelming the server
- MCP server timeout or connection drops

**Fix:**
- If server crashed: reduce `--gpu-memory-utilization` or `--max-concurrency`
- If overloaded: reduce `--max-concurrency` (e.g., from 10 to 5)
- If MCP timeout: increase timeout in environment config

**Resubmit:** With `--resume-from` if partial results exist.

---

#### Error: Python Import / Dependency Errors

**Symptoms in logs:**
- `eval.log`: `ModuleNotFoundError`, `ImportError` at startup
- `results.jsonl`: Missing

**Root cause:** Missing Python packages in the Docker image, or the `pip install -e '.[dev]'` step failed.

**Diagnosis:**
```bash
grep -i "ModuleNotFoundError\|ImportError\|No module named" /tmp/eval_part${PART}.log
```

**Fix:** Add missing dependencies to `requirements.txt` or `pyproject.toml`, or use a Docker image that includes them.

**Resubmit:** Fresh (with fixed dependencies in the code snapshot).

---

## Phase 5: Batch Diagnosis and Recovery

### Automated batch diagnosis script

```bash
#!/bin/bash
S3_BASE="s3://shopqa-users/<user>/awm_results/<dataset>/<model>"
TMP_DIR=$(mktemp -d)

NO_RESULTS=()
CANCELLED_ERROR=()
OOM_ERROR=()
CONTEXT_OVERFLOW=()
IMPORT_ERROR=()
OTHER_ERROR=()
OK=()

for PART in $(seq <start> <end>); do
    DIR=$(aws s3 ls "${S3_BASE}/part_${PART}/" --region us-east-1 2>/dev/null | awk '{print $NF}' | head -1)
    if [ -z "$DIR" ]; then
        NO_RESULTS+=($PART)
        continue
    fi

    # Download logs
    aws s3 cp "${S3_BASE}/part_${PART}/${DIR}eval.log" "${TMP_DIR}/eval_${PART}.log" --region us-east-1 2>/dev/null
    aws s3 cp "${S3_BASE}/part_${PART}/${DIR}results.jsonl" "${TMP_DIR}/results_${PART}.jsonl" --region us-east-1 2>/dev/null
    aws s3 cp "${S3_BASE}/part_${PART}/${DIR}sglang_server.log" "${TMP_DIR}/sglang_${PART}.log" --region us-east-1 2>/dev/null

    # Check if results exist
    if [ ! -f "${TMP_DIR}/results_${PART}.jsonl" ] || [ ! -s "${TMP_DIR}/results_${PART}.jsonl" ]; then
        if grep -qi "OutOfMemory\|OOM\|CUDA" "${TMP_DIR}/sglang_${PART}.log" 2>/dev/null; then
            OOM_ERROR+=($PART)
        elif grep -qi "ModuleNotFoundError\|ImportError" "${TMP_DIR}/eval_${PART}.log" 2>/dev/null; then
            IMPORT_ERROR+=($PART)
        else
            NO_RESULTS+=($PART)
        fi
        continue
    fi

    # Results exist - check for errors
    EXPECTED_LINES=160  # Adjust: n_tasks * n_samples_per_prompt
    ACTUAL_LINES=$(wc -l < "${TMP_DIR}/results_${PART}.jsonl")

    if grep -q "CancelledError" "${TMP_DIR}/eval_${PART}.log" 2>/dev/null; then
        CANCELLED_ERROR+=("${PART}(${ACTUAL_LINES}/${EXPECTED_LINES})")
    elif [ "$ACTUAL_LINES" -lt "$EXPECTED_LINES" ]; then
        ERROR=$(grep -oP '(Error|Exception): .*' "${TMP_DIR}/eval_${PART}.log" 2>/dev/null | tail -1)
        OTHER_ERROR+=("${PART}(${ACTUAL_LINES}/${EXPECTED_LINES}: ${ERROR:-unknown})")
    else
        OVERFLOW=$(grep -c "context_window_overflow" "${TMP_DIR}/results_${PART}.jsonl" 2>/dev/null || echo 0)
        if [ "$OVERFLOW" -gt "$((ACTUAL_LINES / 2))" ]; then
            CONTEXT_OVERFLOW+=($PART)
        else
            OK+=($PART)
        fi
    fi
done

echo "=== Diagnosis Summary ==="
echo "OK (${#OK[@]}): ${OK[*]}"
echo "No results (${#NO_RESULTS[@]}): ${NO_RESULTS[*]}"
echo "CancelledError (${#CANCELLED_ERROR[@]}): ${CANCELLED_ERROR[*]}"
echo "OOM (${#OOM_ERROR[@]}): ${OOM_ERROR[*]}"
echo "Context overflow (${#CONTEXT_OVERFLOW[@]}): ${CONTEXT_OVERFLOW[*]}"
echo "Import error (${#IMPORT_ERROR[@]}): ${IMPORT_ERROR[*]}"
echo "Other error (${#OTHER_ERROR[@]}): ${OTHER_ERROR[*]}"

echo ""
echo "=== Recommended Actions ==="
[ ${#CANCELLED_ERROR[@]} -gt 0 ] && echo "CancelledError parts: Fix mcp_tool.py, resubmit with --resume-from"
[ ${#OOM_ERROR[@]} -gt 0 ] && echo "OOM parts: Reduce --gpu-memory-utilization to 0.7, resubmit fresh"
[ ${#CONTEXT_OVERFLOW[@]} -gt 0 ] && echo "Context overflow parts: Reduce --max-tokens or increase --context-length, resubmit fresh"
[ ${#IMPORT_ERROR[@]} -gt 0 ] && echo "Import error parts: Fix dependencies, resubmit fresh"
[ ${#NO_RESULTS[@]} -gt 0 ] && echo "No results parts: Check Greenland job status, resubmit fresh"
[ ${#OTHER_ERROR[@]} -gt 0 ] && echo "Other error parts: Inspect logs manually, then resubmit (fresh or resume depending on partial results)"

rm -rf "$TMP_DIR"
```

### Apply fixes before resubmitting

1. **Code fixes** (e.g., CancelledError, missing imports): Edit the source, then resubmit. The launcher packs and uploads a fresh code snapshot each time.
2. **Config fixes** (e.g., OOM, context overflow): Adjust launcher args. No code change needed.
3. **Infrastructure fixes** (e.g., Docker image, IAM): Fix outside strands-env, then resubmit.

Verify fixes locally if possible before submitting batch jobs.

### Resubmit fresh jobs (no usable previous results)

Use for: OOM, context overflow, import errors, no results, eval crash with no logs.

```bash
for PART in <space-separated list>; do
    python examples/eval/agentworldmodel/launcher.py \
        greenland \
        --job-name "<user>-<model>-<dataset>_part${PART}" \
        --output-path "s3://shopqa-users/<user>/awm_results/<dataset>/<model>/part_${PART}/" \
        --awm-data-s3 "s3://shopqa-users/<user>/data/awm/<dataset>/part_${PART}" \
        --artifact-path s3://shopqa-users/<user>/artifacts/ \
        ... (same args as Phase 2)
done
```

### Resubmit with resume (have usable partial results)

Use for: CancelledError, connection/timeout errors, or any crash that produced valid partial results.

```bash
# Use a shell function to avoid repeating args
submit_resume() {
    local PART=$1
    local RESUME_PATH=$2
    python examples/eval/agentworldmodel/launcher.py \
        greenland \
        --job-name "<user>-<model>-<dataset>_part${PART}" \
        --output-path "s3://shopqa-users/<user>/awm_results/<dataset>/<model>/part_${PART}/" \
        --awm-data-s3 "s3://shopqa-users/<user>/data/awm/<dataset>/part_${PART}" \
        --artifact-path s3://shopqa-users/<user>/artifacts/ \
        ... (same args as Phase 2) \
        --resume-from "${RESUME_PATH}"
}

submit_resume 8  "s3://shopqa-users/<user>/awm_results/<dataset>/<model>/part_8/<old-run-dir>/"
submit_resume 11 "s3://shopqa-users/<user>/awm_results/<dataset>/<model>/part_11/<old-run-dir>/"
# ... etc
```

### How `--resume-from` works

1. The launcher copies `results.jsonl` from the old run directory into the new run's workspace
2. The evaluator's `load_results()` reads the checkpoint, tracking `completed_ids`
3. Already-completed samples are skipped; aborted samples are retried
4. New results are appended to the existing `results.jsonl`

### Key: deterministic output directory

The launcher uses `create_output_dir_name()` (deterministic, no random suffix) for the results directory, while `create_job_name()` adds a random suffix for uniqueness in Greenland. This means:
- Multiple retries of the same `--job-name` write to the same output directory
- The S3 sync destination is stable across retries
- `--resume-from` is only needed when resuming from a *different* job's results (e.g., from before a fix was applied)

---

## Phase 6: Collect and Aggregate Results

### Download all results

```bash
# Download results for all parts
for PART in $(seq 1 50); do
    mkdir -p /tmp/awm_results/part_${PART}
    aws s3 cp "s3://shopqa-users/<user>/awm_results/<dataset>/<model>/part_${PART}/" \
        /tmp/awm_results/part_${PART}/ --recursive --region us-east-1
done
```

### Aggregate results across parts

```bash
# Combine all results.jsonl into one file
cat /tmp/awm_results/part_*/*/results.jsonl > /tmp/awm_results/all_results.jsonl
echo "Total samples: $(wc -l < /tmp/awm_results/all_results.jsonl)"
```

### Compute metrics

```bash
python3 -c "
import json
from collections import Counter

results = []
with open('/tmp/awm_results/all_results.jsonl') as f:
    for line in f:
        results.append(json.loads(line))

# Overall stats
rewards = [r['reward'] for r in results if r.get('reward') is not None]
print(f'Total samples: {len(results)}')
print(f'Mean reward: {sum(rewards)/len(rewards):.4f}')
print(f'Success (reward > 0): {sum(1 for r in rewards if r > 0)}/{len(rewards)}')

# Termination reasons
reasons = Counter(r.get('termination_reason', 'unknown') for r in results)
print(f'Termination: {dict(reasons)}')
"
```

---

## Error Reference

| Error Type | Log Source | Key Pattern | Has Partial Results? | Resubmit Strategy |
|------------|-----------|-------------|---------------------|-------------------|
| `asyncio.CancelledError` | eval.log | `CancelledError` traceback | Yes | Resume |
| CUDA OOM | sglang_server.log | `OutOfMemoryError`, `Tried to allocate X GiB` | No | Fresh (lower mem) |
| Context overflow | results.jsonl | All `context_window_overflow` termination | Yes (but invalid) | Fresh (lower max_tokens) |
| Server failed to start | eval.log | `ERROR: Server failed to start` | No | Fresh (fix config) |
| Job preempted | (none) | Empty S3 directory | No | Fresh |
| Connection/timeout | eval.log | `ConnectionError`, `TimeoutError` | Sometimes | Resume if partial |
| Import error | eval.log | `ModuleNotFoundError` | No | Fresh (fix deps) |
| Eval crash (no log) | (missing eval.log) | Server log healthy, no eval.log | No | Fresh (fix launcher) |
| Model download fail | sglang_server.log | `AccessDenied`, `NoSuchKey` | No | Fresh (fix S3 path/IAM) |
