# AgentWorldModel

[AgentWorldModel-1K](https://huggingface.co/datasets/Snowflake/AgentWorldModel-1K) agentic tool-use benchmark using `AgentWorldModelEnvironment`. Each task spawns a per-task FastAPI server with MCP tools backed by a SQLite database, and the agent is verified via execution-based reward.

## Setup

1. **Dependencies** - Install additional requirements:
   ```bash
   pip install -r src/strands_env/environments/agentworldmodel/requirements.txt
   ```
2. **Dataset** - Download the AWM dataset partition (e.g., `outputs/all1000/part_1`) containing `gen_tasks.jsonl`, `gen_verifier.jsonl`, `gen_envs.jsonl`, and `databases/`.
3. **Environment variable** - Set `AWM_DATA_DIR` to the dataset partition directory.

## Files

- `awm_env.py` - Environment hook that creates `AgentWorldModelEnvironment` instances with per-task DB copies
- `awm_evaluator.py` - Custom evaluator that loads tasks from `gen_tasks.jsonl` + `gen_verifier.jsonl` and computes pass@k, average reward, and completion rate

## Usage

```bash
#/home/lyichuan/agent-world-model/outputs/all1000
#
AWM_DATA_DIR=~/path/to/outputs/all1000/part_1 \
strands-env eval run \
    --evaluator examples.eval.agentworldmodel.awm_evaluator \
    --env examples.eval.agentworldmodel.awm_env \
    --backend sglang \
    --base-url http://localhost:30000 \
    --max-tokens 16384 \
    --n-samples-per-prompt 3 \
    --max-concurrency 10 \
    --max-tool-iters 15
```


```bash
                                                                                                                                                                                          

    AWM_DATA_DIR=/home/lyichuan/agent-world-model/outputs/all1000 \
    strands-env eval run \
        --evaluator examples.eval.agentworldmodel.awm_evaluator \
        --env examples.eval.agentworldmodel.awm_env \
        --backend bedrock \
        --model-id us.anthropic.claude-sonnet-4-5-20250929-v1:0 \
        --max-tokens 30000 \
        --n-samples-per-prompt 8 \
        --max-concurrency 10 \
        --env-config '{"max_tool_iters": 15, "max_tool_calls": 30}' \
        -o awm_eval_sonnet
```

 Key parameters:
 - --n-samples-per-prompt 3 — controls K (runs each task 3 times for pass@1, pass@2, pass@3)
 - --max-concurrency 10 — 10 parallel MCP servers/agents
 - --env-config '{"max_tool_iters": 15, "max_tool_calls": 30}' — limits agent tool use per episode
 - AWM_DATA_DIR — points to the dataset partition directory containing gen_tasks.jsonl, gen_verifier.jsonl, gen_envs.jsonl, and databases/

 To change the model, swap --model-id. For example, for Opus:
 --model-id us.anthropic.claude-opus-4-20250514-v1:0

 To increase K for higher pass@k values:
 --n-samples-per-prompt 5   # computes pass@1 through pass@5

See `strands-env eval run --help` for all CLI options.
