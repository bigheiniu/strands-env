#!/usr/bin/env python3
"""Test DualAgentWorldModelEnvironment with 8 rollouts using the Evaluator framework.

Usage:
    AWM_DATA_PATH=/path/to/awm_path_multiturn_train_04112026.jsonl \
        python examples/eval/agentworldmodelMultiTurn/test_rollout8.py
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_ID = os.environ.get("MODEL_ID", "us.anthropic.claude-sonnet-4-5-20250929-v1:0")
MODEL_ID_USER = os.environ.get("MODEL_ID_USER", MODEL_ID)
REGION = "us-west-2"
MAX_TURNS = 5
MAX_TOOL_ITERS = 15
MAX_TOOL_CALLS = 30
N_SAMPLES_PER_PROMPT = 8
MAX_CONCURRENCY = 8
MAX_SAMPLES = 1


async def main():
    from strands_env.core.models import bedrock_model_factory
    from strands_env.utils.aws import get_session

    from examples.eval.agentworldmodelMultiTurn.awm_env import create_env_factory
    from examples.eval.agentworldmodelMultiTurn.awm_evaluator import EvaluatorClass

    data_path = os.environ.get("AWM_DATA_PATH")
    if not data_path:
        print("ERROR: Set AWM_DATA_PATH env var")
        sys.exit(1)

    boto_session = get_session(region=REGION)
    sampling_params = {"max_new_tokens": 16384}

    assistant_factory = bedrock_model_factory(
        model_id=MODEL_ID,
        boto_session=boto_session,
        sampling_params=sampling_params,
    )
    user_factory = bedrock_model_factory(
        model_id=MODEL_ID_USER,
        boto_session=boto_session,
        sampling_params=sampling_params,
    )

    env_factory_fn = create_env_factory(
        model_factory=assistant_factory,
        model_factory_user=user_factory,
        max_turns=MAX_TURNS,
        max_tool_iters=MAX_TOOL_ITERS,
        max_tool_calls=MAX_TOOL_CALLS,
    )

    output_dir = Path("awm_multiturn_rollout8_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluator = EvaluatorClass(
        env_factory=env_factory_fn,
        max_concurrency=MAX_CONCURRENCY,
        n_samples_per_prompt=N_SAMPLES_PER_PROMPT,
        output_path=output_dir / "results.jsonl",
        save_interval=4,
    )

    actions = list(evaluator.load_dataset())[:MAX_SAMPLES]
    logger.info("Running %d rollouts x %d sample(s) = %d total", N_SAMPLES_PER_PROMPT, len(actions), N_SAMPLES_PER_PROMPT * len(actions))

    results = await evaluator.run(actions)
    metrics = evaluator.compute_metrics(results)

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Metrics: %s", json.dumps(metrics, indent=2))
    logger.info("Results saved to %s", output_dir)

    # Summary per rollout
    for prompt_id, samples in results.items():
        logger.info("Prompt: %s", prompt_id)
        for i, s in enumerate(samples):
            reward = s.step_result.reward.reward if s.step_result.reward else None
            turns = s.step_result.observation.metrics.get("turns", "?")
            term = s.step_result.termination_reason.value
            logger.info("  rollout %d: reward=%s turns=%s termination=%s", i, reward, turns, term)


if __name__ == "__main__":
    asyncio.run(main())
