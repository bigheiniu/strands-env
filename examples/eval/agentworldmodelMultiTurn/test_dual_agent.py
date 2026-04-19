#!/usr/bin/env python3
"""Quick smoke test for DualAgentWorldModelEnvironment with Bedrock Sonnet 4.5.

Usage:
    AWM_DATA_PATH=/path/to/awm_path_multiturn_test_04112026.jsonl \
        python examples/eval/agentworldmodelMultiTurn/test_dual_agent.py
"""

import asyncio
import json
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
REGION = "us-west-2"
MAX_TURNS = 5
MAX_TOOL_ITERS = 15
MAX_TOOL_CALLS = 30
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

    # Build model factories (both assistant and user use the same Bedrock model)
    boto_session = get_session(region=REGION)
    sampling_params = {"max_new_tokens": 16384}

    assistant_factory = bedrock_model_factory(
        model_id=MODEL_ID,
        boto_session=boto_session,
        sampling_params=sampling_params,
    )
    user_factory = bedrock_model_factory(
        model_id=MODEL_ID,
        boto_session=boto_session,
        sampling_params=sampling_params,
    )

    # Build env factory
    env_factory_fn = create_env_factory(
        model_factory=assistant_factory,
        model_factory_user=user_factory,
        max_turns=MAX_TURNS,
        max_tool_iters=MAX_TOOL_ITERS,
        max_tool_calls=MAX_TOOL_CALLS,
    )

    # Load dataset
    evaluator = EvaluatorClass(env_factory=env_factory_fn)
    actions = list(evaluator.load_dataset())[:MAX_SAMPLES]
    logger.info("Testing with %d sample(s)", len(actions))

    for i, action in enumerate(actions):
        logger.info(
            "=== Sample %d/%d: %s (task_idx=%s) ===",
            i + 1, len(actions), action.task_context.scenario, action.task_context.task_idx,
        )
        logger.info("Prompt: %s", action.message[:200])
        logger.info("User prompt: %s", str(action.task_context.user_prompt)[:200])

        env = await env_factory_fn(action)
        try:
            await env.reset()
            logger.info("Server started, %d MCP tools available", len(env.get_tools()))

            result = await env.step(action)

            logger.info("--- Result ---")
            logger.info("Termination: %s", result.termination_reason)
            logger.info("Reward: %s", result.reward)
            logger.info("Metrics: %s", json.dumps(result.observation.metrics, indent=2, default=str))
            logger.info("Messages: %d total", len(result.observation.messages))

            # Print conversation flow
            for j, msg in enumerate(result.observation.messages):
                role = msg.get("role", "?")
                content = msg.get("content", [])
                if isinstance(content, list):
                    texts = [b.get("text", "")[:150] for b in content if isinstance(b, dict) and "text" in b]
                    tool_uses = [b.get("toolUse", {}).get("name", "") for b in content if isinstance(b, dict) and "toolUse" in b]
                    if texts:
                        logger.info("  [%d] %s: %s", j, role, texts[0])
                    if tool_uses:
                        logger.info("  [%d] %s: tool_calls=%s", j, role, tool_uses)
                elif isinstance(content, str):
                    logger.info("  [%d] %s: %s", j, role, content[:150])
        finally:
            await env.cleanup()
            logger.info("Cleanup done")

    logger.info("=== Test complete ===")


if __name__ == "__main__":
    asyncio.run(main())
