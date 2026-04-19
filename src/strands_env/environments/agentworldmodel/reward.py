# Copyright 2025-2026 Horizon RL Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reward functions for AgentWorldModel tasks.

Provides two reward strategies:
- `AgentWorldModelRewardFunction`: execution-based verification via ``exec()`` for binary reward.
- `AgentWorldModelLLMJudgeReward`: LLM-as-a-Judge that scores the agent's output against the task instruction.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import traceback
from typing import Any, Literal

from pydantic import BaseModel, Field
from typing_extensions import override

from strands_env.core.types import Action, Observation, RewardFunction, RewardResult, StepResult
from strands_env.rewards.llm_judge_reward import LLMJudgeReward

logger = logging.getLogger(__name__)

OUTCOME_COMPLETED = "COMPLETED"
OUTCOME_AGENT_FAILED = "AGENT_FAILED"
OUTCOME_VERIFY_ERROR = "VERIFY_ERROR"


def _run_verification(verify_code: str, initial_db_path: str, work_db_path: str, final_answer: str) -> dict:
    """Execute verification code in a thread (blocking exec + SQLite I/O)."""
    namespace: dict = {"sqlite3": sqlite3, "json": json}
    exec(verify_code, namespace)  # noqa: S102
    return namespace["verify_task_completion"](
        initial_db_path=initial_db_path,
        final_db_path=work_db_path,
        final_answer=final_answer,
    )


class AgentWorldModelRewardFunction(RewardFunction):
    """Binary reward via execution-based verification."""

    async def compute(self, action: Action, step_result: StepResult) -> RewardResult:
        """Run verification code against the agent's final response."""
        ctx: Any = action.task_context
        final_answer = Observation.get_final_response(step_result.observation.messages) or ""

        try:
            result = await asyncio.to_thread(
                _run_verification,
                ctx.verify_code,
                ctx.initial_db_path,
                ctx.work_db_path,
                final_answer,
            )
        except Exception as e:
            logger.warning("Verification failed for %s task %s: %s", ctx.scenario, ctx.task_idx, e)
            return RewardResult(
                reward=0.0,
                info={
                    "outcome": OUTCOME_VERIFY_ERROR,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "traceback": traceback.format_exception_only(type(e), e)[-1].strip(),
                },
            )

        is_complete = isinstance(result, dict) and result.get("result") == "complete"
        outcome = OUTCOME_COMPLETED if is_complete else OUTCOME_AGENT_FAILED
        logger.info("Verification %s task %d: %s (outcome=%s)", ctx.scenario, ctx.task_idx, result, outcome)
        return RewardResult(
            reward=1.0 if is_complete else 0.0, info={"outcome": outcome, "verification_result": result}
        )


AWM_JUDGE_TEMPLATE = """
You are an expert judge evaluating whether an AI agent successfully completed a task.

## Task Instruction
The agent was given the following task:
```
{instruction}
```

## Agent's Final Response
```
{model_response}
```

## Database State Context
The agent had access to a SQLite database and MCP tools to perform operations.

## Grading Criteria
Evaluate whether the agent's response and actions indicate successful task completion:
- **COMPLETED**: The agent clearly completed the task as instructed. The response indicates all required operations were performed and the agent confirmed success.
- **PARTIAL**: The agent made meaningful progress but did not fully complete the task. Some steps were done but the task is not fully finished, or the agent expressed uncertainty about completion.
- **FAILED**: The agent failed to complete the task, encountered errors it could not resolve, gave up, or produced output that does not address the task.

Grade the agent's performance as one of: "COMPLETED", "PARTIAL", "FAILED".
Just return the grade, with no text around it.
""".strip()


class AgentWorldModelJudgment(BaseModel):
    """Judgment for AgentWorldModel LLM-as-a-Judge reward."""

    grade: Literal["COMPLETED", "PARTIAL", "FAILED"] = Field(
        ...,
        description=(
            "The grade of the agent's task completion. "
            "COMPLETED: task fully done. PARTIAL: meaningful progress but incomplete. FAILED: task not completed."
        ),
    )


class AgentWorldModelLLMJudgeReward(LLMJudgeReward[AgentWorldModelJudgment]):
    """LLM-as-a-Judge reward for AgentWorldModel tasks.

    Evaluates the agent's output against the original task instruction
    and assigns a score: 1.0 for COMPLETED, 0.5 for PARTIAL, 0.0 for FAILED.
    """

    judgment_format = AgentWorldModelJudgment

    @override
    async def get_judge_prompt(self, action: Action, step_result: StepResult) -> str:
        """Format the judge prompt with the task instruction and agent response."""
        instruction = action.message if isinstance(action.message, str) else str(action.message)
        model_response = step_result.observation.final_response or "(no response)"
        return AWM_JUDGE_TEMPLATE.format(instruction=instruction, model_response=model_response)

    @override
    async def get_reward(self, judgment: AgentWorldModelJudgment | str) -> float:
        """Map judgment grade to a scalar reward."""
        if isinstance(judgment, AgentWorldModelJudgment):
            return {"COMPLETED": 1.0, "PARTIAL": 0.5, "FAILED": 0.0}[judgment.grade]
        return self.default_reward
