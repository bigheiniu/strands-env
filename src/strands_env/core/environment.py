# Copyright 2025 Horizon RL Contributors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Base Environment class for `strands-env`."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar

from strands import Agent
from strands.handlers.callback_handler import PrintingCallbackHandler
from strands.telemetry.metrics import EventLoopMetrics
from strands_sglang import TokenManager, ToolIterationLimiter

from .models import ModelFactory
from .types import (
    Action,
    Observation,
    RewardFunction,
    StepResult,
    TerminationReason,
    TokenObservation,
)

logger = logging.getLogger(__name__)


class Environment:
    """Base RL rollout environment for Strands agents."""

    default_system_prompt_path: ClassVar[Path | None] = None

    def __init__(
        self,
        *,
        model_factory: ModelFactory,
        system_prompt: str | None = None,
        reward_fn: RewardFunction | None = None,
        max_tool_iterations: int = 10,
        verbose: bool = False,
    ):
        self.model_factory = model_factory
        self.reward_fn = reward_fn
        self.max_tool_iterations = max_tool_iterations
        self.verbose = verbose

        path = self.default_system_prompt_path
        self.system_prompt = system_prompt or (path.read_text() if path and path.exists() else None)

    async def reset(self) -> None:
        """Reset for a new episode. Override for environment-specific init."""
        pass

    async def step(self, action: Action) -> StepResult:
        """Run one agent episode and return observation + reward + termination."""
        conversation_history = action.task_context.conversation_history
        tool_limiter = ToolIterationLimiter(self.max_tool_iterations)
        model = self.model_factory()
        model.token_manager = TokenManager()
        agent = Agent(
            model=model,
            messages=list(conversation_history),
            tools=list(self.get_tools()),
            system_prompt=self.system_prompt,
            hooks=[tool_limiter] + list(self.get_hooks()),
            callback_handler=PrintingCallbackHandler() if self.verbose else None,
        )
        error = None
        try:
            await agent.invoke_async(action.message)
        except Exception as e:
            error = e
        termination_reason = TerminationReason.from_error(error)

        step_messages = list(agent.messages)[len(conversation_history) :]
        token_obs = TokenObservation.from_token_manager(agent.model.token_manager)
        tool_parse_errors = getattr(agent.model, "tool_parse_errors", None)
        metrics = {
            "message_count": len(step_messages),
            "tool_iters": tool_limiter.iteration_count,
            **self.compute_metrics(agent.event_loop_metrics, tool_parse_errors=tool_parse_errors),
        }
        observation = Observation(messages=step_messages, tokens=token_obs, metrics=metrics)
        step_result = StepResult(observation=observation, termination_reason=termination_reason)
        step_result.reward = (
            (await self.reward_fn.compute(action=action, step_result=step_result)) if self.reward_fn else None
        )
        return step_result

    async def cleanup(self) -> None:
        """Release resources. Override in subclasses."""
        pass

    def get_tools(self) -> list:
        """Tools available to the agent. Override in subclasses."""
        return []

    def get_hooks(self) -> list:
        """Agent hooks. Override and call `super()` to extend."""
        return []

    def compute_metrics(
        self,
        event_loop_metrics: EventLoopMetrics,
        tool_parse_errors: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        """Extract metrics from the event loop. Override to add custom metrics."""
        usage = event_loop_metrics.accumulated_usage
        metrics_data = event_loop_metrics.accumulated_metrics
        latency_ms = metrics_data.get("latencyMs")

        per_tool_metrics = {
            name: {
                "calls": tm.call_count,
                "successes": tm.success_count,
                "errors": tm.error_count,
                "parse_errors": (tool_parse_errors or {}).get(name, 0),
                "latency_s": round(tm.total_time, 4),
            }
            for name, tm in event_loop_metrics.tool_metrics.items()
        }

        return {
            "model_calls": event_loop_metrics.cycle_count,
            "model_latency_s": round(latency_ms / 1000.0, 4) if latency_ms is not None else None,
            "input_tokens": usage.get("inputTokens"),
            "output_tokens": usage.get("outputTokens"),
            "total_tokens": usage.get("totalTokens"),
            "per_tool_metrics": per_tool_metrics or None,
        }
