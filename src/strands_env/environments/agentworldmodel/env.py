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

"""AgentWorldModel environment backed by a FastAPI + SQLite server subprocess."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import random
import shutil
import subprocess
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import TextContent
from mcp.types import Tool as MCPToolDef
from strands import Agent
from strands.agent.conversation_manager import NullConversationManager
from strands.handlers.callback_handler import PrintingCallbackHandler
from strands.types.content import Message, Messages
from strands.types.tools import ToolResultContent
from strands_sglang import TokenManager, ToolLimiter
from typing_extensions import NotRequired, Unpack, override

from strands_env.core.environment import Environment, EnvironmentConfig
from strands_env.core.models import ModelFactory
from strands_env.core.types import Observation, RewardFunction, TerminationReason, TokenObservation
from strands_env.tools.mcp_tool import MCPToolAdapter

from ...core import Action, StepResult
from .reward import AgentWorldModelRewardFunction
from .server import kill_server, start_server

logger = logging.getLogger(__name__)



class AgentWorldModelMCPTool(MCPToolAdapter):
    """MCP tool backed by a `ClientSession` (single-server, direct connection).

    If `server_proc` is provided, polls the process before each call
    to fail fast when the server has exited.
    """

    def __init__(
        self,
        mcp_tool: MCPToolDef,
        session: ClientSession,
        *,
        server_proc: subprocess.Popen | None = None,
        timeout: timedelta | None = None,
    ):
        """Initialize an `AgentWorldModelMCPTool` instance."""
        super().__init__(mcp_tool, timeout=timeout)
        self._session = session
        self._server_proc = server_proc

    @override
    async def call_tool(
        self, name: str, args: dict[str, Any]
    ) -> tuple[list[ToolResultContent], Literal["success", "error"]]:
        """Execute tool via MCP session, failing fast if server process has exited."""
        if self._server_proc is not None:
            returncode = self._server_proc.poll()
            if returncode is not None:
                raise RuntimeError(f"Server process exited with code {returncode}")
        # Coerce string values to schema-declared types (XML parsers extract everything as strings)
        properties = self._mcp_tool.inputSchema.get("properties", {})
        for key, value in args.items():
            if isinstance(value, str) and properties.get(key, {}).get("type") in (
                "array",
                "object",
                "integer",
                "number",
                "boolean",
            ):
                try:
                    args[key] = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    pass
        result = await self._session.call_tool(name, args, self._timeout)
        content = [ToolResultContent(text=item.text) for item in result.content if isinstance(item, TextContent)]
        status: Literal["success", "error"] = "error" if result.isError else "success"
        return content, status


class AgentWorldModelConfig(EnvironmentConfig):
    """Serializable configuration for `AgentWorldModelEnvironment`."""

    scenario: str
    envs_path: str
    work_db_path: str
    initial_db_path: str
    temp_dir: str
    tool_call_timeout: NotRequired[int]


class AgentWorldModelEnvironment(Environment):
    """MCP environment backed by an AgentWorldModel FastAPI server subprocess.

    Notes:
        - `reset()` starts a per-task FastAPI server, opens an MCP session,
          and discovers tools.
        - `cleanup()` closes the session, kills the server, and removes the
          temp directory.
    """

    default_system_prompt_path = Path(__file__).parent / "system_prompt.md"

    def __init__(
        self,
        *,
        model_factory: ModelFactory,
        reward_fn: RewardFunction | None = None,
        http_client: httpx.AsyncClient | None = None,
        **config: Unpack[AgentWorldModelConfig],
    ):
        """Initialize an `AgentWorldModelEnvironment` instance."""
        super().__init__(
            model_factory=model_factory,
            reward_fn=reward_fn or AgentWorldModelRewardFunction(),
            **config,  # type: ignore[misc]
        )
        self._http_client = http_client
        self._tool_call_timeout = timedelta(seconds=int(self.config.get("tool_call_timeout", 60)))
        self._scenario: str = str(self.config["scenario"])
        self._envs_path = Path(str(self.config["envs_path"]))
        self._work_db_path = Path(str(self.config["work_db_path"]))
        self._initial_db_path = Path(str(self.config["initial_db_path"]))
        self._temp_dir = Path(str(self.config["temp_dir"]))
        self._server_proc: subprocess.Popen | None = None
        self._exit_stack: contextlib.AsyncExitStack | None = None
        self._tools: list[AgentWorldModelMCPTool] = []

    @override
    async def reset(self) -> None:
        """Start AgentWorldModel server, open MCP session, discover tools."""
        await asyncio.sleep(random.uniform(0, 5))  # stagger concurrent server spawns
        self._server_proc, port = await start_server(
            self._scenario,
            self._envs_path,
            self._work_db_path,
            self._temp_dir,
        )

        # Open MCP session and discover tools
        stack = contextlib.AsyncExitStack()
        try:
            transport = streamable_http_client(
                f"http://localhost:{port}/mcp",
                http_client=self._http_client,
                terminate_on_close=False,  # we kill the server ourselves in cleanup()
            )
            read_stream, write_stream, *_ = await stack.enter_async_context(transport)
            session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
            await session.initialize()

            result = await session.list_tools()
            self._tools = [
                AgentWorldModelMCPTool(
                    tool,
                    session,
                    server_proc=self._server_proc,
                    timeout=self._tool_call_timeout,
                )
                for tool in result.tools
            ]
            logger.info("Listed %d MCP tools", len(self._tools))
        except BaseException:
            await stack.aclose()
            raise

        self._exit_stack = stack

    @override
    def get_tools(self) -> list:
        """Return the MCP tools discovered during `reset()`."""
        return list(self._tools)

    @override
    async def cleanup(self) -> None:
        """Close MCP session/transport, kill server, remove temp dir."""
        self._tools = []
        if self._exit_stack:
            with contextlib.suppress(Exception):
                await self._exit_stack.aclose()
            self._exit_stack = None
        await kill_server(self._server_proc)
        self._server_proc = None
        if self._temp_dir:
            await asyncio.to_thread(shutil.rmtree, self._temp_dir, True)


class DualAgentWorldModelConfig(AgentWorldModelConfig):
    """Serializable configuration for `DualAgentWorldModelEnvironment`."""

    user_system_prompt: NotRequired[str | None]
    max_turns: NotRequired[int]


class DualAgentWorldModelEnvironment(AgentWorldModelEnvironment):
    """Dual-agent AWM environment with assistant + user simulator.

    Each step runs a multi-turn conversation:
      1. Assistant agent processes a user message (may use MCP tools).
      2. User simulator generates a follow-up based on the assistant's response.
      3. Repeat until a stop signal, max turns, or an error.

    The user simulator is a separate LLM that role-plays as a user. It receives
    the assistant's text responses and generates realistic follow-ups, maintaining
    its own conversation history with implicit role flipping (the strands `Agent`
    naturally treats incoming messages as "user" and its own outputs as
    "assistant").
    """

    STOP_TOKENS: frozenset[str] = frozenset({"###STOP###", "###TRANSFER###", "###OUT-OF-SCOPE###"})
    default_user_system_prompt_path = Path(__file__).parent / "user_system_prompt.md"

    def __init__(
        self,
        *,
        model_factory: ModelFactory,
        model_factory_user: ModelFactory,
        reward_fn: RewardFunction | None = None,
        http_client: httpx.AsyncClient | None = None,
        **config: Unpack[DualAgentWorldModelConfig],
    ):
        """Initialize a `DualAgentWorldModelEnvironment` instance."""
        super().__init__(
            model_factory=model_factory,
            reward_fn=reward_fn,
            http_client=http_client,
            **config,  # type: ignore[arg-type]
        )
        self.model_factory_user = model_factory_user
        self._max_turns: int = int(self.config.get("max_turns", 10))
        self._user_system_prompt: str | None = self.config.get("user_system_prompt") or (
            self.default_user_system_prompt_path.read_text().strip()
            if self.default_user_system_prompt_path.exists()
            else None
        )

    @override
    async def step(self, action: Action) -> StepResult:  # type: ignore[override]
        """Run a multi-turn assistant-user conversation episode.

        Args:
            action: Task for the assistant. User-simulator instructions are read from
                ``action.task_context.user_prompt`` if present.
        """
        assistant_conversation_history: Messages = list(action.task_context.conversation_history)
        user_conversation: Messages = []
        all_step_messages: Messages = []

        user_prompt = getattr(action.task_context, "user_prompt", None)
        user_system_prompt = self._build_user_system_prompt(action, user_prompt)
        current_user_msg = action.message if isinstance(action.message, str) else action.message["content"]

        # Seed user-agent history so it knows the initial message it "sent".
        user_conversation.append({"role": "assistant", "content": [{"text": current_user_msg}]})

        termination_reason = TerminationReason.TASK_COMPLETE
        last_tool_limiter: ToolLimiter | None = None
        last_assistant_model = None
        last_event_loop_metrics = None
        total_turns = 0

        for turn in range(self._max_turns):
            total_turns = turn + 1
            logger.info(
                "[turn %d/%d] Assistant processing user message (%d chars)",
                turn + 1,
                self._max_turns,
                len(current_user_msg),
            )

            # ── Assistant turn ──────────────────────────────────────────
            tool_limiter = ToolLimiter(
                max_tool_iters=self.max_tool_iters,
                max_tool_calls=self.max_tool_calls,
                max_parallel_tool_calls=self.max_parallel_tool_calls,
            )
            assistant_model = self.model_factory()
            assistant_agent = Agent(
                model=assistant_model,
                messages=list(assistant_conversation_history),
                tools=list(self.get_tools()),
                system_prompt=self.system_prompt,
                hooks=[tool_limiter] + list(self.get_hooks()),
                conversation_manager=self.get_conversation_manager(),
                callback_handler=PrintingCallbackHandler() if self.verbose else None,
            )

            error: BaseException | None = None
            try:
                await assistant_agent.invoke_async(current_user_msg)
            except (Exception, asyncio.CancelledError) as exc:
                error = exc

            last_tool_limiter = tool_limiter
            last_assistant_model = assistant_model
            last_event_loop_metrics = assistant_agent.event_loop_metrics

            assistant_step_msgs = list(assistant_agent.messages)[len(assistant_conversation_history) :]
            all_step_messages.extend(assistant_step_msgs)
            assistant_conversation_history = list(assistant_agent.messages)

            termination_reason = TerminationReason.from_error(error)
            logger.info(
                "[turn %d/%d] Assistant done: %d msgs, %d tool_calls, termination=%s",
                turn + 1,
                self._max_turns,
                len(assistant_step_msgs),
                tool_limiter.tool_call_count,
                termination_reason.value,
            )
            if termination_reason != TerminationReason.TASK_COMPLETE:
                break

            # Last allowed turn — skip user generation.
            if turn >= self._max_turns - 1:
                break

            # ── User simulator turn ─────────────────────────────────────
            assistant_response = Observation.get_final_response(assistant_step_msgs)
            if not assistant_response:
                break

            user_model = self.model_factory_user()
            user_agent = Agent(
                model=user_model,
                messages=list(user_conversation),
                tools=[],
                system_prompt=user_system_prompt,
                conversation_manager=NullConversationManager(),
                callback_handler=PrintingCallbackHandler() if self.verbose else None,
            )

            try:
                await user_agent.invoke_async(assistant_response)
            except (Exception, asyncio.CancelledError):
                logger.warning("User simulator failed on turn %d, ending conversation", turn)
                break

            user_step_msgs = list(user_agent.messages)[len(user_conversation) :]
            user_conversation = list(user_agent.messages)

            user_response = Observation.get_final_response(user_step_msgs)
            if not user_response:
                logger.info("[turn %d/%d] User simulator produced no response, ending", turn + 1, self._max_turns)
                break
            if any(tok in user_response for tok in self.STOP_TOKENS):
                logger.info("[turn %d/%d] User simulator sent stop token, ending", turn + 1, self._max_turns)
                break
            logger.info(
                "[turn %d/%d] User simulator responded (%d chars)", turn + 1, self._max_turns, len(user_response)
            )

            # Next turn's invoke_async will add this as the user message,
            # and assistant_step_msgs will capture it naturally (no duplicate).
            current_user_msg = user_response

        logger.info(
            "Conversation finished: %d turn(s), %d messages, termination=%s",
            total_turns,
            len(all_step_messages),
            termination_reason.value,
        )

        # ── Build observation ───────────────────────────────────────────
        token_obs = TokenObservation.from_token_manager(
            getattr(last_assistant_model, "token_manager", TokenManager()) if last_assistant_model else TokenManager()
        )
        tool_parse_errors = getattr(last_assistant_model, "tool_parse_errors", None) if last_assistant_model else None
        metrics: dict[str, Any] = {
            "message_count": len(all_step_messages),
            "turns": total_turns,
        }
        if last_tool_limiter is not None:
            metrics.update(
                {
                    "tool_iters": last_tool_limiter.tool_iter_count,
                    "tool_calls": last_tool_limiter.tool_call_count,
                    "cancelled_tool_calls": last_tool_limiter.cancelled_tool_call_count,
                }
            )
        if last_event_loop_metrics is not None:
            metrics.update(self.compute_metrics(last_event_loop_metrics, tool_parse_errors=tool_parse_errors))
        routed_experts = getattr(last_assistant_model, "routed_experts", None) if last_assistant_model else None
        observation = Observation(
            messages=all_step_messages, tokens=token_obs, metrics=metrics, routed_experts=routed_experts
        )

        step_result = StepResult(observation=observation, termination_reason=termination_reason)
        if self.reward_fn:
            try:
                step_result.reward = await self.reward_fn.compute(action=action, step_result=step_result)
            except (Exception, asyncio.CancelledError) as exc:
                logger.warning("Reward computation failed: %s", exc)
                step_result.reward = None

        observation.messages = all_step_messages + [{"role": "user", "content": [{"text": user_response}]}] if user_response is not None else []
        return step_result

    def _build_user_system_prompt(self, action: Action, user_prompt: str) -> str | None:
        """Assemble the user simulator's system prompt from template + per-task instructions."""
        parts: list[str] = []
        if self._user_system_prompt:
            parts.append(self._user_system_prompt)


        parts.append(f"## Your Task Instructions\n{user_prompt}")

        return "\n".join(parts) if parts else None
