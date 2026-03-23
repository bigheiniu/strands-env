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
import logging
import shutil
from datetime import timedelta
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Any, Literal

import httpx
from awm.tools import get_random_available_port
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import TextContent
from mcp.types import Tool as MCPToolDef
from strands.types.tools import ToolResultContent
from typing_extensions import NotRequired, Unpack, override

from strands_env.core.environment import Environment, EnvironmentConfig
from strands_env.core.models import ModelFactory
from strands_env.core.types import RewardFunction
from strands_env.tools.mcp_tool import MCPToolAdapter

from .reward import AgentWorldModelRewardFunction
from .server import start_server, stop_server, write_server_script

logger = logging.getLogger(__name__)


class AgentWorldModelMCPTool(MCPToolAdapter):
    """MCP tool that checks server health before each call."""

    def __init__(
        self,
        mcp_tool: MCPToolDef,
        session: ClientSession,
        *,
        server_proc: BaseProcess | None = None,
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
        """Execute tool via MCP session, failing fast if server has exited."""
        if self._server_proc is not None and not self._server_proc.is_alive():
            raise RuntimeError(f"Server process exited with code {self._server_proc.exitcode}")
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
    """MCP environment backed by an AgentWorldModel FastAPI server.

    Uses ``mp.Process`` for server lifecycle: pipe-based readiness,
    graceful shutdown via SIGTERM.
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
        self._tool_call_timeout = timedelta(seconds=self.config.get("tool_call_timeout", 60))
        self._server_proc: BaseProcess | None = None
        self._exit_stack: contextlib.AsyncExitStack | None = None
        self._tools: list[AgentWorldModelMCPTool] = []

    @override
    async def reset(self) -> None:
        """Start server, open MCP session, discover tools."""
        port = get_random_available_port()
        script = Path(self.config["temp_dir"]) / "server.py"
        await asyncio.to_thread(
            write_server_script,
            script,
            self.config["scenario"],
            Path(self.config["envs_path"]),
            Path(self.config["work_db_path"]),
        )
        self._server_proc = await asyncio.to_thread(start_server, script, port)
        logger.info("Server pid=%d for %s on port %d", self._server_proc.pid, self.config["scenario"], port)

        stack = contextlib.AsyncExitStack()
        try:
            transport = streamable_http_client(
                f"http://localhost:{port}/mcp",
                http_client=self._http_client,
                terminate_on_close=False,  # we stop the server ourselves
            )
            read_stream, write_stream, *_ = await stack.enter_async_context(transport)
            session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
            await session.initialize()

            result = await session.list_tools()
            self._tools = [
                AgentWorldModelMCPTool(tool, session, server_proc=self._server_proc, timeout=self._tool_call_timeout)
                for tool in result.tools
            ]
            logger.info("Listed %d MCP tools", len(self._tools))
        except BaseException:
            await stack.aclose()
            raise

        self._exit_stack = stack

    @override
    def get_tools(self) -> list:
        """Return the MCP tools discovered during ``reset()``."""
        return list(self._tools)

    @override
    async def cleanup(self) -> None:
        """Close MCP session, stop server, remove temp dir."""
        self._tools = []
        if self._exit_stack:
            with contextlib.suppress(Exception):
                await self._exit_stack.aclose()
            self._exit_stack = None
        await asyncio.to_thread(stop_server, self._server_proc)
        self._server_proc = None
        temp_dir = self.config.get("temp_dir")
        if temp_dir:
            await asyncio.to_thread(shutil.rmtree, temp_dir, True)
