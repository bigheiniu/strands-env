"""Environment hook for terminal-bench-2 evaluation with `TerminalBenchEnv`."""

from __future__ import annotations

from strands_env.cli.config import EnvConfig
from strands_env.core.models import ModelFactory
from strands_env.core.types import Action
from strands_env.environments.terminal_bench import TerminalBenchEnv


def create_env_factory(model_factory: ModelFactory, env_config: EnvConfig):
    """Create env_factory for TerminalBenchEnv.

    Args:
        model_factory: Model factory provided by CLI.
        env_config: Environment configuration from CLI.

    Returns:
        Async env_factory function.
    """

    async def env_factory(action: Action) -> TerminalBenchEnv:
        """Create a new TerminalBenchEnv with its own Docker container."""
        ctx = action.task_context
        return TerminalBenchEnv(
            model_factory=model_factory,
            config=ctx.config,
            system_prompt=env_config.system_prompt,
            max_tool_iterations=env_config.max_tool_iterations,
            verbose=env_config.verbose,
        )

    return env_factory
