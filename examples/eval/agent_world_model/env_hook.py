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

"""Environment hook file for the synthetic benchmark.

Usage:
    strands-env eval run synthetic --env examples/eval/agent_world_model/env_hook.py --data-dir /path/to/AWM-1K
"""

from __future__ import annotations

from pathlib import Path

from strands_env.cli.config import EnvConfig
from strands_env.core.models import ModelFactory
from strands_env.environments.agent_world_model import SyntheticEnv, SyntheticEnvConfig


def create_env_factory(model_factory: ModelFactory, env_config: EnvConfig):
    """Create an async env_factory for SyntheticEnv.

    The returned factory extracts ``scenario`` and ``task_idx`` from
    the action's TaskContext extra fields (set by SyntheticEvaluator).
    """

    async def env_factory(action):
        ctx = action.task_context
        config = SyntheticEnvConfig(
            scenario=ctx.scenario,
            task_idx=ctx.task_idx,
            data_dir=Path(ctx.data_dir),
        )
        return SyntheticEnv(
            model_factory=model_factory,
            config=config,
            system_prompt=env_config.system_prompt,
            max_tool_iters=env_config.max_tool_iters,
        )

    return env_factory
