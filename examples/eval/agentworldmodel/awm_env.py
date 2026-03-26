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

"""Environment hook for AgentWorldModel evaluation with `AgentWorldModelEnvironment`."""

import logging
import os
import re
import shutil
import tempfile
from pathlib import Path

from strands_env.core.models import ModelFactory
from strands_env.core.types import Action
from strands_env.environments.agentworldmodel import AgentWorldModelEnvironment
from strands_env.environments.agentworldmodel.reward import AgentWorldModelRewardFunction

logger = logging.getLogger(__name__)


def _normalize_scenario_name(scenario: str) -> str:
    """Normalize scenario name to match database filenames."""
    s = scenario.lower()
    s = re.sub(r"[^a-z0-9_]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_").strip()
    return s


def create_env_factory(model_factory: ModelFactory, **env_config):
    """Create env_factory for `AgentWorldModelEnvironment`.

    Requires ``AWM_DATA_DIR`` environment variable pointing to a dataset partition
    directory (e.g., ``outputs/all1000/part_1``) containing ``gen_envs.jsonl``
    and a ``databases/`` subdirectory.

    Extra ``env_config`` keys are forwarded to `AgentWorldModelEnvironment`.
    """
    data_dir = env_config.pop("data_dir", None) or os.environ.get("AWM_DATA_DIR")
    if not data_dir:
        raise RuntimeError("AWM_DATA_DIR env var or data_dir in env_config is required")
    data_dir = Path(data_dir)

    reward_fn = AgentWorldModelRewardFunction()

    async def env_factory(action: Action):
        ctx = action.task_context
        scenario = ctx.scenario
        normalized = _normalize_scenario_name(scenario)

        # Resolve paths
        envs_path = data_dir / "gen_envs.jsonl"
        initial_db_path = data_dir / "databases" / f"{normalized}.db"
        if not initial_db_path.exists():
            raise FileNotFoundError(f"Initial database not found: {initial_db_path}")

        # Create a temp directory with a working copy of the database
        temp_dir = Path(tempfile.mkdtemp(prefix=f"awm_{normalized}_{ctx.task_idx}_"))
        work_db_path = temp_dir / f"{normalized}.db"
        shutil.copy2(initial_db_path, work_db_path)

        # Store DB paths in task_context so the reward function can access them
        ctx.initial_db_path = str(initial_db_path)
        ctx.work_db_path = str(work_db_path)

        return AgentWorldModelEnvironment(
            model_factory=model_factory,
            reward_fn=reward_fn,
            scenario=scenario,
            envs_path=str(envs_path),
            work_db_path=str(work_db_path),
            initial_db_path=str(initial_db_path),
            temp_dir=str(temp_dir),
            **env_config,
        )

    return env_factory
