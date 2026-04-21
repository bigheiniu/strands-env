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

"""Environment hook for multi-turn AgentWorldModel evaluation with `DualAgentWorldModelEnvironment`.

Uses a self-contained JSONL dataset (``AWM_DATA_PATH``) where each line includes
inline database schema, sample data, server code, and user-simulator instructions.
A ``DualAgentWorldModelEnvironment`` drives a multi-turn assistant/user-simulator
conversation loop for each task.
"""

import json
import logging
import re
import shutil
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

from strands_env.core.models import ModelFactory
from strands_env.core.types import Action
from strands_env.environments.agentworldmodel import DualAgentWorldModelEnvironment
from strands_env.environments.agentworldmodel.reward import AgentWorldModelRewardFunction

logger = logging.getLogger(__name__)


def _normalize_scenario_name(scenario: str) -> str:
    """Normalize scenario name to match database filenames."""
    s = scenario.lower()
    s = re.sub(r"[^a-z0-9_]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_").strip()
    return s


def _create_database(db_path: Path, db_schema: dict[str, Any], sample_data: dict[str, Any]) -> None:
    """Create and populate a SQLite database from inline schema and sample data.

    Args:
        db_path: Path where the SQLite database file will be created.
        db_schema: Schema dict with a ``tables`` list, each containing ``ddl`` and optional ``indexes``.
        sample_data: Data dict with a ``tables`` list, each containing ``table_name`` and ``insert_statements``.
    """
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    try:
        for table in db_schema["tables"]:
            cursor.execute(table["ddl"])
            for idx_sql in table.get("indexes", []):
                cursor.execute(idx_sql)
        for table in sample_data["tables"]:
            for stmt in table["insert_statements"]:
                try:
                    cursor.execute(stmt)
                except sqlite3.OperationalError as e:
                    logger.warning("Skipping INSERT for %s: %s", table.get("table_name", "?"), e)
        conn.commit()
    finally:
        conn.close()


def create_env_factory(
    model_factory_or_config=None,
    model_factory_user: ModelFactory | None = None,
    model_config: dict | None = None,
    **env_config,
):
    """Create env_factory for `DualAgentWorldModelEnvironment`.

    Can be called in two ways:
      1. Direct: ``create_env_factory(model_factory=..., model_factory_user=...)``
      2. Config-driven (CLI path): ``create_env_factory(model_config_dict, **env_config)``
         Uses ``model_id_user`` in the config to build the user model factory.

    Requires ``AWM_DATA_PATH`` environment variable pointing to a self-contained JSONL
    dataset file where each line contains inline database schema, sample data, server
    code, verification code, and user-simulator instructions.

    Extra ``env_config`` keys (e.g., ``max_turns``, ``user_system_prompt``) are
    forwarded to `DualAgentWorldModelEnvironment`.
    """
    from strands_env.core.models import build_model_factory, build_user_model_factory

    # Support CLI calling convention: create_env_factory(model_config_dict, **env_config)
    if isinstance(model_factory_or_config, dict):
        model_config = model_factory_or_config
        model_factory = None
    else:
        model_factory = model_factory_or_config

    if model_factory is None and model_config is not None:
        model_factory = build_model_factory(model_config)
        model_factory_user = build_user_model_factory(model_config)
    if model_factory is None:
        raise ValueError("Either model_factory or model_config must be provided")
    if model_factory_user is None:
        model_factory_user = model_factory
    reward_fn = AgentWorldModelRewardFunction()

    async def env_factory(action: Action):
        ctx = action.task_context
        scenario = ctx.scenario
        normalized = _normalize_scenario_name(scenario)

        # Create temp dir for this task
        temp_dir = Path(tempfile.mkdtemp(prefix=f"awm_mt_{normalized}_{ctx.task_idx}_"))

        # Create database from inline schema + data
        work_db_path = temp_dir / f"{normalized}.db"
        _create_database(work_db_path, ctx.db_schema, ctx.sample_data)

        # Copy as initial DB (before agent modifies it)
        initial_db_path = temp_dir / f"{normalized}_initial.db"
        shutil.copy2(work_db_path, initial_db_path)

        # Write gen_envs.jsonl for the AWM server
        envs_path = temp_dir / "gen_envs.jsonl"
        envs_path.write_text(json.dumps({"scenario": scenario, "full_code": ctx.full_code}) + "\n")

        # Store paths and verification code in task_context for the reward function
        ctx.initial_db_path = str(initial_db_path)
        ctx.work_db_path = str(work_db_path)

        return DualAgentWorldModelEnvironment(
            model_factory=model_factory,
            model_factory_user=model_factory_user,
            reward_fn=reward_fn,
            scenario=scenario,
            envs_path=str(envs_path),
            work_db_path=str(work_db_path),
            initial_db_path=str(initial_db_path),
            temp_dir=str(temp_dir),
            **env_config,
        )

    return env_factory
