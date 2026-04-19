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

"""Example evaluator hook for AgentWorldModel multi-turn benchmark.

Usage::

    strands-env eval run --evaluator examples.eval.agentworldmodelMultiTurn.awm_evaluator \
        --env examples.eval.agentworldmodelMultiTurn.awm_env \
        --backend sglang \
        --n-samples 3

Requires ``AWM_DATA_PATH`` environment variable pointing to a single JSONL file
(e.g., ``data/awm_multiturn.jsonl``).
"""

import json
import logging
import os
import re
from collections.abc import Iterable
from functools import partial

from strands_env.core import Action, TaskContext
from strands_env.eval import Evaluator
from strands_env.eval.metrics import compute_pass_at_k

logger = logging.getLogger(__name__)


def _normalize_scenario_name(scenario: str) -> str:
    """Normalize scenario name to match database filenames."""
    s = scenario.lower()
    s = re.sub(r"[^a-z0-9_]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_").strip()
    return s


class AgentWorldModelMultiTurnEvaluator(Evaluator):
    """Evaluator for AgentWorldModel multi-turn agentic benchmark with pass@k metrics."""

    benchmark_name = "awm_multiturn"

    def load_dataset(self) -> Iterable[Action]:
        """Load AWM multi-turn tasks from a single JSONL file.

        Each line contains a self-contained task with prompt, metadata (scenario,
        task_idx, db_schema, sample_data, full_code, verify_code, user_prompt).
        Requires ``AWM_DATA_PATH`` environment variable.
        """
        data_path = os.environ.get("AWM_DATA_PATH")
        if not data_path:
            raise RuntimeError("AWM_DATA_PATH environment variable is required (path to JSONL dataset file)")

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset file not found: {data_path}")

        count = 0
        with open(data_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                metadata = entry["metadata"]
                scenario = metadata["scenario"]
                task_idx = metadata["task_idx"]
                normalized = _normalize_scenario_name(scenario)

                yield Action(
                    message=entry["prompt"],
                    task_context=TaskContext(
                        id=f"awm_mt_{normalized}_{task_idx}",
                        scenario=scenario,
                        task_idx=task_idx,
                        verify_code=metadata["verify_code"],
                        full_code=metadata["full_code"],
                        sample_data=metadata["sample_data"],
                        db_schema=metadata["db_schema"],
                        user_prompt=metadata["user_prompt"],
                        data_path=data_path,
                    ),
                )
                count += 1

        logger.info("Loaded %d AWM multi-turn tasks from %s", count, data_path)

    def get_metric_fns(self):
        """Return pass@k metrics plus average reward."""
        return [
            partial(
                compute_pass_at_k,
                k_values=list(range(1, self.n_samples_per_prompt + 1)),
                reward_threshold=1.0,
            ),
            self.compute_average_reward,
            self.compute_completion_rate,
        ]

    def compute_average_reward(self, results: dict) -> dict:
        """Average reward across all non-aborted samples."""
        total_reward = 0.0
        count = 0
        for samples in results.values():
            for sample in samples:
                if sample.step_result.reward:
                    total_reward += sample.step_result.reward.reward
                    count += 1
        avg = total_reward / count if count > 0 else 0.0
        return {"avg_reward": avg}

    def compute_completion_rate(self, results: dict) -> dict:
        """Fraction of samples where the agent completed the task (reward=1.0)."""
        completed = 0
        total = 0
        for samples in results.values():
            for sample in samples:
                total += 1
                if sample.step_result.reward and sample.step_result.reward.reward == 1.0:
                    completed += 1
        rate = completed / total if total > 0 else 0.0
        return {"completion_rate": rate}


# Required export - the CLI looks for this
EvaluatorClass = AgentWorldModelMultiTurnEvaluator
