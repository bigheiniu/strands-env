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

"""Example evaluator hook for AgentWorldModel benchmark.

Usage::

    strands-env eval run --evaluator examples.eval.agentworldmodel.awm_evaluator \
        --env examples.eval.agentworldmodel.awm_env \
        --backend sglang \
        --n-samples 3

Requires ``AWM_DATA_DIR`` environment variable pointing to a dataset partition
directory (e.g., ``outputs/all1000/part_1``).
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


class AgentWorldModelEvaluator(Evaluator):
    """Evaluator for AgentWorldModel agentic tool-use benchmark with pass@k metrics."""

    benchmark_name = "awm"

    def load_dataset(self) -> Iterable[Action]:
        """Load AWM tasks from local JSONL files.

        Reads ``gen_tasks.jsonl`` for task texts and ``gen_verifier.jsonl``
        for per-task verification code. Requires ``AWM_DATA_DIR`` environment variable.
        """
        data_dir = os.environ.get("AWM_DATA_DIR")
        if not data_dir:
            raise RuntimeError("AWM_DATA_DIR environment variable is required (path to AWM dataset partition)")

        tasks_path = os.path.join(data_dir, "gen_tasks.jsonl")
        verifier_path = os.path.join(data_dir, "gen_verifier.pure_code.jsonl")

        # Try gen_verifier.pure_code.jsonl first, fall back to gen_verifier.jsonl
        if not os.path.exists(verifier_path):
            alt = os.path.join(data_dir, "gen_verifier_code.jsonl")
            if os.path.exists(alt):
                verifier_path = alt
            else:
                raise FileNotFoundError(f"No verifier file found in {data_dir}")

        with open(tasks_path) as f:
            tasks_data = [json.loads(line) for line in f]
        with open(verifier_path) as f:
            verifiers = [json.loads(line) for line in f]

        # Build lookup: (normalized_scenario, task_idx) -> verify_code
        verify_lookup: dict[tuple[str, int], str] = {}
        for v in verifiers:
            key = (_normalize_scenario_name(v["scenario"]), v["task_idx"])
            verify_lookup[key] = v["verification"]["code"]

        count = 0
        for entry in tasks_data:
            scenario = entry["scenario"]
            normalized = _normalize_scenario_name(scenario)
            for task_idx, task_text in enumerate(entry["tasks"]):
                verify_code = verify_lookup.get((normalized, task_idx))
                if verify_code is None:
                    logger.warning("No verifier for %s task %d, skipping", scenario, task_idx)
                    continue
                yield Action(
                    message=task_text,
                    task_context=TaskContext(
                        id=f"awm_{normalized}_{task_idx}",
                        scenario=scenario,
                        task_idx=task_idx,
                        verify_code=verify_code,
                        data_dir=data_dir,
                    ),
                )
                count += 1

        logger.info("Loaded %d AWM tasks from %s", count, data_dir)

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
EvaluatorClass = AgentWorldModelEvaluator
