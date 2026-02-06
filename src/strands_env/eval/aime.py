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

"""AIME (American Invitational Mathematics Examination) evaluator."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Literal

from datasets import load_dataset

from strands_env.core import Action, TaskContext

from .evaluator import Evaluator

logger = logging.getLogger(__name__)

_AIME_DATASETS = {
    "2024": "HuggingFaceH4/aime_2024",
    "2025": "MathArena/aime_2025",
}


class AIMEEvaluator(Evaluator):
    """Evaluator for AIME math competition problems."""

    benchmark_name = "AIME"

    def load_dataset(self, version: Literal["2024", "2025"] = "2024") -> Iterable[Action]:
        """Load AIME dataset from HuggingFace."""
        self.benchmark_name = f"{self.benchmark_name}_{version}"
        dataset = load_dataset(_AIME_DATASETS[version], split="train")

        actions = []
        for i, row in enumerate(dataset):
            problem, answer = row.get("problem"), row.get("answer")
            if problem is None or answer is None:
                logger.warning(f"Row {i}: missing problem/answer, skipped")
                continue
            actions.append(
                Action(
                    message=str(problem),
                    task_context=TaskContext(
                        id=f"{self.benchmark_name}_{row.get('id', i)}",
                        ground_truth=str(answer),
                    ),
                )
            )

        logger.info(f"[{self.benchmark_name}] Loaded {len(actions)}/{len(dataset)} problems")
        return actions
