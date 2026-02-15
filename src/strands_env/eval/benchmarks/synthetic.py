"""SyntheticEvaluator for AWM-format (AgentWorldModel) benchmarks."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

from typing_extensions import override

from strands_env.core import Action, TaskContext
from strands_env.environments.synthetic_env.data_loader import AWMDataLoader

from ..evaluator import AsyncEnvFactory, Evaluator
from ..registry import register_eval

logger = logging.getLogger(__name__)


@register_eval("synthetic")
class SyntheticEvaluator(Evaluator):
    """Evaluator for AWM-format synthetic agentic environments."""

    benchmark_name: str = "synthetic"

    def __init__(
        self,
        env_factory: AsyncEnvFactory,
        *,
        data_dir: Path | str,
        scenarios: list[str] | None = None,
        max_tasks_per_scenario: int | None = None,
        max_concurrency: int = 10,
        n_samples_per_prompt: int = 1,
        output_path: Path | str = Path.cwd() / "results.jsonl",
        save_interval: int = 10,
        keep_tokens: bool = False,
    ):
        super().__init__(
            env_factory=env_factory,
            max_concurrency=max_concurrency,
            n_samples_per_prompt=n_samples_per_prompt,
            output_path=output_path,
            save_interval=save_interval,
            keep_tokens=keep_tokens,
        )
        self.data_dir = Path(data_dir)
        self.scenarios = scenarios
        self.max_tasks_per_scenario = max_tasks_per_scenario

    @override
    def load_dataset(self) -> Iterable[Action]:
        """Load AWM scenarios and yield one Action per scenario+task pair."""
        loader = AWMDataLoader(self.data_dir)
        scenario_names = self.scenarios if self.scenarios is not None else loader.list_scenarios()

        for scenario in scenario_names:
            try:
                tasks = loader.get_tasks(scenario)
            except KeyError:
                logger.warning("Scenario '%s' not found in data, skipping", scenario)
                continue

            limit = self.max_tasks_per_scenario if self.max_tasks_per_scenario is not None else len(tasks)
            for task_idx in range(min(limit, len(tasks))):
                yield Action(
                    message=tasks[task_idx],
                    task_context=TaskContext(
                        id=f"{scenario}_{task_idx}",
                        ground_truth=None,
                        scenario=scenario,
                        task_idx=task_idx,
                    ),
                )
