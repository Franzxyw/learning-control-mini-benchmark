from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .agents import Agent
from .environments import LinearSystemEnv
from .evaluation import EpisodeResult, cumulative, regret


class BenchmarkRunner:
    def __init__(self, env_factory, output_dir: str | Path = "results"):
        self.env_factory = env_factory
        self.output_dir = Path(output_dir)

    def run_episode(self, agent: Agent, *, seed: int, initial_state: np.ndarray) -> EpisodeResult:
        env: LinearSystemEnv = self.env_factory()
        observation, _ = env.reset(seed=seed, initial_state=initial_state)
        agent.reset(seed=seed)

        costs: list[float] = []
        states: list[list[float]] = [observation.tolist()]
        actions: list[list[float]] = []

        while True:
            action = agent.act(observation)
            next_observation, reward, terminated, truncated, info = env.step(action)
            cost = float(info["cost"])

            agent.observe(observation, action, next_observation, reward)
            costs.append(cost)
            actions.append(np.asarray(action, dtype=float).tolist())
            states.append(next_observation.tolist())

            observation = next_observation
            if terminated or truncated:
                break

        return EpisodeResult(
            agent_name=agent.name,
            costs=costs,
            cumulative_costs=cumulative(costs),
            states=states,
            actions=actions,
        )

    def run(self, agents: list[Agent], *, seed: int = 7) -> dict:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        initial_rng = np.random.default_rng(seed)
        initial_state = initial_rng.normal(0.0, 1.0, 2)

        results = [self.run_episode(agent, seed=seed, initial_state=initial_state) for agent in agents]
        reference = next((item for item in results if item.agent_name == "lqr_reference"), results[0])

        payload = {
            "seed": seed,
            "initial_state": initial_state.tolist(),
            "episodes": {},
            "regret_against_lqr_reference": {},
        }

        for result in results:
            payload["episodes"][result.agent_name] = {
                "total_cost": result.total_cost,
                "costs": result.costs,
                "cumulative_costs": result.cumulative_costs,
                "states": result.states,
                "actions": result.actions,
            }
            payload["regret_against_lqr_reference"][result.agent_name] = regret(result.costs, reference.costs)

        self._write_json(payload)
        self._plot_costs(results)
        self._plot_regret(results, reference)
        return payload

    def _write_json(self, payload: dict) -> None:
        with (self.output_dir / "results.json").open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)

    def _plot_costs(self, results: list[EpisodeResult]) -> None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for result in results:
            ax.plot(result.cumulative_costs, label=result.agent_name)
        ax.set_title("Cumulative cost")
        ax.set_xlabel("time step")
        ax.set_ylabel("cumulative cost")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.output_dir / "cost_curve.png", dpi=160)
        plt.close(fig)

    def _plot_regret(self, results: list[EpisodeResult], reference: EpisodeResult) -> None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for result in results:
            if result.agent_name == reference.agent_name:
                continue
            ax.plot(regret(result.costs, reference.costs), label=f"{result.agent_name} vs {reference.agent_name}")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title("Cumulative regret against LQR reference")
        ax.set_xlabel("time step")
        ax.set_ylabel("regret")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.output_dir / "regret_curve.png", dpi=160)
        plt.close(fig)
