from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EpisodeResult:
    agent_name: str
    costs: list[float]
    cumulative_costs: list[float]
    states: list[list[float]]
    actions: list[list[float]]

    @property
    def total_cost(self) -> float:
        return float(self.cumulative_costs[-1])


def cumulative(values: list[float]) -> list[float]:
    return np.cumsum(np.asarray(values, dtype=float)).tolist()


def regret(agent_costs: list[float], reference_costs: list[float]) -> list[float]:
    agent_cum = np.cumsum(np.asarray(agent_costs, dtype=float))
    ref_cum = np.cumsum(np.asarray(reference_costs, dtype=float))
    return (agent_cum - ref_cum).tolist()
