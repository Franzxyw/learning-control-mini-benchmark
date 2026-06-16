from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC))

from alesco_benchmark import ActiveLearningAgent, BenchmarkRunner, LQRAgent, LinearSystemEnv, RandomAgent


def make_env() -> LinearSystemEnv:
    return LinearSystemEnv.default()


def main() -> None:
    env = make_env()
    config = env.config

    agents = [
        LQRAgent(config.A, config.B, config.Q, config.R),
        ActiveLearningAgent(env.state_dim, env.action_dim, config.Q, config.R),
        RandomAgent(env.action_dim, action_scale=1.0),
    ]

    runner = BenchmarkRunner(make_env, output_dir=PROJECT_ROOT / "results")
    payload = runner.run(agents, seed=11)

    print("Benchmark complete.")
    for name, episode in payload["episodes"].items():
        print(f"{name:>16}: total_cost = {episode['total_cost']:.3f}")
    print(f"Results written to: {PROJECT_ROOT / 'results'}")


if __name__ == "__main__":
    main()
