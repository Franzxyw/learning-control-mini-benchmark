from __future__ import annotations

import numpy as np

from .lqr import solve_discrete_lqr


class Agent:
    name = "agent"

    def reset(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def act(self, observation: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def observe(self, observation: np.ndarray, action: np.ndarray, next_observation: np.ndarray, reward: float) -> None:
        return None


class RandomAgent(Agent):
    name = "random"

    def __init__(self, action_dim: int, action_scale: float = 1.0):
        self.action_dim = action_dim
        self.action_scale = action_scale
        self.rng = np.random.default_rng()

    def act(self, observation: np.ndarray) -> np.ndarray:
        return self.rng.normal(0.0, self.action_scale, self.action_dim)


class LQRAgent(Agent):
    name = "lqr_reference"

    def __init__(self, A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray):
        self.K = solve_discrete_lqr(A, B, Q, R)
        self.rng = np.random.default_rng()

    def act(self, observation: np.ndarray) -> np.ndarray:
        return -self.K @ observation


class ActiveLearningAgent(Agent):
    """Simple learning controller with occasional probing actions.

    The agent first collects data with small probing actions. Afterwards it
    estimates A and B by least squares and computes an LQR controller for the
    learned model. Every few steps it probes again to keep collecting data.
    """

    name = "active_learning"

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        Q: np.ndarray,
        R: np.ndarray,
        *,
        warmup_steps: int = 18,
        ridge: float = 1e-3,
        probe_scale: float = 0.9,
        probe_every: int = 12,
        control_clip: float = 4.0,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.Q = Q
        self.R = R
        self.warmup_steps = warmup_steps
        self.ridge = ridge
        self.probe_scale = probe_scale
        self.probe_every = probe_every
        self.control_clip = control_clip
        self.rng = np.random.default_rng()
        self._transitions: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed)
        self._transitions = []

    def act(self, observation: np.ndarray) -> np.ndarray:
        step = len(self._transitions)
        if step < self.warmup_steps or step % self.probe_every == 0:
            return self._probe_action(step)

        try:
            A_hat, B_hat = self._estimate_model()
            K = solve_discrete_lqr(A_hat, B_hat, self.Q, self.R)
            action = -K @ observation
        except np.linalg.LinAlgError:
            action = self._probe_action(step)

        return np.clip(action, -self.control_clip, self.control_clip)

    def observe(self, observation: np.ndarray, action: np.ndarray, next_observation: np.ndarray, reward: float) -> None:
        self._transitions.append((observation.copy(), action.copy(), next_observation.copy()))

    def _probe_action(self, step: int) -> np.ndarray:
        sign = 1.0 if step % 2 == 0 else -1.0
        noise = self.rng.normal(0.0, 0.15, self.action_dim)
        return sign * self.probe_scale * np.ones(self.action_dim) + noise

    def _estimate_model(self) -> tuple[np.ndarray, np.ndarray]:
        rows = []
        targets = []
        for x, u, next_x in self._transitions:
            rows.append(np.concatenate([x, u]))
            targets.append(next_x)

        Phi = np.asarray(rows)
        Y = np.asarray(targets)
        regularizer = self.ridge * np.eye(Phi.shape[1])
        theta = np.linalg.solve(Phi.T @ Phi + regularizer, Phi.T @ Y)

        A = theta[: self.state_dim, :].T
        B = theta[self.state_dim :, :].T
        return A, B
