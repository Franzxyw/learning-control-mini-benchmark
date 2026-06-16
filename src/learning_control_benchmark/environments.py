from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LinearSystemConfig:
    """Configuration for a discrete-time linear system."""

    A: np.ndarray
    B: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    max_steps: int = 120
    process_noise_std: float = 0.02
    initial_state_std: float = 1.0


class LinearSystemEnv:
    """Small linear dynamical system with a Gymnasium-inspired API."""

    def __init__(self, config: LinearSystemConfig):
        self.config = config
        self.state_dim = config.A.shape[0]
        self.action_dim = config.B.shape[1]
        self._rng = np.random.default_rng()
        self._state = np.zeros(self.state_dim)
        self._step_count = 0

    @classmethod
    def default(cls) -> "LinearSystemEnv":
        """Create a stable but nontrivial second-order control problem."""

        A = np.array([[1.02, 0.10], [0.00, 0.96]], dtype=float)
        B = np.array([[0.00], [0.12]], dtype=float)
        Q = np.diag([1.0, 0.2])
        R = np.array([[0.05]], dtype=float)
        return cls(LinearSystemConfig(A=A, B=B, Q=Q, R=R))

    @property
    def state(self) -> np.ndarray:
        return self._state.copy()

    def reset(self, seed: int | None = None, initial_state: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if initial_state is None:
            self._state = self._rng.normal(0.0, self.config.initial_state_std, self.state_dim)
        else:
            self._state = np.asarray(initial_state, dtype=float).reshape(self.state_dim)

        self._step_count = 0
        return self.state, {"step": self._step_count}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        u = np.asarray(action, dtype=float).reshape(self.action_dim)
        x = self._state

        cost = float(x.T @ self.config.Q @ x + u.T @ self.config.R @ u)
        noise = self._rng.normal(0.0, self.config.process_noise_std, self.state_dim)
        self._state = self.config.A @ x + self.config.B @ u + noise

        self._step_count += 1
        terminated = False
        truncated = self._step_count >= self.config.max_steps
        info = {"cost": cost, "step": self._step_count}
        return self.state, -cost, terminated, truncated, info
