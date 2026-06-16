from __future__ import annotations

import numpy as np


def solve_discrete_lqr(
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    *,
    max_iter: int = 500,
    tolerance: float = 1e-9,
) -> np.ndarray:
    """Solve the infinite-horizon discrete LQR problem by Riccati iteration."""

    P = Q.copy()
    for _ in range(max_iter):
        regularized = R + B.T @ P @ B
        K = np.linalg.solve(regularized, B.T @ P @ A)
        next_P = Q + A.T @ P @ (A - B @ K)

        if np.linalg.norm(next_P - P, ord="fro") < tolerance:
            P = next_P
            break
        P = next_P

    return np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
