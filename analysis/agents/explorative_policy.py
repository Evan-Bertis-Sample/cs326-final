from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from analysis.rl import PolicyAgent


@dataclass
class ExplorativePolicyAgent(PolicyAgent):
    """
    Agent that leaves all policies as-is except ONE target policy column,
    which it sets to a fixed (integer) level at every decision step.
    """
    policy_columns: tuple[str, ...]
    target_column: str
    level: float

    def __init__(
        self,
        target_column: str,
        level: float,
        policy_columns: Sequence[str],
    ) -> None:
        self.policy_columns = tuple(policy_columns)
        self.target_column = target_column
        self.level = float(level)

        try:
            self._idx = self.policy_columns.index(self.target_column)
        except ValueError as e:
            raise ValueError(
                f"target_column {self.target_column!r} not found in policy_columns"
            ) from e

    def act(
        self,
        window_df: pd.DataFrame,
        baseline_policy: np.ndarray,
    ) -> np.ndarray:
        # Defensive checks that the ordering is consistent
        if baseline_policy.shape[0] != len(self.policy_columns):
            raise ValueError(
                f"baseline_policy length {baseline_policy.shape[0]} does not match "
                f"policy_columns length {len(self.policy_columns)}"
            )

        action = baseline_policy.astype(float).copy()
        action[self._idx] = self.level
        return action


@dataclass
class BaselineAgent(PolicyAgent):
    """
    Agent that does nothing: always returns the baseline policy unchanged.
    Useful as a reference.
    """
    def act(
        self,
        window_df: pd.DataFrame,
        baseline_policy: np.ndarray,
    ) -> np.ndarray:
        return baseline_policy.astype(float).copy()


@dataclass
class SuperPolicyAgent(PolicyAgent):
    """
    Agent that applies multiple fixed policy levels at once – a "super" agent
    that uses the best level found for each policy independently.
    """
    policy_columns: tuple[str, ...]
    fixed_levels: dict[str, float]

    def __init__(self, policy_columns: Sequence[str], fixed_levels: dict[str, float]):
        self.policy_columns = tuple(policy_columns)
        # Copy and coerce to float
        self.fixed_levels = {k: float(v) for k, v in fixed_levels.items()}

        # Precompute column indices for fast application
        self._indices: dict[str, int] = {}
        for col in self.fixed_levels.keys():
            try:
                self._indices[col] = self.policy_columns.index(col)
            except ValueError as e:
                raise ValueError(
                    f"SuperPolicyAgent fixed_levels key {col!r} not in policy_columns"
                ) from e

    def act(
        self,
        window_df: pd.DataFrame,
        baseline_policy: np.ndarray,
    ) -> np.ndarray:
        if baseline_policy.shape[0] != len(self.policy_columns):
            raise ValueError(
                f"baseline_policy length {baseline_policy.shape[0]} does not match "
                f"policy_columns length {len(self.policy_columns)}"
            )

        action = baseline_policy.astype(float).copy()
        for col, lvl in self.fixed_levels.items():
            idx = self._indices[col]
            action[idx] = lvl
        return action
