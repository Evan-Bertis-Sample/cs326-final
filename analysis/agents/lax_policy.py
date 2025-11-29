from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from analysis.rl import PolicyAgent
from analysis.policy_levels import get_laxest_action_for_columns, POLICY_SCALES


class LaxPolicyAgent(PolicyAgent):
    def __init__(self, policy_columns: Sequence[str]) -> None:
        self.policy_columns = tuple(policy_columns)
        # Precompute the strictest action as a vector.
        lax_map = get_laxest_action_for_columns(list(self.policy_columns))

        self._action_vector = np.array(
            [lax_map.get(col, 0) for col in self.policy_columns],
            dtype=np.int64,
        )

    def act(
        self,
        window_df: pd.DataFrame,
        baseline_policy: np.ndarray,
    ) -> np.ndarray:
        if baseline_policy.shape[-1] != len(self.policy_columns):
            raise ValueError(
                f"baseline_policy has shape {baseline_policy.shape}, "
                f"expected last dim = {len(self.policy_columns)}"
            )

        return self._action_vector.copy()
