# rl.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Dict, Any, Optional
from tqdm import tqdm

import numpy as np
import pandas as pd

from analysis.config import AnalysisConfig
from analysis.oxcgrt_data import OxCGRTData
from analysis.predict import ModelInputs, ModelOutput
from analysis.fwd import ModelForwarder


@dataclass(frozen=True)
class StepResult:
    date: pd.Timestamp
    y_true: np.ndarray  # shape (O,) or NaNs if unavailable
    y_pred: np.ndarray  # shape (O,)
    policy_baseline: np.ndarray  # shape (P,)
    policy_action: np.ndarray  # shape (P,)
    reward_actual: float
    reward_simulated : float


@dataclass(frozen=True)
class EpisodeResult:
    geo_id: str
    steps: List[StepResult]


class PolicyAgent:
    def act(
        self,
        window_df: pd.DataFrame,
        baseline_policy: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError


class RelaxationAgent(PolicyAgent):
    def __init__(self, scale: float = 0.9):
        self.scale = float(scale)

    def act(
        self,
        window_df: pd.DataFrame,
        baseline_policy: np.ndarray,
    ) -> np.ndarray:
        # Simple: scale all policy entries down, but not below 0
        new_policy = baseline_policy.astype(float) * self.scale
        new_policy = np.clip(new_policy, 0.0, None)
        return new_policy 


class RewardFunction:
    """
    Reward that encourages:
      - Lower deaths and cases
      - Less strict policies

    Reward is *higher* when deaths/cases are lower and policy strictness is lower.
    """

    def __init__(
        self,
        w_deaths: float = 1.0,
        w_cases: float = 0.1,
        w_strict: float = 0.05,
        neg_outcome_penalty: float = 1e5
    ):
        self.w_deaths = float(w_deaths)
        self.w_cases = float(w_cases)
        self.w_strict = float(w_strict)

        md = AnalysisConfig.metadata
        self.outcome_cols = md.outcome_columns
        self.policy_cols = md.policy_columns
        self.neg_outcome_penalty = neg_outcome_penalty

        # indices of outcomes
        self.idx_cases = (
            self.outcome_cols.index("ConfirmedCases")
            if "ConfirmedCases" in self.outcome_cols
            else 0
        )
        self.idx_deaths = (
            self.outcome_cols.index("ConfirmedDeaths")
            if "ConfirmedDeaths" in self.outcome_cols
            else min(1, len(self.outcome_cols) - 1)
        )

        # indices of "strictness" policy proxies
        strict_names = [
            "StringencyIndex_Average",
            "GovernmentResponseIndex_Average",
            "ContainmentHealthIndex_Average",
        ]
        self.strict_indices = [
            self.policy_cols.index(n) for n in strict_names if n in self.policy_cols
        ]

    def __call__(self, y_pred: np.ndarray, policy_vec: np.ndarray) -> float:
            y_pred = np.asarray(y_pred, dtype=float)

            neg_mask = y_pred < 0.0
            penalty = 0.0
            if np.any(neg_mask):
                # total magnitude of invalid predictions
                neg_magnitude = float(np.sum(np.abs(y_pred[neg_mask])))
                penalty = self.neg_outcome_penalty * neg_magnitude

                # clip so rest of reward sees valid outcomes
                y_pred = np.maximum(y_pred, 0.0)

            # outcomes (>= 0)
            cases = float(y_pred[self.idx_cases]) if self.idx_cases < len(y_pred) else 0.0
            deaths = float(y_pred[self.idx_deaths]) if self.idx_deaths < len(y_pred) else 0.0

            # strictness term
            if self.strict_indices:
                strict_vals = policy_vec[self.strict_indices].astype(float)
                strict_mean = float(np.mean(strict_vals)) / 100.0
            else:
                strict_mean = 0.0

            # Negative cost -> reward = -cost
            cost = (
                self.w_deaths * deaths
                + self.w_cases * cases
                + self.w_strict * strict_mean
                + penalty
            )
            return -cost


class RLSimulator:
    def __init__(self, data: OxCGRTData, forwarder: ModelForwarder):
        self.data = data
        self.forwarder = forwarder
        self.reward_fn = RewardFunction()

        md = AnalysisConfig.metadata
        self.policy_cols = md.policy_columns
        self.outcome_cols = md.outcome_columns
        self.date_col = md.date_column
        self.geoid_col = getattr(AnalysisConfig.metadata, "geo_id_column", "GeoID")

        # Detect if dayssince2020 accidentally got treated as a "policy" dim
        self._dayssince_idx: int | None = None
        for idx, col in enumerate(self.policy_cols):
            if col.lower() == "dayssince2020":
                self._dayssince_idx = idx
                break

    def simulate_episode(
        self,
        geo_id: str,
        agent: PolicyAgent,
        start_index: int,
        n_steps: int,
        verbose: bool = True,
    ) -> EpisodeResult:
        predictor, hp = self.forwarder.get_predictor_for_geo(geo_id)

        window_size = int(hp.get("window_size", 14))
        horizon = int(hp.get("horizon", 1))  # default to 1 if missing

        geo_df = self.data.get_timeseries(geo_id)
        geo_df = geo_df.reset_index(drop=True)

        if len(geo_df) < window_size + 1:
            raise ValueError(
                f"Not enough rows for geo {geo_id} with window_size={window_size}"
            )

        max_index = len(geo_df) - 1
        last_index = min(start_index + n_steps - 1, max_index)

        steps: List[StepResult] = []

        it = range(start_index, last_index + 1)
        if verbose:
            it = tqdm(it, desc="Simulating", unit="day")

        for idx in it:
            end_index = idx - 1  # last day in the window
            if end_index < window_size - 1:
                continue

            # Build baseline inputs and window
            inputs, window_df = self.forwarder.build_initial_window_inputs(
                geo_id=geo_id,
                window_size=window_size,
                end_index=end_index,
                horizon=horizon,
            )

            # Baseline policy: last day in window, coerced to numeric
            baseline_policy = (
                pd.to_numeric(
                    window_df[self.policy_cols].iloc[-1],
                    errors="coerce",   # handles things like "70-74 yrs"
                )
                .fillna(0.0)
                .to_numpy(dtype=float)
            )

            # Agent chooses action (new policy for last day)
            policy_action = agent.act(
                window_df=window_df,
                baseline_policy=baseline_policy,
            )

            # prevent agent from changing dayssince2020
            if self._dayssince_idx is not None:
                true_day_val = baseline_policy[self._dayssince_idx]
                # Ensure both baseline and action use the *actual* day
                baseline_policy[self._dayssince_idx] = true_day_val
                policy_action[self._dayssince_idx] = true_day_val

            # Override last policy row and run prediction
            modified_inputs = self.forwarder.apply_policy_override(
                inputs,
                policy_action,
            )
            output: ModelOutput = predictor.predict(modified_inputs)

            pred_date = output.pred_date
            y_pred = output.outcomes.astype(float)

            # Actual outcome for the predicted date, if available
            tgt_row = geo_df.loc[geo_df[self.date_col] == pred_date]
            if not tgt_row.empty:
                y_true = (
                    tgt_row[self.outcome_cols]
                    .iloc[0]
                    .to_numpy(dtype=float)
                )
            else:
                if verbose:
                    print("No y_true!")
                y_true = np.full_like(y_pred, np.nan, dtype=float)

            reward_sim = self.reward_fn(y_pred=y_pred, policy_vec=policy_action)
            if np.all(np.isnan(y_true)):
                reward_actual = np.nan
            else:
                reward_actual = self.reward_fn(
                    y_pred=y_true,
                    policy_vec=baseline_policy,
                )

            steps.append(
                StepResult(
                    date=pred_date,
                    y_true=y_true,
                    y_pred=y_pred,
                    policy_baseline=baseline_policy,
                    policy_action=policy_action,
                    reward_simulated=reward_sim,
                    reward_actual=reward_actual,
                )
            )

        return EpisodeResult(geo_id=geo_id, steps=steps)
