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
        new_policy = np.ones(len(baseline_policy))
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
    ):
        self.w_deaths = float(w_deaths)
        self.w_cases = float(w_cases)
        self.w_strict = float(w_strict)

        md = AnalysisConfig.metadata
        self.outcome_cols = md.outcome_columns
        self.policy_cols = md.policy_columns

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
        # outcomes
        cases = float(y_pred[self.idx_cases]) if self.idx_cases < len(y_pred) else 0.0
        deaths = float(y_pred[self.idx_deaths]) if self.idx_deaths < len(y_pred) else 0.0

        if self.strict_indices:
            strict_vals = policy_vec[self.strict_indices].astype(float)
            strict_mean = float(np.mean(strict_vals)) / 100.0 
        else:
            strict_mean = 0.0

        # Negative cost, so reward is larger when deaths/cases/strictness are smaller
        cost = (
            self.w_deaths * deaths
            + self.w_cases * cases
            + self.w_strict * strict_mean
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

    def simulate_episode(
        self,
        geo_id: str,
        agent: PolicyAgent,
        start_index: int,
        n_steps: int,
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

        iter = range(start_index, last_index + 1)
    
        for idx in tqdm(iter, desc="Simulating", unit="day"):
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

            # Baseline policy: last day in window
            baseline_policy = (
                window_df[self.policy_cols]
                .tail(1)
                .iloc[0]
                .to_numpy(dtype=float)
            )

            # Agent chooses action (new policy for last day)
            policy_action = agent.act(window_df=window_df, baseline_policy=baseline_policy)

            # Override last policy row and run prediction
            modified_inputs = self.forwarder.apply_policy_override(inputs, policy_action)
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
                print("No y_true!")
                y_true = np.full_like(y_pred, np.nan, dtype=float)

            reward_sim = self.reward_fn(y_pred=y_pred, policy_vec=policy_action)
            if np.all(np.isnan(y_true)):
                reward_actual = np.nan
            else:
                reward_actual = self.reward_fn(y_pred=y_true, policy_vec=baseline_policy)

            steps.append(
                StepResult(
                    date=pred_date,
                    y_true=y_true,
                    y_pred=y_pred,
                    policy_baseline=baseline_policy,
                    policy_action=policy_action,
                    reward_simulated=reward_sim,
                    reward_actual=reward_actual
                )
            )

        return EpisodeResult(geo_id=geo_id, steps=steps)
