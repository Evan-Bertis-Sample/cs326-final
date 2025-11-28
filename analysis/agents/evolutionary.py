from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed  # NEW

from analysis.rl import RLSimulator, PolicyAgent
from analysis.config import AnalysisConfig


@dataclass
class StateAwareDeltaPolicyAgent(PolicyAgent):
    policy_columns: Tuple[str, ...]
    outcome_columns: Tuple[str, ...]
    weights: np.ndarray  # shape: (n_policies, n_features)
    max_levels: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.policy_columns = tuple(self.policy_columns)
        self.outcome_columns = tuple(self.outcome_columns)
        self.weights = np.asarray(self.weights, dtype=float)

        n_policies = len(self.policy_columns)
        if self.weights.ndim != 2:
            raise ValueError(f"weights must be 2D, got shape {self.weights.shape}")
        if self.weights.shape[0] != n_policies:
            raise ValueError(
                f"weights.shape[0] (={self.weights.shape[0]}) "
                f"!= len(policy_columns) (={n_policies})"
            )

        if self.max_levels is not None:
            self.max_levels = np.asarray(self.max_levels, dtype=float)
            if self.max_levels.shape[0] != n_policies:
                raise ValueError(
                    f"max_levels length {self.max_levels.shape[0]} != "
                    f"len(policy_columns)={n_policies}"
                )

    @property
    def n_features(self) -> int:
        return self.weights.shape[1]

    def _build_features(
        self,
        window_df: pd.DataFrame,
        baseline_policy: np.ndarray,
        lookback: int = 7,
    ) -> np.ndarray:
        """
        Build a feature vector using outcome & policy history.

        Features (in order):
          - 1.0 bias
          - For each outcome column:
              * log1p(last_value_clipped_to_0)
              * log1p(mean_of_last_`lookback`_days_clipped_to_0)
          - Baseline policy (current day) scaled by 1/100
        """
        last_row = window_df.iloc[-1]

        feats: List[float] = []
        # Bias
        feats.append(1.0)

        # Outcomes: level + short-term average
        for col in self.outcome_columns:
            if col not in window_df.columns:
                # if outcome wasn't present in this geo/model, just 0s
                feats.append(0.0)
                feats.append(0.0)
                continue

            series = pd.to_numeric(window_df[col], errors="coerce").fillna(0.0)
            series = np.maximum(series.to_numpy(dtype=float), 0.0)

            last_val = series[-1]
            window = series[-lookback:] if len(series) >= lookback else series
            avg_val = float(np.mean(window)) if len(window) > 0 else 0.0

            feats.append(np.log1p(last_val))
            feats.append(np.log1p(avg_val))

        # Baseline policy levels (current day) – normalized
        bp = np.asarray(baseline_policy, dtype=float)
        feats.extend(bp / 100.0)

        return np.asarray(feats, dtype=float)

    def act(
        self,
        window_df: pd.DataFrame,
        baseline_policy: np.ndarray,
    ) -> np.ndarray:
        baseline_policy = np.asarray(baseline_policy, dtype=float)
        if baseline_policy.shape[0] != len(self.policy_columns):
            raise ValueError(
                f"baseline_policy length {baseline_policy.shape[0]} != "
                f"len(policy_columns)={len(self.policy_columns)}"
            )

        features = self._build_features(window_df, baseline_policy)
        if features.shape[0] != self.n_features:
            raise ValueError(
                f"Feature length {features.shape[0]} != n_features={self.n_features}"
            )

        # delta = W @ x
        deltas = self.weights @ features  # shape: (n_policies,)

        new_policy = baseline_policy + deltas

        # Clamp at >= 0
        new_policy = np.maximum(new_policy, 0.0)

        # Optional clamp to max per column
        if self.max_levels is not None:
            new_policy = np.minimum(new_policy, self.max_levels)

        # Round to nearest int to stay on ordinal grid
        return np.rint(new_policy).astype(float)


@dataclass
class EvolutionaryPolicyTrainer:
    sim: RLSimulator
    train_geos: Sequence[str]
    start_index: int
    n_steps: int
    policy_columns: Sequence[str]
    max_levels: np.ndarray | None = None
    n_jobs: int = 1 

    def __post_init__(self) -> None:
        self.policy_columns = tuple(self.policy_columns)
        self.outcome_columns = tuple(AnalysisConfig.metadata.outcome_columns)

        if self.max_levels is not None:
            self.max_levels = np.asarray(self.max_levels, dtype=float)

        # Features: 1 (bias) +
        #   2 * n_outcomes (last + rolling avg) +
        #   n_policies (baseline policy)
        self.n_outcomes = len(self.outcome_columns)
        self.n_policies = len(self.policy_columns)
        self.n_features = 1 + 2 * self.n_outcomes + self.n_policies

        # Total parameter dimension = n_policies * n_features
        self.param_dim = self.n_policies * self.n_features

    def _build_agent(self, flat_params: np.ndarray) -> StateAwareDeltaPolicyAgent:
        flat_params = np.asarray(flat_params, dtype=float)
        if flat_params.shape[0] != self.param_dim:
            raise ValueError(
                f"Expected param_dim={self.param_dim}, " f"got {flat_params.shape[0]}"
            )

        weights = flat_params.reshape(self.n_policies, self.n_features)

        return StateAwareDeltaPolicyAgent(
            policy_columns=self.policy_columns,
            outcome_columns=self.outcome_columns,
            weights=weights,
            max_levels=self.max_levels,
        )

    def _evaluate_params_single_geo(
        self,
        deltas_flat: np.ndarray,
        geo: str,
    ) -> float:
        agent = self._build_agent(deltas_flat)
        ep = self.sim.simulate_episode(
            geo_id=geo,
            agent=agent,
            start_index=self.start_index,
            n_steps=self.n_steps,
            verbose=False,
        )

        r = np.array([s.reward_simulated for s in ep.steps], dtype=float)
        if np.all(np.isnan(r)):
            return np.nan

        return float(np.nanmean(r))

    def evaluate_params(self, deltas_flat: np.ndarray) -> float:
        if self.n_jobs == 1:
            rewards: List[float] = []
            for geo in self.train_geos:
                val = self._evaluate_params_single_geo(deltas_flat, geo)
                if not np.isnan(val):
                    rewards.append(val)
        else:
            vals = Parallel(n_jobs=self.n_jobs, prefer="threads")(
                delayed(self._evaluate_params_single_geo)(deltas_flat, geo)
                for geo in self.train_geos
            )
            rewards = [v for v in vals if not np.isnan(v)]

        if not rewards:
            return -np.inf

        return float(np.mean(rewards))

    def train(
        self,
        n_generations: int = 2,
        pop_size: int = 32,
        elite_frac: float = 0.25,
        init_std: float = 0.5,
        mutation_std: float = 0.5,
        seed: int | None = None,
        verbose: bool = True,
    ) -> tuple[PolicyAgent, pd.DataFrame]:
        rng = np.random.default_rng(seed)

        # Population: pop_size x param_dim
        pop = rng.normal(loc=0.0, scale=init_std, size=(pop_size, self.param_dim))
        n_elite = max(1, int(pop_size * elite_frac))

        history_rows: List[dict] = []

        for gen in range(n_generations):
            fitness = np.empty(pop_size, dtype=float)

            # Evaluate each individual (this can itself be parallelized if you want)
            for i in tqdm(range(pop_size), desc=f"Gen {gen:03d}", unit="agents"):
                fitness[i] = self.evaluate_params(pop[i])

            # Sort by descending fitness
            idx = np.argsort(fitness)[::-1]
            pop = pop[idx]
            fitness = fitness[idx]

            best_fit = fitness[0]
            mean_fit = float(np.mean(fitness))

            history_rows.append(
                {
                    "generation": gen,
                    "best_fitness": float(best_fit),
                    "mean_fitness": mean_fit,
                }
            )

            if verbose:
                print(f"[Gen {gen:03d}] best={best_fit:.4f}, " f"mean={mean_fit:.4f}")

            #  keep top n_elite, refill the rest with mutated copies
            elites = pop[:n_elite]
            new_pop = [elites[0]]  # keep the single best unchanged

            while len(new_pop) < pop_size:
                parent_idx = rng.integers(low=0, high=n_elite)
                parent = elites[parent_idx]
                child = parent + rng.normal(
                    loc=0.0,
                    scale=mutation_std,
                    size=self.param_dim,
                )
                new_pop.append(child)

            pop = np.stack(new_pop, axis=0)

        # Final evaluation to pick best individual
        final_fitness = np.array([self.evaluate_params(p) for p in pop], dtype=float)
        best_idx = int(np.argmax(final_fitness))
        best_params = pop[best_idx]

        if verbose:
            print("Training complete.")
            print(f"Best fitness: {final_fitness[best_idx]:.4f}")

        history_df = pd.DataFrame(history_rows)
        best_agent = self._build_agent(best_params)
        return best_agent, history_df
