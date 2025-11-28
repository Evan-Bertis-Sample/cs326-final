# analysis/agents/evolutionary_policy.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Mapping, Any, Iterable, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from analysis.rl import RLSimulator, PolicyAgent
from analysis.config import AnalysisConfig


@dataclass
class DeltaPolicyAgent(PolicyAgent):
    policy_columns: Tuple[str, ...]
    deltas: np.ndarray
    max_levels: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.deltas = np.asarray(self.deltas, dtype=float)
        if self.max_levels is not None:
            self.max_levels = np.asarray(self.max_levels, dtype=float)

        if self.deltas.shape[0] != len(self.policy_columns):
            raise ValueError(
                f"deltas length {self.deltas.shape[0]} != "
                f"len(policy_columns)={len(self.policy_columns)}"
            )
        if self.max_levels is not None and self.max_levels.shape[0] != len(self.policy_columns):
            raise ValueError(
                f"max_levels length {self.max_levels.shape[0]} != "
                f"len(policy_columns)={len(self.policy_columns)}"
            )

    def act(
        self,
        window_df: pd.DataFrame,
        baseline_policy: np.ndarray,
    ) -> np.ndarray:
        # baseline_policy is usually integer-coded policies.
        base = baseline_policy.astype(float)
        if base.shape[0] != self.deltas.shape[0]:
            raise ValueError(
                f"baseline_policy length {base.shape[0]} != "
                f"len(deltas)={self.deltas.shape[0]}"
            )

        new_policy = base + self.deltas

        # Clamp at >= 0
        new_policy = np.maximum(new_policy, 0.0)

        # Optional clamp to max per column
        if self.max_levels is not None:
            new_policy = np.minimum(new_policy, self.max_levels)

        # Round to nearest int to stay on the ordinal grid
        return np.rint(new_policy).astype(float)
    

@dataclass
class EvolutionaryPolicyTrainer:
    sim: RLSimulator
    train_geos: Sequence[str]
    start_index: int
    n_steps: int
    policy_columns: Sequence[str]
    max_levels: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.policy_columns = tuple(self.policy_columns)
        if self.max_levels is not None:
            self.max_levels = np.asarray(self.max_levels, dtype=float)

    def _build_agent(self, deltas: np.ndarray) -> DeltaPolicyAgent:
        return DeltaPolicyAgent(
            policy_columns=self.policy_columns,
            deltas=deltas,
            max_levels=self.max_levels,
        )

    def evaluate_params(self, deltas: np.ndarray) -> float:
        agent = self._build_agent(deltas)
        rewards: List[float] = []

        for geo in self.train_geos:
            ep = self.sim.simulate_episode(
                geo_id=geo,
                agent=agent,
                start_index=self.start_index,
                n_steps=self.n_steps,
                verbose=False
            )
            # Use the simulated reward (under the agent's policy)
            r = np.array([s.reward_simulated for s in ep.steps], dtype=float)
            if np.all(np.isnan(r)):
                continue
            rewards.append(float(np.nanmean(r)))

        if not rewards:
            return -np.inf  # everything NaN / invalid

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
    ) -> tuple[DeltaPolicyAgent, pd.DataFrame]:
        rng = np.random.default_rng(seed)
        dim = len(self.policy_columns)

        # Population: pop_size x dim, initialised near 0
        pop = rng.normal(loc=0.0, scale=init_std, size=(pop_size, dim))

        n_elite = max(1, int(pop_size * elite_frac))

        history_rows: List[dict] = []

    
        for gen in tqdm(range(n_generations), desc="Training", unit="generations"):
            fitness = np.empty(pop_size, dtype=float)

            # Evaluate each individual
            for i in tqdm(range(pop_size), desc="Evaluating", unit="agents"):
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
                print(
                    f"[Gen {gen:03d}] best={best_fit:.4f}, "
                    f"mean={mean_fit:.4f}"
                )

            # Elitism: keep top n_elite, refill the rest with mutated copies
            elites = pop[:n_elite]
            new_pop = [elites[0]]  # always keep the single best unchanged

            # Sample parents from elites for the rest
            while len(new_pop) < pop_size:
                parent_idx = rng.integers(low=0, high=n_elite)
                parent = elites[parent_idx]
                child = parent + rng.normal(loc=0.0, scale=mutation_std, size=dim)
                new_pop.append(child)

            pop = np.stack(new_pop, axis=0)

        # Final evaluation to pick best individual
        final_fitness = np.array([self.evaluate_params(p) for p in pop], dtype=float)
        best_idx = int(np.argmax(final_fitness))
        best_deltas = pop[best_idx]

        if verbose:
            print("Training complete.")
            print(f"Best fitness: {final_fitness[best_idx]:.4f}")
            print(f"Best deltas: {best_deltas}")

        history_df = pd.DataFrame(history_rows)
        best_agent = self._build_agent(best_deltas)
        return best_agent, history_df
