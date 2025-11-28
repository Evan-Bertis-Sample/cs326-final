from __future__ import annotations

from pathlib import Path
from typing import List

import argparse
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.config import AnalysisConfig
from analysis.oxcgrt_data import OxCGRTData
from analysis.fwd import ModelForwarder
from analysis.rl import RLSimulator, EpisodeResult
from analysis.agents.evolutionary import EvolutionaryPolicyTrainer
from analysis.rl_eval import plot_outcomes, plot_reward, plot_differences
from analysis.cache import Cache, CacheConfig
from analysis.policy_levels import get_strictest_action_for_columns


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RL-style simulation using pretrained COVID models."
    )
    p.add_argument(
        "--geos",
        nargs="+",
        required=True,
        help="GeoIDs to simulate (e.g. USA CHN ITA).",
    )
    p.add_argument(
        "--start-index",
        type=int,
        default=60,
        help="Index of the first prediction day in the per-geo time series.",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=60,
        help="Number of simulated steps (days) per geo.",
    )
    p.add_argument(
        "--generations",
        type=int,
        default=20,
        help="Number of generations to train the simulation on.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="figures/rl",
        help="Directory to store plots and stats.",
    )
    return p.parse_args()


def plot_policy_decisions(
    ep: EpisodeResult,
    policy_columns: List[str],
    agent_name: str,
    output_dir: Path,
) -> None:
    """
    Plot the baseline vs agent policy levels over time for each policy column.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    dates = [s.date for s in ep.steps]
    baseline = np.stack([s.policy_baseline for s in ep.steps], axis=0)
    actions = np.stack([s.policy_action for s in ep.steps], axis=0)

    # Build a DataFrame indexed by date for easier plotting
    df = pd.DataFrame(index=pd.to_datetime(dates))
    df.index.name = "Date"

    for i, col in enumerate(policy_columns):
        if i >= baseline.shape[1] or i >= actions.shape[1]:
            continue

        base_col = f"{col}_baseline"
        act_col = f"{col}_action"

        df[base_col] = baseline[:, i]
        df[act_col] = actions[:, i]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df.index, df[base_col], label="Baseline", linewidth=1.5)
        ax.plot(
            df.index,
            df[act_col],
            label=f"{agent_name} policy",
            linestyle="--",
            linewidth=1.5,
        )

        ax.set_title(f"{ep.geo_id} – {col}: Baseline vs {agent_name}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Policy level")
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.legend(loc="best")

        fig.autofmt_xdate()
        out_path = output_dir / f"{ep.geo_id}_{agent_name}_{col}_policy.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = AnalysisConfig
    data_path = cfg.paths.data

    cache_root = AnalysisConfig.paths.cache
    Cache.init(
        CacheConfig(root=cache_root, compress=3)
    )

    print(f"Loading OxCGRT data from {data_path}...")
    data = OxCGRTData(data_path)
    forwarder = ModelForwarder(data=data)
    sim = RLSimulator(data=data, forwarder=forwarder)

    out_dir = Path(args.output_dir) / "evolutionary"
    out_dir.mkdir(parents=True, exist_ok=True)

    policy_cols = list(cfg.metadata.policy_columns)

    # actually get the strictest level per policy
    strictest_map = get_strictest_action_for_columns(policy_cols)
    max_levels = np.array(
        [strictest_map.get(col, 0) for col in policy_cols],
        dtype=float,
    )

    train_geos = args.geos

    trainer = EvolutionaryPolicyTrainer(
        sim=sim,
        train_geos=train_geos,
        start_index=args.start_index,
        n_steps=args.steps,
        policy_columns=policy_cols,
        max_levels=max_levels,
        n_jobs=8,
    )

    # use Cache to memoize the training run
    Cache.Begin(f"RL_Training_{args.steps}_steps")
    best_agent, history_df = Cache.call(
        trainer.train,
        n_generations=args.generations,
        pop_size=32,
        elite_frac=0.25,
        init_std=0.5,
        mutation_std=0.3,
        seed=42,
        verbose=True,
    )
    Cache.End()

    history_df.to_csv(out_dir / "evolution_history.csv", index=False)

    eval_geos = args.geos
    agent_name = "Evolutionary"

    for geo in eval_geos:
        print(f"Evaluating best evolutionary agent on {geo}")
        ep = sim.simulate_episode(
            geo_id=geo,
            agent=best_agent,
            start_index=args.start_index,
            n_steps=args.steps,
        )
        geo_dir = out_dir / geo

        # outcomes & rewards
        plot_outcomes(ep, agent_name, output_dir=geo_dir)
        plot_reward(ep, agent_name, output_dir=geo_dir)
        plot_differences(ep, agent_name, output_dir=geo_dir)

        # plot the decisions made by the agent
        plot_policy_decisions(
            ep=ep,
            policy_columns=policy_cols,
            agent_name=agent_name,
            output_dir=geo_dir,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
