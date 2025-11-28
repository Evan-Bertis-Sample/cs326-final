from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd

from analysis.config import AnalysisConfig
from analysis.oxcgrt_data import OxCGRTData
from analysis.fwd import ModelForwarder
from analysis.rl import RLSimulator, RelaxationAgent
from analysis.rl_eval import plot_outcomes, plot_reward, plot_differences
from analysis.agents.strict_policy import StrictPolicyAgent


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
        "--output-dir",
        type=str,
        default="figures/rl",
        help="Directory to store plots and stats.",
    )
    return p.parse_args()


def build_agents() -> Dict[str, Any]:
    policy_cols = AnalysisConfig.metadata.policy_columns

    agents: Dict[str, Any] = {
        "strict": StrictPolicyAgent(policy_cols),
        "relax_0.9": RelaxationAgent(scale=0.9),
        "relax_0.8": RelaxationAgent(scale=0.8),
    }
    return agents


def compute_episode_stats(agent_name: str, ep) -> Dict[str, Any]:
    md = AnalysisConfig.metadata
    outcome_cols = md.outcome_columns

    n_steps = len(ep.steps)
    if n_steps == 0:
        return {
            "agent": agent_name,
            "geo": ep.geo_id,
            "n_steps": 0,
        }

    reward_actual = np.array([s.reward_actual for s in ep.steps], dtype=float)
    reward_sim = np.array([s.reward_simulated for s in ep.steps], dtype=float)

    row: Dict[str, Any] = {
        "agent": agent_name,
        "geo": ep.geo_id,
        "n_steps": n_steps,
        "reward_actual_mean": float(np.nanmean(reward_actual)),
        "reward_sim_mean": float(np.nanmean(reward_sim)),
        "reward_delta_mean": float(np.nanmean(reward_sim - reward_actual)),
    }

    y_true = np.stack([s.y_true for s in ep.steps], axis=0)
    y_pred = np.stack([s.y_pred for s in ep.steps], axis=0)

    for i, col in enumerate(outcome_cols):
        if i >= y_true.shape[1] or i >= y_pred.shape[1]:
            continue
        delta = y_pred[:, i] - y_true[:, i]  # sim - actual
        row[f"{col}_delta_mean"] = float(np.nanmean(delta))

    return row


def main() -> None:
    args = parse_args()

    # Load analysis config and data
    cfg = AnalysisConfig
    data_path = cfg.paths.data

    print(f"Loading OxCGRT data from {data_path}...")
    data = OxCGRTData(data_path)

    # Forwarder and RL simulator
    forwarder = ModelForwarder(data=data)
    sim = RLSimulator(data=data, forwarder=forwarder)

    # Build all agents
    agents = build_agents()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats_rows: List[Dict[str, Any]] = []

    for agent_name, agent in agents.items():
        print(f"\nRunning agent: {agent_name}")
        agent_dir = out_dir / agent_name

        for geo in args.geos:
            print(f"\nSimulating geo: {geo} with agent {agent_name}")
            ep = sim.simulate_episode(
                geo_id=geo,
                agent=agent,
                start_index=args.start_index,
                n_steps=args.steps,
            )

            geo_dir = agent_dir / geo
            plot_outcomes(ep, agent_name, output_dir=geo_dir)
            plot_reward(ep, agent_name, output_dir=geo_dir)
            plot_differences(ep, agent_name, output_dir=geo_dir)

            # Collect stats for this (agent, geo) pair
            stats_row = compute_episode_stats(agent_name, ep)
            stats_rows.append(stats_row)

            print(f"Plots saved under {geo_dir}")
            print("")

    # Save stats
    if stats_rows:
        stats_df = pd.DataFrame(stats_rows)
        stats_path = out_dir / "agent_geo_stats.csv"
        stats_df.to_csv(stats_path, index=False)
        print(f"\nPer-geo stats saved to {stats_path}")

        # Aggregate per agent
        agg_df = stats_df.groupby("agent").mean(numeric_only=True).reset_index()
        agg_path = out_dir / "agent_stats_overall.csv"
        agg_df.to_csv(agg_path, index=False)
        print(f"Per-agent aggregated stats saved to {agg_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
