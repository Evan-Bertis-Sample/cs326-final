from __future__ import annotations

from pathlib import Path

from analysis.config import AnalysisConfig
from analysis.oxcgrt_data import OxCGRTData
from analysis.fwd import ModelForwarder
from analysis.rl import RLSimulator
from analysis.agents.evolutionary import EvolutionaryPolicyTrainer
from analysis.rl_eval import plot_outcomes, plot_reward, plot_differences
import pandas as pd
import argparse


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
        help="Number of generations to train the simulation on."
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="figures/rl",
        help="Directory to store plots and stats.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = AnalysisConfig
    data_path = cfg.paths.data

    print(f"Loading OxCGRT data from {data_path}...")
    data = OxCGRTData(data_path)
    forwarder = ModelForwarder(data=data)
    sim = RLSimulator(data=data, forwarder=forwarder)

    out_dir = Path(args.output_dir) / "evolutionary"
    out_dir.mkdir(parents=True, exist_ok=True)

    policy_cols = cfg.metadata.policy_columns

    max_levels = None

    train_geos = ["USA"]

    trainer = EvolutionaryPolicyTrainer(
        sim=sim,
        train_geos=train_geos,
        start_index=args.start_index,
        n_steps=args.steps,
        policy_columns=policy_cols,
        max_levels=max_levels,
    )

    best_agent, history_df = trainer.train(
        n_generations=args.generations,
        pop_size=32,
        elite_frac=0.25,
        init_std=0.5,
        mutation_std=0.3,
        seed=42,
        verbose=True,
    )


    history_df.to_csv(out_dir / "evolution_history.csv", index=False)

    eval_geos = train_geos

    for geo in eval_geos:
        print(f"Evaluating best evolutionary agent on {geo}")
        ep = sim.simulate_episode(
            geo_id=geo,
            agent=best_agent,
            start_index=args.start_index,
            n_steps=args.steps,
        )
        geo_dir = out_dir / geo
        agent_name = "Evolutionary"
        plot_outcomes(ep, agent_name, output_dir=geo_dir)
        plot_reward(ep, agent_name, output_dir=geo_dir)
        plot_differences(ep, agent_name, output_dir=geo_dir)


if __name__ == "__main__":
    main()
