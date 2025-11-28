# sim.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from analysis.config import AnalysisConfig
from analysis.oxcgrt_data import OxCGRTData
from analysis.fwd import ModelForwarder
from analysis.rl import RLSimulator, RelaxationAgent
from analysis.rl_eval import plot_outcomes, plot_reward, plot_differences
from analysis.agents.strict_policy import StrictPolicyAgent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RL-style simulation using pretrained COVID models.")
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
        help="Directory to store plots.",
    )
    return p.parse_args()


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

    agent = StrictPolicyAgent(AnalysisConfig.metadata.policy_columns)
    out_dir = Path(args.output_dir)

    for geo in args.geos:
        print(f"\nSimulating geo: {geo}")
        ep = sim.simulate_episode(
            geo_id=geo,
            agent=agent,
            start_index=args.start_index,
            n_steps=args.steps,
        )

        geo_dir = out_dir / geo
        plot_outcomes(ep, output_dir=geo_dir)
        plot_reward(ep, output_dir=geo_dir)
        plot_differences(ep, output_dir=geo_dir)  # <-- add this

        print(f"Plots saved under {geo_dir}")
        print("")

    print("\nDone.")


if __name__ == "__main__":
    main()
