from pathlib import Path
import argparse

from analysis.config import AnalysisConfig
from analysis.oxcgrt_data import OxCGRTData
from analysis.fwd import ModelForwarder
from analysis.rl import RLSimulator
from analysis.agents.evolutionary import EvolutionaryPolicyTrainer
from analysis.rl_eval import plot_outcomes, plot_reward, plot_differences
from analysis.policy_levels import get_strictest_action_for_columns

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evolutionary policy search."
    )
    p.add_argument("--geos", nargs="+", required=True)
    p.add_argument("--start-index", type=int, default=60)
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--generations", type=int, default=20)
    p.add_argument("--output-dir", type=str, default="figures/rl")
    p.add_argument("--no_resume", action="store_true")
    p.add_argument("--stop-file", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = AnalysisConfig

    data = OxCGRTData(cfg.paths.data)
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

    trainer = EvolutionaryPolicyTrainer(
        sim=sim,
        train_geos=args.geos,
        start_index=args.start_index,
        n_steps=args.steps,
        policy_columns=policy_cols,
        max_levels=max_levels,
        n_jobs=1,
    )

    checkpoint_path = Path(cfg.paths.output) / "agents" / f"{('_'.join(args.geos))}_start{args.start_index}_steps{args.steps}_gen{args.generations}.joblib"

    stop_file = Path(args.stop_file) if args.stop_file else None

    best_agent, history_df = trainer.train(
        n_generations=args.generations,
        pop_size=32,
        elite_frac=0.25,
        init_std=0.5,
        mutation_std=0.3,
        seed=42,
        verbose=True,
        checkpoint_path=checkpoint_path,
        resume=not args.no_resume,
        stop_file=stop_file,
    )

    history_df.to_csv(out_dir / "evolution_history.csv", index=False)

    # Evaluate best agent, as you were doing
    for geo in args.geos:
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
