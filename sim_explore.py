from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.config import AnalysisConfig
from analysis.oxcgrt_data import OxCGRTData
from analysis.fwd import ModelForwarder
from analysis.rl import RLSimulator, EpisodeResult
from analysis.rl_eval import episode_to_dataframe
from analysis.policy_levels import POLICY_SCALES, COLUMN_TO_POLICY_ID
from analysis.agents.explorative_policy import (
    ExplorativePolicyAgent,
    BaselineAgent,
    SuperPolicyAgent,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Explore effectiveness of individual policies via single-policy interventions."
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
        default="figures/rl/explore",
        help="Directory to store plots and stats.",
    )
    p.add_argument(
        "--smooth-window",
        type=int,
        default=0,
        help="Optional rolling window size for smoothing predictions (0 = no smoothing).",
    )
    return p.parse_args()



def summarize_episode(ep: EpisodeResult) -> Dict[str, float]:
    md = AnalysisConfig.metadata
    outcome_cols = md.outcome_columns

    rewards = np.array([s.reward_simulated for s in ep.steps], dtype=float)
    mean_reward = float(np.nanmean(rewards)) if rewards.size > 0 else np.nan

    # Stack predictions
    y_pred = np.stack([s.y_pred for s in ep.steps], axis=0) if ep.steps else None

    total_cases = np.nan
    total_deaths = np.nan

    if y_pred is not None and outcome_cols:
        try:
            idx_cases = outcome_cols.index("ConfirmedCases")
        except ValueError:
            idx_cases = None
        try:
            idx_deaths = outcome_cols.index("ConfirmedDeaths")
        except ValueError:
            idx_deaths = None

        if idx_cases is not None and idx_cases < y_pred.shape[1]:
            total_cases = float(np.nansum(y_pred[:, idx_cases]))
        if idx_deaths is not None and idx_deaths < y_pred.shape[1]:
            total_deaths = float(np.nansum(y_pred[:, idx_deaths]))

    return {
        "mean_reward": mean_reward,
        "total_cases_pred": total_cases,
        "total_deaths_pred": total_deaths,
    }


def maybe_smooth(df: pd.DataFrame, window: int) -> pd.DataFrame:
    if window <= 1:
        return df
    return df.rolling(window=window, min_periods=1).mean()


def plot_multi_for_policy(
    geo_id: str,
    policy_column: str,
    episodes_by_label: Dict[str, EpisodeResult],
    output_dir: Path,
    smooth_window: int = 0,
    focus_outcomes: List[str] | None = None,
) -> None:
    md = AnalysisConfig.metadata
    if focus_outcomes is None:
        focus_outcomes = md.outcome_columns

    # Convert episodes to dataframes
    dfs: Dict[str, pd.DataFrame] = {
        label: episode_to_dataframe(ep) for label, ep in episodes_by_label.items()
    }

    # Apply optional smoothing to numeric columns
    if smooth_window > 1:
        for label, df in dfs.items():
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            smoothed = df.copy()
            smoothed[numeric_cols] = df[numeric_cols].rolling(
                window=smooth_window, min_periods=1
            ).mean()
            dfs[label] = smoothed

    out_base = output_dir / geo_id / f"policy_{policy_column.replace(' ', '_')}"
    out_base.mkdir(parents=True, exist_ok=True)

    # Outcomes: one figure per focus outcome, multiple agents
    for col in focus_outcomes:
        true_col = f"{col}_true"
        pred_col = f"{col}_pred"

        # Only plot if baseline has the column
        if "baseline" not in dfs or pred_col not in dfs["baseline"].columns:
            continue

        # Absolute simulated trajectories
        fig, ax = plt.subplots(figsize=(10, 5))
        for label, df in dfs.items():
            if pred_col not in df.columns:
                continue
            ax.plot(
                df.index,
                df[pred_col],
                label=label,
                linewidth=1.5,
            )
        ax.set_title(f"{geo_id} – {policy_column}: simulated {col} (multi-agent)")
        ax.set_xlabel("Date")
        ax.set_ylabel(col)
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.legend(loc="best")
        fig.autofmt_xdate()

        fig.tight_layout()
        fig.savefig(out_base / f"{geo_id}_{col}_simulated_multi.png", dpi=200)
        plt.close(fig)

        # Deltas vs baseline for this outcome
        base_df = dfs["baseline"]
        base_series = base_df[pred_col]

        fig, ax = plt.subplots(figsize=(10, 5))
        for label, df in dfs.items():
            if label == "baseline":
                continue
            if pred_col not in df.columns:
                continue

            # Align to baseline index to avoid misalignment
            aligned = df[pred_col].reindex(base_series.index)
            delta = aligned - base_series

            ax.plot(
                base_series.index,
                delta,
                label=f"{label} - baseline",
                linewidth=1.5,
            )

        ax.axhline(0.0, color="black", linestyle=":", linewidth=1.0)
        ax.set_title(f"{geo_id} – {policy_column}: Δ {col} vs baseline")
        ax.set_xlabel("Date")
        ax.set_ylabel(f"{col} Δ")
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.legend(loc="best")
        fig.autofmt_xdate()

        fig.tight_layout()
        fig.savefig(out_base / f"{geo_id}_{col}_delta_multi.png", dpi=200)
        plt.close(fig)

    # Reward trajectories
    fig, ax = plt.subplots(figsize=(10, 4))
    for label, df in dfs.items():
        if "reward_simulated" not in df.columns:
            continue
        ax.plot(df.index, df["reward_simulated"], label=label, linewidth=1.5)
    ax.set_title(f"{geo_id} – {policy_column}: simulated reward (multi-agent)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Reward")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="best")
    fig.autofmt_xdate()

    fig.tight_layout()
    fig.savefig(out_base / f"{geo_id}_reward_multi.png", dpi=200)
    plt.close(fig)

    # Reward deltas vs baseline
    if "baseline" in dfs:
        base_df = dfs["baseline"]
        if "reward_simulated" in base_df.columns:
            fig, ax = plt.subplots(figsize=(10, 4))
            base_rewards = base_df["reward_simulated"]

            for label, df in dfs.items():
                if label == "baseline":
                    continue
                if "reward_simulated" not in df.columns:
                    continue
                aligned = df["reward_simulated"].reindex(base_rewards.index)
                delta = aligned - base_rewards
                ax.plot(
                    base_rewards.index,
                    delta,
                    label=f"{label} - baseline",
                    linewidth=1.5,
                )

            ax.axhline(0.0, color="black", linestyle=":", linewidth=1.0)
            ax.set_title(f"{geo_id} – {policy_column}: reward Δ vs baseline")
            ax.set_xlabel("Date")
            ax.set_ylabel("Reward Δ")
            ax.grid(True, linestyle=":", alpha=0.4)
            ax.legend(loc="best")
            fig.autofmt_xdate()

            fig.tight_layout()
            fig.savefig(out_base / f"{geo_id}_reward_delta_multi.png", dpi=200)
            plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = AnalysisConfig
    data_path = cfg.paths.data

    print("Loading config...")
    print(f"Loading OxCGRT data from {data_path}...")
    data = OxCGRTData(data_path)
    forwarder = ModelForwarder(data=data)
    sim = RLSimulator(data=data, forwarder=forwarder)

    out_dir = Path(args.output_dir) / f"{args.start_index}_to_{args.start_index + args.steps}"
    out_dir.mkdir(parents=True, exist_ok=True)

    policy_cols = list(cfg.metadata.policy_columns)

    # Filter to policy columns that are actually in POLICY_SCALES (i.e., with discrete levels)
    explorable_policy_cols: List[str] = [
        col for col in policy_cols if col in COLUMN_TO_POLICY_ID
    ]

    print("Explorable policy columns:")
    for col in explorable_policy_cols:
        pid = COLUMN_TO_POLICY_ID[col]
        print(f"  - {pid}: {col} (levels={sorted(POLICY_SCALES[pid].levels.keys())})")

    # Collect all results in a single dataframe
    results_rows: List[Dict[str, object]] = []

    for geo in args.geos:
        print(f"\nGeo: {geo}")

        # Baseline episode (no policy change)
        print("  Simulating baseline (no policy intervention)...")
        baseline_agent = BaselineAgent()
        baseline_ep = sim.simulate_episode(
            geo_id=geo,
            agent=baseline_agent,
            start_index=args.start_index,
            n_steps=args.steps,
            verbose=True,
        )
        baseline_summary = summarize_episode(baseline_ep)
        baseline_reward = baseline_summary["mean_reward"]

        best_level_by_policy: Dict[str, float] = {}
        best_reward_by_policy: Dict[str, float] = {}

        for policy_col in explorable_policy_cols:
            pid = COLUMN_TO_POLICY_ID[policy_col]
            scale = POLICY_SCALES[pid]
            levels = sorted(scale.levels.keys())

            print(f"  Exploring policy {pid} ({policy_col}) with levels {levels}...")

            # For multi-agent plotting for THIS policy
            episodes_for_plot: Dict[str, EpisodeResult] = {"baseline": baseline_ep}

            best_level = None
            best_reward = -np.inf

            for lvl in levels:
                label = f"level_{lvl}"
                agent = ExplorativePolicyAgent(
                    target_column=policy_col,
                    level=lvl,
                    policy_columns=policy_cols,
                )

                ep = sim.simulate_episode(
                    geo_id=geo,
                    agent=agent,
                    start_index=args.start_index,
                    n_steps=args.steps,
                    verbose=False,
                )
                episodes_for_plot[label] = ep

                summary = summarize_episode(ep)
                mean_reward = summary["mean_reward"]

                # Store per-agent row
                results_rows.append(
                    {
                        "geo_id": geo,
                        "policy_id": pid,
                        "policy_column": policy_col,
                        "level": lvl,
                        "level_desc": scale.levels.get(lvl, ""),
                        "mean_reward": mean_reward,
                        "mean_reward_delta_vs_baseline": (
                            mean_reward - baseline_reward
                            if not np.isnan(mean_reward)
                            and not np.isnan(baseline_reward)
                            else np.nan
                        ),
                        "total_cases_pred": summary["total_cases_pred"],
                        "total_deaths_pred": summary["total_deaths_pred"],
                        "baseline_mean_reward": baseline_reward,
                    }
                )

                if mean_reward > best_reward:
                    best_reward = mean_reward
                    best_level = lvl

            # Plot multi-agent trajectories (baseline + all levels) for this policy
            print(f"  Plotting multi-agent curves for policy {pid} ({policy_col})...")
            plot_multi_for_policy(
                geo_id=geo,
                policy_column=policy_col,
                episodes_by_label=episodes_for_plot,
                output_dir=out_dir,
                smooth_window=args.smooth_window,
            )

            if best_level is not None:
                best_level_by_policy[policy_col] = float(best_level)
                best_reward_by_policy[policy_col] = float(best_reward)
                print(
                    f"  Best level for {pid} ({policy_col}): {best_level} "
                    f"(mean_reward={best_reward:.4f})"
                )

        # Build and evaluate "super" agent for this geo
        if best_level_by_policy:
            print(f"  Building super agent for {geo} using best levels per policy...")
            super_agent = SuperPolicyAgent(
                policy_columns=policy_cols,
                fixed_levels=best_level_by_policy,
            )

            super_ep = sim.simulate_episode(
                geo_id=geo,
                agent=super_agent,
                start_index=args.start_index,
                n_steps=args.steps,
                verbose=True,
            )
            super_summary = summarize_episode(super_ep)
            super_reward = super_summary["mean_reward"]

            results_rows.append(
                {
                    "geo_id": geo,
                    "policy_id": "SUPER",
                    "policy_column": "SUPER",
                    "level": np.nan,
                    "level_desc": "Best per-policy levels combined",
                    "mean_reward": super_reward,
                    "mean_reward_delta_vs_baseline": (
                        super_reward - baseline_reward
                        if not np.isnan(super_reward) and not np.isnan(baseline_reward)
                        else np.nan
                    ),
                    "total_cases_pred": super_summary["total_cases_pred"],
                    "total_deaths_pred": super_summary["total_deaths_pred"],
                    "baseline_mean_reward": baseline_reward,
                }
            )

            # Also plot super vs baseline
            print(f"  Plotting super agent vs baseline for {geo}...")
            episodes_for_super_plot = {
                "baseline": baseline_ep,
                "super": super_ep,
            }
            plot_multi_for_policy(
                geo_id=geo,
                policy_column="SUPER",
                episodes_by_label=episodes_for_super_plot,
                output_dir=out_dir,
                smooth_window=args.smooth_window,
            )

    # Save global results 
    results_df = pd.DataFrame(results_rows)
    results_path = out_dir / "explorative_policy_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nAll results saved to: {results_path}")


if __name__ == "__main__":
    main()
