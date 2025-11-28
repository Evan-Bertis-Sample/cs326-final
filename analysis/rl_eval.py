from __future__ import annotations

from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.config import AnalysisConfig
from analysis.rl import EpisodeResult


def episode_to_dataframe(ep: EpisodeResult) -> pd.DataFrame:
    md = AnalysisConfig.metadata
    outcome_cols = md.outcome_columns

    dates = [s.date for s in ep.steps]
    y_true = np.stack([s.y_true for s in ep.steps], axis=0)
    y_pred = np.stack([s.y_pred for s in ep.steps], axis=0)
    reward_actual = np.array([s.reward_actual for s in ep.steps], dtype=float)
    reward_simulated = np.array([s.reward_simulated for s in ep.steps], dtype=float)

    df = pd.DataFrame(index=pd.to_datetime(dates))
    df.index.name = "Date"

    for i, col in enumerate(outcome_cols):
        if i < y_true.shape[1]:
            df[f"{col}_true"] = y_true[:, i]
        if i < y_pred.shape[1]:
            df[f"{col}_pred"] = y_pred[:, i]

    df["reward_actual"] = reward_actual
    df["reward_simulated"] = reward_simulated
    return df


def _smooth_series(
    s: pd.Series,
    window: Optional[int],
) -> pd.Series:
    if window is None or window <= 1:
        return s
    return s.rolling(window=window, min_periods=1).mean()


def plot_outcomes(
    ep: EpisodeResult,
    agent_name : str,
    output_dir: Path,
    focus_columns: Optional[List[str]] = None,
    smooth_window: Optional[int] = 7,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = episode_to_dataframe(ep)
    md = AnalysisConfig.metadata

    cols = md.outcome_columns if focus_columns is None else focus_columns

    for col in cols:
        true_col = f"{col}_true"
        pred_col = f"{col}_pred"
        if true_col not in df.columns or pred_col not in df.columns:
            continue

        true_raw = df[true_col]
        pred_raw = df[pred_col]

        true_smooth = _smooth_series(true_raw, smooth_window)
        pred_smooth = _smooth_series(pred_raw, smooth_window)

        fig, ax = plt.subplots(figsize=(10, 5))

        # Optional: plot raw faint + smoothed bold
        ax.plot(df.index, true_raw, label="Actual (raw)", linewidth=0.8, alpha=0.25)
        ax.plot(df.index, pred_raw, label="Simulated (raw)", linewidth=0.8, alpha=0.25)

        ax.plot(df.index, true_smooth, label="Actual (smoothed)", linewidth=1.8)
        ax.plot(
            df.index,
            pred_smooth,
            label="Simulated (smoothed)",
            linestyle="--",
            linewidth=1.8,
        )

        ax.set_title(f"{ep.geo_id} – {col}: Actual vs Simulated | {agent_name}")
        ax.set_xlabel("Date")
        ax.set_ylabel(col)
        ax.legend(loc="best")
        ax.grid(True, linestyle=":", alpha=0.4)

        fig.autofmt_xdate()
        out_path = output_dir / f"{ep.geo_id}_{col}_actual_vs_sim.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)


def plot_differences(
    ep: EpisodeResult,
    agent_name : str,
    output_dir: Path,
    focus_columns: Optional[List[str]] = None,
    smooth_window: Optional[int] = 7,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = episode_to_dataframe(ep)

    # Plot rewards
    if "reward_actual" in df.columns and "reward_simulated" in df.columns:
        df["reward_delta"] = df["reward_simulated"] - df["reward_actual"]
        delta_raw = df["reward_delta"]
        delta_smooth = _smooth_series(delta_raw, smooth_window)

        fig, ax = plt.subplots(figsize=(10, 4))

        ax.plot(df.index, delta_raw, label="Δ Reward (raw)", linewidth=0.8, alpha=0.25)
        ax.plot(
            df.index,
            delta_smooth,
            label="Δ Reward (smoothed)",
            linewidth=1.8,
        )

        ax.axhline(0.0, color="black", linestyle=":", linewidth=1.0)

        ax.set_title(f"{ep.geo_id} – Reward difference over time (sim - actual) | {agent_name}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Reward Δ")
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.legend(loc="best")

        fig.autofmt_xdate()
        out_path = output_dir / f"{ep.geo_id}_reward_diff.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)

    # Plot outcomes
    md = AnalysisConfig.metadata
    all_outcomes = md.outcome_columns

    # Default to a small, interpretable subset
    if focus_columns is None:
        focus_candidates = ["ConfirmedCases", "ConfirmedDeaths"]
        focus_columns = [c for c in focus_candidates if c in all_outcomes]

    for col in focus_columns:
        true_col = f"{col}_true"
        pred_col = f"{col}_pred"
        if true_col not in df.columns or pred_col not in df.columns:
            continue

        diff_raw = df[pred_col] - df[true_col]
        diff_smooth = _smooth_series(diff_raw, smooth_window)

        fig, ax = plt.subplots(figsize=(10, 4))

        ax.plot(
            df.index,
            diff_raw,
            label=f"Δ {col} (raw)",
            linewidth=0.8,
            alpha=0.25,
        )
        ax.plot(
            df.index,
            diff_smooth,
            label=f"Δ {col} (smoothed)",
            linewidth=1.8,
        )

        ax.axhline(0.0, color="black", linestyle=":", linewidth=1.0)

        ax.set_title(f"{ep.geo_id} – {col} difference over time (sim - actual) | {agent_name}")
        ax.set_xlabel("Date")
        ax.set_ylabel(f"{col} Δ")
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.legend(loc="best")

        fig.autofmt_xdate()
        out_path = output_dir / f"{ep.geo_id}_{col}_diff.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)


def plot_reward(
    ep: EpisodeResult,
    agent_name : str,
    output_dir: Path,
    smooth_window: Optional[int] = 7,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = episode_to_dataframe(ep)

    actual_raw = df["reward_actual"]
    sim_raw = df["reward_simulated"]

    actual_smooth = _smooth_series(actual_raw, smooth_window)
    sim_smooth = _smooth_series(sim_raw, smooth_window)

    fig, ax = plt.subplots(figsize=(10, 4))

    # Raw
    ax.plot(
        df.index, actual_raw, label="Actual reward (raw)", linewidth=0.8, alpha=0.25
    )
    ax.plot(
        df.index,
        sim_raw,
        label="Simulated reward (raw)",
        linestyle="--",
        linewidth=0.8,
        alpha=0.25,
    )

    # Smoothed
    ax.plot(df.index, actual_smooth, label="Actual reward (smoothed)", linewidth=1.8)
    ax.plot(
        df.index,
        sim_smooth,
        label="Simulated reward (smoothed)",
        linestyle="--",
        linewidth=1.8,
    )

    ax.set_title(f"{ep.geo_id} – Reward over time | {agent_name}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Reward")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="best")

    fig.autofmt_xdate()
    out_path = output_dir / f"{ep.geo_id}_reward.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
