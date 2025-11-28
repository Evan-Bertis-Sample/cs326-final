# rl_eval.py
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


def plot_outcomes(
    ep: EpisodeResult,
    output_dir: Path,
    focus_columns: Optional[List[str]] = None,
) -> None:
    """
    Plot actual vs predicted for each outcome column (or a subset).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    df = episode_to_dataframe(ep)
    md = AnalysisConfig.metadata

    cols = md.outcome_columns if focus_columns is None else focus_columns

    for col in cols:
        true_col = f"{col}_true"
        pred_col = f"{col}_pred"
        if true_col not in df.columns or pred_col not in df.columns:
            continue

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df.index, df[true_col], label="Actual", linewidth=1.5)
        ax.plot(
            df.index, df[pred_col], label="Simulated", linestyle="--", linewidth=1.5
        )

        ax.set_title(f"{ep.geo_id} – {col}: Actual vs Simulated")
        ax.set_xlabel("Date")
        ax.set_ylabel(col)
        ax.legend(loc="best")
        ax.grid(True, linestyle=":", alpha=0.4)

        fig.autofmt_xdate()
        out_path = output_dir / f"{ep.geo_id}_{col}_actual_vs_sim.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)


def plot_reward(ep: EpisodeResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = episode_to_dataframe(ep)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df["reward_actual"], label="Actual reward", linewidth=1.5)
    ax.plot(df.index, df["reward_simulated"], label="Simulated reward", linestyle="--", linewidth=1.5)

    ax.set_title(f"{ep.geo_id} – Reward over time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Reward")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="best")

    fig.autofmt_xdate()
    out_path = output_dir / f"{ep.geo_id}_reward.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
