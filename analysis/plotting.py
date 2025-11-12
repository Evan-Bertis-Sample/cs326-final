# analysis/plotting.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable, Any
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

from analysis.config import AnalysisConfig
from analysis.predict import ModelInputs, ModelOutput, PredictorModel

Pair = Tuple[ModelInputs, ModelOutput]

matplotlib.use("Agg")  # fast, non-interactive backend


class ModelGrapher:
    def __init__(
        self,
        model: PredictorModel,
        *,
        out_root: Path | str = "models",
        hyperparams: Optional[Dict[str, Any]] = None,
        window_size: Optional[int] = None,
        geo_max: Optional[int] = None,
        cluster_file : Optional[str] = None
    ):
        self.model = model
        self.out_root = Path(out_root)
        self.hyperparams = hyperparams or {}
        self.window_size = window_size
        self.geo_max = geo_max
        self.cluster_file = cluster_file.split('.')[0] if cluster_file is not None else None

        # Precompute dirs
        self.model_root = self.out_root / self.model.name()
        self.base_dir = self.model_root
        if self.cluster_file is not None:
            self.base_dir = self.cluster_file / self.base_dir
        if self.window_size is not None:
            self.base_dir = self.base_dir / f"window_{self.window_size}"
        if self.geo_max is not None:
            self.base_dir = self.base_dir / f"geo_max_{self.geo_max}"

        self._save_hyperparams_json()

    @staticmethod
    def _sanitize_name(s: str, maxlen: int = 64) -> str:
        s = (s or "unknown").strip().replace("\\", "_").replace("/", "_")
        s = s.replace(":", "_").replace("?", "_").replace("*", "_").replace("|", "_")
        return s[:maxlen] or "unknown"

    @staticmethod
    def _ensure_dt(dts: Iterable[pd.Timestamp]) -> np.ndarray:
        return np.array(pd.to_datetime(list(dts), errors="coerce"))

    def _save_hyperparams_json(self) -> None:
        self.model_root.mkdir(parents=True, exist_ok=True)
        with (self.model_root / "hyperparameters.json").open("w", encoding="utf-8") as f:
            json.dump(self.hyperparams, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _as_compact_params(hp: Dict[str, Any]) -> str:
        if not hp:
            return "{}"
        parts = [f"{k}={hp[k]!r}" for k in sorted(hp.keys())]
        s = ", ".join(parts)
        return s if len(s) <= 120 else (s[:117] + "...")

    @staticmethod
    def _group_by_geo(pairs: List[Pair]) -> Dict[str, List[Pair]]:
        grouped: Dict[str, List[Pair]] = {}
        for xin, yout in pairs:
            gid = ModelGrapher._sanitize_name(xin.geo_id)
            grouped.setdefault(gid, []).append((xin, yout))
        return grouped

    def _plot_one_outcome(
        self,
        dates: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        *,
        split_name: str,
        geo_id: str,
        outcome_name: str,
    ) -> None:
        out_dir = self.base_dir / split_name / self._sanitize_name(geo_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        diff = y_pred - y_true

        plt.figure(figsize=(9, 4.8))
        plt.plot(dates, y_true, label="Actual")
        plt.plot(dates, y_pred, label="Predicted")
        plt.plot(dates, diff, label="Diff (Pred - Actual)")
        plt.xlabel("Date")
        plt.ylabel(outcome_name)
        plt.legend(loc="best")
        plt.title(f"{outcome_name} — {self.model.name()} [{split_name}]")

        hp_text = self._as_compact_params(self.hyperparams)
        plt.gcf().text(0.99, 0.01, hp_text, ha="right", va="bottom", fontsize=8)

        plt.tight_layout()
        plt.savefig(out_dir / f"{self._sanitize_name(outcome_name)}.png", dpi=150)
        plt.close()

    def _plot_split(self, pairs: Optional[List[Pair]], split_name: str) -> None:
        if not pairs:
            return

        outcome_names = AnalysisConfig.metadata.outcome_columns
        grouped = self._group_by_geo(pairs)

        for geo_id_sanitized, plist in grouped.items():
            # sort by prediction date to keep lines monotonic in time
            plist_sorted = sorted(plist, key=lambda pr: pd.to_datetime(pr[1].pred_date))

            # extract series once
            dates = self._ensure_dt([p[1].pred_date for p in plist_sorted])
            Y_true = np.vstack([p[1].outcomes for p in plist_sorted]).astype(float)

            # predict once per pair
            Y_pred = np.vstack([self.model.predict(p[0]).outcomes for p in plist_sorted]).astype(float)

            # plot each outcome
            for j, outcome_name in enumerate(outcome_names):
                y_t = np.nan_to_num(Y_true[:, j], nan=0.0)
                y_p = np.nan_to_num(Y_pred[:, j], nan=0.0)
                self._plot_one_outcome(
                    dates,
                    y_t,
                    y_p,
                    split_name=split_name,
                    geo_id=geo_id_sanitized,
                    outcome_name=outcome_name,
                )

    def plot_all(self, pairset: "ModelTrainingPairSet") -> None:
        """Plot training, testing, and validation results."""
        self._plot_split(pairset.training, split_name="training")
        self._plot_split(pairset.testing, split_name="testing")
        self._plot_split(pairset.validation, split_name="validation")
