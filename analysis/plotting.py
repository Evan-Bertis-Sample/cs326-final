# analysis/plotting.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable, Any
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import datetime

from analysis.predict import (
    ModelInputs,
    ModelOutput,
    PredictorModel,
    OutcomePredictor,
    ModelPerformanceMetrics,
)

from analysis.config import AnalysisConfig
from analysis.predict import ModelInputs, ModelOutput, PredictorModel
from analysis.io import banner

Pair = Tuple[ModelInputs, ModelOutput]

matplotlib.use("Agg")  # fast, non-interactive backend


class ModelGrapher:
    def __init__(
        self,
        model: PredictorModel,
        *,
        out_root: Path | str = "figures",
        hyperparams: Optional[Dict[str, Any]] = None,
        window_size: Optional[int] = None,
        geo_max: Optional[int] = None,
        cluster_file: Optional[str | Path] = None
    ):
        self.model = model
        self.out_root = Path(out_root)
        self.hyperparams = hyperparams or {}
        self.window_size = window_size
        self.geo_max = geo_max

        self.hyperparams["geo_max"] = self.geo_max
        self.hyperparams["window_size"] = self.window_size

        # Extract filename stem (without extension)
        self.cluster_name = Path(cluster_file).stem if cluster_file else None

        # Precompute dirs
        self.model_root = self.out_root / self.model.name()
        self.base_dir = self.model_root
        if self.cluster_name is not None:
            self.base_dir = self.base_dir / self.cluster_name
        if self.window_size is not None:
            self.base_dir = self.base_dir / f"window_{self.window_size}"
        if self.geo_max is not None:
            self.base_dir = self.base_dir / f"geo_max_{self.geo_max}"

    @staticmethod
    def _sanitize_name(s: str, maxlen: int = 64) -> str:
        s = (s or "unknown").strip().replace("\\", "_").replace("/", "_")
        s = s.replace(":", "_").replace("?", "_").replace("*", "_").replace("|", "_")
        return s[:maxlen] or "unknown"

    @staticmethod
    def _ensure_dt(dts: Iterable[pd.Timestamp]) -> np.ndarray:
        return np.array(pd.to_datetime(list(dts), errors="coerce"))

    def _write_results_json(
            self,
            geo_ids_seen: list[str],
            metrics_by_split: dict[str, Optional[ModelPerformanceMetrics]],
        ) -> None:
            """
            Write a summary JSON including hyperparameters, splits' scores, and geo coverage.
            """
            out_path = self.base_dir / "results.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)

            def _metrics_dict(m: Optional[ModelPerformanceMetrics]) -> Optional[dict[str, float]]:
                if m is None:
                    return None
                return {
                    "mae": m.mae,
                    "mse": m.mse,
                    "rmse": m.rmse,
                    "mape": m.mape,
                    "r2": m.r2,
                }

            data = {
                "model": self.model.name(),
                "hyperparameters": self.hyperparams,
                "cluster_name": self.cluster_name,
                "window_size": self.window_size,
                "geo_max": self.geo_max,
                "geos": sorted(set(geo_ids_seen)),
                "splits": {
                    split: _metrics_dict(m)
                    for split, m in metrics_by_split.items()
                },
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            }

            with out_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

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
    
    def _needs_plot(self, *, split_name: str, geo_id: str, outcome_name: str) -> bool:
        """
        Returns True if this outcome image should be (re)generated.
        Skips plotting if the PNG already exists and is non-empty.
        """
        out_path = (
            self.base_dir
            / split_name
            / self._sanitize_name(geo_id)
            / f"{self._sanitize_name(outcome_name)}.png"
        )
        # Already plotted? (non-empty file)
        if out_path.exists() and out_path.stat().st_size > 1024:  # ~1KB sanity check
            return False
        return True

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
        plt.title(f"{geo_id} - {outcome_name} - {self.model.name()} [{split_name}]")

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

        for geo_id, plist in grouped.items():
            plist_sorted = sorted(plist, key=lambda pr: pd.to_datetime(pr[1].pred_date))
            dates = self._ensure_dt([p[1].pred_date for p in plist_sorted])
            Y_true = np.vstack([p[1].outcomes for p in plist_sorted]).astype(float)
            Y_pred = np.vstack([self.model.predict(p[0]).outcomes for p in plist_sorted]).astype(float)

            for j, outcome_name in enumerate(outcome_names):
                if not self._needs_plot(split_name=split_name, geo_id=geo_id, outcome_name=outcome_name):
                    continue  # skip — already exists

                y_t = np.nan_to_num(Y_true[:, j], nan=0.0)
                y_p = np.nan_to_num(Y_pred[:, j], nan=0.0)
                self._plot_one_outcome(
                    dates,
                    y_t,
                    y_p,
                    split_name=split_name,
                    geo_id=geo_id,
                    outcome_name=outcome_name,
                )

    @banner(skip_args=("pairset",))
    def plot_all(self, pairset: "ModelTrainingPairSet") -> None:
        """Plot training, testing, and validation results, and write results.json."""
        geo_ids_seen: list[str] = []

        for split_name, pairs in [
            ("training", pairset.training),
            ("testing", pairset.testing),
            ("validation", pairset.validation),
        ]:
            if not pairs:
                continue
            grouped = self._group_by_geo(pairs)
            geo_ids_seen.extend(grouped.keys())
            self._plot_split(pairs, split_name=split_name)

        predictor = OutcomePredictor(self.model)

        def _eval(pairs: Optional[list[Pair]]) -> Optional[ModelPerformanceMetrics]:
            if not pairs:
                return None
            return predictor.evaluate(pairs)

        metrics_by_split: dict[str, Optional[ModelPerformanceMetrics]] = {
            "training": _eval(pairset.training),
            "testing": _eval(pairset.testing),
            "validation": _eval(pairset.validation),
        }

        self._write_results_json(geo_ids_seen, metrics_by_split)

