# analysis/predict_api.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Protocol
import numpy as np
import pandas as pd

from analysis.config import AnalysisConfig
from analysis.oxcgrt_data import OxCGRTData


@dataclass(frozen=True)
class ModelInputs:
    meta: np.ndarray  # shape (M,)
    policy_history: np.ndarray  # shape (W, P)
    outcome_history: np.ndarray  # shape (W, O)
    end_date: pd.Timestamp
    horizon: int  # usually 1


@dataclass(frozen=True)
class ModelOutput:
    pred_date: pd.Timestamp
    outcomes: np.ndarray  # shape (O,)


@dataclass(frozen=True)
class ModelError:
    inputs: ModelInputs
    y_true: np.ndarray  # shape (O,)
    y_pred: np.ndarray  # shape (O,)
    diff: np.ndarray  # y_pred - y_true (shape O,)


@dataclass(frozen=True)
class ModelPerformanceMetrics:
    mae: float
    mse: float
    rmse: float
    mape: float
    r2: float
    outcome_diffs: List[ModelError]


class PredictorModel(Protocol):
    def name(self) -> str: ...
    def fit_batch(self, batch: List[Tuple[ModelInputs, ModelOutput]]) -> None: ...
    def predict(self, x: ModelInputs) -> ModelOutput: ...


class ModelIOPairBuilder:
    @staticmethod
    def build_pairs(
        data: OxCGRTData,
        window_size: int,
        horizon: int = 1,
        max_per_geo: Optional[int] = None,
    ) -> List[Tuple[ModelInputs, ModelOutput]]:
        md = AnalysisConfig.metadata
        date_col = md.date_column
        geoid_col = getattr(md, "geo_id_column", "GeoID")

        df = data.data.copy()
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

        def _coerce_window(frame: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
            # Coerce to numeric, then ffill/bfill within the window
            w = frame[cols].apply(pd.to_numeric, errors="coerce")
            w = w.ffill().bfill()
            return w

        pairs: List[Tuple[ModelInputs, ModelOutput]] = []

        for geo, gdf in df.groupby(geoid_col):
            gdf = gdf.sort_values(date_col).reset_index(drop=True)
            if len(gdf) <= window_size + horizon:
                continue

            # candidate indices where target exists at t + horizon
            start = window_size
            stop = len(gdf) - horizon + 1
            if max_per_geo and max_per_geo > 0:
                idxs = np.linspace(
                    start, stop - 1, num=min(max_per_geo, stop - start), dtype=int
                )
            else:
                idxs = range(start, stop)

            for i in idxs:
                end_date = pd.to_datetime(gdf.iloc[i - 1][date_col])
                pred_date = end_date + pd.Timedelta(days=horizon)

                tgt_row = gdf[gdf[date_col] == pred_date]
                if tgt_row.empty:
                    continue

                hist = gdf.iloc[i - window_size : i]

                # Coerce numeric windows
                pol_win = _coerce_window(hist, md.policy_columns)
                out_win = _coerce_window(hist, md.outcome_columns)

                # If after ffill/bfill the window still has NaNs, skip
                if pol_win.isna().any().any() or out_win.isna().any().any():
                    continue

                # Target vector (coerce once)
                y = (
                    tgt_row[md.outcome_columns]
                    .apply(pd.to_numeric, errors="coerce")
                    .iloc[0]
                    .to_numpy(dtype=float)
                )
                if np.isnan(y).any():
                    continue

                # Meta (optional, numeric only)
                meta_cols = getattr(md, "extra_columns", [])
                if meta_cols:
                    meta_series = hist[meta_cols].iloc[-1]
                    meta_vec = pd.to_numeric(meta_series, errors="coerce").to_numpy(
                        dtype=float
                    )
                    # if all NaN, drop meta
                    if np.isnan(meta_vec).all():
                        meta_vec = np.zeros((0,), dtype=float)
                else:
                    meta_vec = np.zeros((0,), dtype=float)

                xin = ModelInputs(
                    meta=meta_vec,
                    policy_history=pol_win.to_numpy(dtype=float),
                    outcome_history=out_win.to_numpy(dtype=float),
                    end_date=end_date,
                    horizon=horizon,
                )
                yout = ModelOutput(pred_date=pred_date, outcomes=y)
                pairs.append((xin, yout))

        return pairs


class OutcomePredictor:
    def __init__(self, model: PredictorModel):
        self.model = model
        md = AnalysisConfig.metadata
        self.date_col = md.date_column
        self.outcome_cols = md.outcome_columns

    def predict(self, inputs: ModelInputs) -> ModelOutput:
        return self.model.predict(inputs)

    def evaluate(
        self, tests: List[Tuple[ModelInputs, ModelOutput]]
    ) -> ModelPerformanceMetrics:
        if not tests:
            return ModelPerformanceMetrics(
                mae=np.nan,
                mse=np.nan,
                rmse=np.nan,
                mape=np.nan,
                r2=np.nan,
                outcome_diffs=[],
            )

        errs: List[ModelError] = []
        y_true_all: List[np.ndarray] = []
        y_pred_all: List[np.ndarray] = []

        for xin, ytrue in tests:
            yhat = self.model.predict(xin)
            yt = ytrue.outcomes.astype(float)
            yp = yhat.outcomes.astype(float)
            diff = yp - yt
            errs.append(ModelError(inputs=xin, y_true=yt, y_pred=yp, diff=diff))
            y_true_all.append(yt)
            y_pred_all.append(yp)

        yt = np.vstack(y_true_all).reshape(-1)
        yp = np.vstack(y_pred_all).reshape(-1)

        mask = ~np.isnan(yt) & ~np.isnan(yp)
        yt, yp = yt[mask], yp[mask]
        if yt.size == 0:
            return ModelPerformanceMetrics(
                mae=np.nan,
                mse=np.nan,
                rmse=np.nan,
                mape=np.nan,
                r2=np.nan,
                outcome_diffs=errs,
            )

        mae = float(np.mean(np.abs(yt - yp)))
        mse = float(np.mean((yt - yp) ** 2))
        rmse = float(np.sqrt(mse))
        mape = float(np.mean(np.abs((yt - yp) / np.maximum(np.abs(yt), 1e-8))) * 100.0)
        ss_res = float(np.sum((yt - yp) ** 2))
        ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

        return ModelPerformanceMetrics(
            mae=mae, mse=mse, rmse=rmse, mape=mape, r2=r2, outcome_diffs=errs
        )
