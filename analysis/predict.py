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
    def __init__(
        self,
        window_size: int,
        horizon: int = 1,
        max_per_geo: Optional[int] = None,
        policy_missing: str = "ffill_then_zero",  # "ffill_then_zero" | "zero" | "drop"
        outcome_missing: str = "ffill",           # "ffill" | "drop"
        verbose: bool = True,
    ):
        self.window = window_size
        self.horizon = horizon
        self.max_per_geo = max_per_geo
        self.policy_missing = policy_missing
        self.outcome_missing = outcome_missing
        self.verbose = verbose

        md = AnalysisConfig.metadata
        self.date_col = md.date_column
        self.geoid_col = getattr(md, "geo_id_column", "GeoID")
        self.policy_cols = md.policy_columns
        self.outcome_cols = md.outcome_columns

    def _prepare_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if not pd.api.types.is_datetime64_any_dtype(df[self.date_col]):
            df = df.copy()
            df[self.date_col] = pd.to_datetime(df[self.date_col], errors="coerce")
        return df

    def _select_eval_indices(self, n_rows: int) -> List[int]:
        start = self.window
        stop = n_rows - self.horizon + 1
        if stop <= start:
            return []
        if self.max_per_geo and self.max_per_geo > 0:
            count = stop - start
            take = min(self.max_per_geo, count)
            return np.linspace(start, stop - 1, num=take, dtype=int).tolist()
        return list(range(start, stop))

    def _encode_policies(self, win: pd.DataFrame) -> Optional[np.ndarray]:
        # policies are 0/1 or missing; coerce safely first
        arr_df = win[self.policy_cols].apply(pd.to_numeric, errors="coerce")
        if self.policy_missing == "drop":
            if arr_df.isna().any().any():
                return None
            return arr_df.to_numpy(dtype=float, copy=True)
        if self.policy_missing == "zero":
            arr = arr_df.to_numpy(dtype=float, copy=True)
            np.nan_to_num(arr, copy=False, nan=0.0)
            return arr
        # "ffill_then_zero"
        filled = arr_df.ffill()
        arr = filled.to_numpy(dtype=float, copy=True)
        np.nan_to_num(arr, copy=False, nan=0.0)
        return arr

    def _encode_outcomes(self, win: pd.DataFrame) -> Optional[np.ndarray]:
        # outcomes may contain strings like "NV" -> coerce to NaN first
        arr_df = win[self.outcome_cols].apply(pd.to_numeric, errors="coerce")
        if self.outcome_missing == "drop":
            if arr_df.isna().any().any():
                return None
            return arr_df.to_numpy(dtype=float, copy=True)
        # "ffill"
        filled = arr_df.ffill()
        if filled.isna().any().any():
            return None
        return filled.to_numpy(dtype=float, copy=True)

    def _build_inputs_from_window(self, hist: pd.DataFrame, end_date: pd.Timestamp) -> Optional[ModelInputs]:
        pol_hist = self._encode_policies(hist)
        if pol_hist is None:
            return None
        out_hist = self._encode_outcomes(hist)
        if out_hist is None:
            return None
        meta_vec = np.zeros((0,), dtype=float)
        return ModelInputs(meta=meta_vec, policy_history=pol_hist, outcome_history=out_hist, end_date=end_date, horizon=self.horizon)

    def _build_target_row(self, gdf: pd.DataFrame, pred_date: pd.Timestamp) -> Optional[ModelOutput]:
        tgt = gdf[gdf[self.date_col] == pred_date]
        if tgt.empty:
            return None
        y = tgt[self.outcome_cols].to_numpy(dtype=float).reshape(-1)
        if np.isnan(y).any():
            return None
        return ModelOutput(pred_date=pred_date, outcomes=y)

    def _build_pair_for_index(self, gdf: pd.DataFrame, i: int) -> Optional[Tuple[ModelInputs, ModelOutput]]:
        end_date = pd.to_datetime(gdf.iloc[i - 1][self.date_col])
        pred_date = end_date + pd.Timedelta(days=self.horizon)
        hist = gdf.iloc[i - self.window : i]
        xin = self._build_inputs_from_window(hist, end_date)
        if xin is None:
            return None
        yout = self._build_target_row(gdf, pred_date)
        if yout is None:
            return None
        return (xin, yout)

    def get_pairs(self, data: OxCGRTData) -> List[Tuple[ModelInputs, ModelOutput]]:
        df = self._prepare_df(data.data.copy())
        pairs: List[Tuple[ModelInputs, ModelOutput]] = []
        skips: Dict[str, int] = {"too_short": 0, "missing_target": 0, "bad_policy": 0, "bad_outcome": 0}

        for _, gdf in df.groupby(self.geoid_col):
            gdf = gdf.sort_values(self.date_col).reset_index(drop=True)
            idxs = self._select_eval_indices(len(gdf))
            if not idxs:
                skips["too_short"] += 1
                continue

            for i in idxs:
                pair = self._build_pair_for_index(gdf, i)
                if pair is None:
                    hist = gdf.iloc[i - self.window : i]
                    if self._encode_policies(hist) is None:
                        skips["bad_policy"] += 1
                    elif self._encode_outcomes(hist) is None:
                        skips["bad_outcome"] += 1
                    else:
                        skips["missing_target"] += 1
                    continue
                pairs.append(pair)

        if self.verbose:
            total_skipped = sum(skips.values())
            print(f"[ModelIOPairBuilder] Built {len(pairs)} pairs out of {len(pairs) + len(skips)} possible (skipped {total_skipped})")
            for k, v in skips.items():
                if v:
                    print(f"  - {k:>15}: {v}")
            print("")

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
