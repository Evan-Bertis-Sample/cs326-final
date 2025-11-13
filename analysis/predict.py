from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Protocol, Dict, Any
import numpy as np
import pandas as pd
import hashlib
import math
from tqdm import tqdm

from analysis.config import AnalysisConfig
from analysis.oxcgrt_data import OxCGRTData


@dataclass(frozen=True)
class ModelInputs:
    geo_id : str
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
    def fit_batch(self, batch: List[Tuple["ModelInputs", "ModelOutput"]]) -> None: ...
    def predict(self, x: "ModelInputs") -> "ModelOutput": ...
    def set_hyperparameters(self, **params: Any) -> None: ...
    def get_hyperparameters(self) -> Dict[str, List[Any]]: ...
    def __repr__(self) -> str: ...


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
        
    def _to_num(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.apply(pd.to_numeric, errors="coerce")

    def _hash_str(self, s: str) -> float:
        # stable-ish numeric embed for metadata strings
        if s is None:
            return 0.0
        h = hashlib.sha256(str(s).encode("utf-8")).hexdigest()
        return float(int(h[:12], 16))  # compact to float range


    def _date_to_ordinal(self, d: pd.Series) -> np.ndarray:
        d = pd.to_datetime(d, errors="coerce")
        return d.dt.date.map(lambda x: x.toordinal() if pd.notna(x) else 0).to_numpy(dtype=float)


    def _ensure_dt(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        if not pd.api.types.is_datetime64_any_dtype(df[col]):
            df = df.copy()
            df[col] = pd.to_datetime(df[col], errors="coerce")
        return df


    def _encode_meta(self, df: pd.DataFrame) -> np.ndarray:
        # one numeric vector per-geo using first non-null value per metadata column
        cols = AnalysisConfig.metadata.id_columns
        first_row = df.iloc[0][cols]
        out = []
        for c in cols:
            v = first_row.get(c, None)
            # If looks like a date column, encode ordinal; else hash string
            if c == AnalysisConfig.metadata.date_column:
                out.append(self._date_to_ordinal(pd.Series([v]))[0])
            else:
                out.append(self._hash_str("" if pd.isna(v) else str(v)))
        return np.asarray(out, dtype=float)


    def _encode_policies(self, df: pd.DataFrame, start: int, end: int) -> np.ndarray:
        cols = AnalysisConfig.metadata.policy_columns
        date_col = AnalysisConfig.metadata.date_column
        win = df.iloc[start:end].copy()
        # keep only policy columns (numeric 0/1, missing→0)
        pol = self._to_num(win[cols]).fillna(0.0).to_numpy(dtype=float)
        return pol  # shape (W, P)


    def _encode_outcomes(self, df: pd.DataFrame, start: int, end: int) -> np.ndarray:
        cols = AnalysisConfig.metadata.outcome_columns
        win = df.iloc[start:end].copy()

        arr_df = self._to_num(win[cols])
        # forward fill but *don't* replace remaining NaNs with 0
        arr_ff = arr_df.ffill()

        # If any column is still NaN for the *entire* window, we can decide to:
        # 1) treat as constant 0, or
        # 2) skip this pair.
        if arr_ff.isna().all(axis=0).any():
            # option: return None and handle skip in get_pairs
            return None

        arr_ff = arr_ff.fillna(0.0)  # occasional NaN at the very start
        out = arr_ff.to_numpy(dtype=float, copy=True)
        assert out.shape[0] == (end - start)
        return out


    def get_pairs(self, data: OxCGRTData, verbose: bool = True) -> List[Tuple[ModelInputs, ModelOutput]]:
        pairs: List[Tuple[ModelInputs, ModelOutput]] = []

        md = AnalysisConfig.metadata
        date_col = md.date_column

        geo_ids = data.geo_id_strings(unique=True)
        geo_iter = tqdm(geo_ids, desc="Building training pairs", unit="region") if verbose else geo_ids

        skipped = {
            "bad_policy" : 0,
            "bad_outcome" : 0
        }

        for geo_id in geo_iter:
            geo_df = data.get_timeseries(str(geo_id))
            if geo_df.empty:
                continue
            geo_df = self._ensure_dt(geo_df, date_col).sort_values(date_col).reset_index(drop=True)
            encoded_meta = self._encode_meta(geo_df)

            n = len(geo_df)

            step_size = 1
            if self.max_per_geo != None and self.max_per_geo > 0:
                total_possible = n - self.window - self.horizon + 1
                step_size = math.floor(total_possible / self.max_per_geo)
        
            if step_size <= 0:
                step_size = 1

            inner_iter = range(0, n - self.window - self.horizon + 1, step_size)
            if verbose:
                inner_iter = tqdm(inner_iter, leave=False, desc=f"{geo_id[:10]}...", unit="window")

            for i in inner_iter:
                start, end = i, i + self.window
                policy_history = self._encode_policies(geo_df, start, end)
                outcome_history = self._encode_outcomes(geo_df, start, end)

                if policy_history is None:
                    skipped["bad_policy"] += 1
                    continue
                
                if outcome_history is None:
                    skipped["bad_outcome"] += 1
                    continue

                end_date = pd.to_datetime(geo_df.iloc[end - 1][date_col])
                pred_date = end_date + pd.Timedelta(days=self.horizon)

                tgt_row = geo_df.loc[geo_df[date_col] == pred_date]
                if tgt_row.empty:
                    continue
                y = self._to_num(tgt_row[md.outcome_columns]).fillna(0.0).iloc[0].to_numpy(dtype=float)

                xin = ModelInputs(
                    geo_id=geo_id,
                    meta=encoded_meta,
                    policy_history=policy_history,
                    outcome_history=outcome_history,
                    end_date=end_date,
                    horizon=self.horizon,
                )
                yout = ModelOutput(pred_date=pred_date, outcomes=y)
                pairs.append((xin, yout))

        if verbose:
            tqdm.write(f"Finished building {len(pairs):,} training pairs.")
            tqdm.write(f"Pairs skipped due to bad policy: {skipped['bad_policy']}")
            tqdm.write(f"Pairs skipped due to bad outcome: {skipped['bad_outcome']}")


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

        # Shape: (N_samples, O)
        Y_true = np.vstack(y_true_all)
        Y_pred = np.vstack(y_pred_all)

        # Mask NaNs per element
        mask = ~np.isnan(Y_true) & ~np.isnan(Y_pred)
        # if a whole column is invalid, drop it
        valid_cols = mask.any(axis=0)
        if not valid_cols.any():
            return ModelPerformanceMetrics(
                mae=np.nan,
                mse=np.nan,
                rmse=np.nan,
                mape=np.nan,
                r2=np.nan,
                outcome_diffs=errs,
            )

        Yt = Y_true[:, valid_cols]
        Yp = Y_pred[:, valid_cols]
        M = mask[:, valid_cols]

        per_mae = []
        per_mse = []
        per_rmse = []
        per_mape = []
        per_r2 = []

        for j in range(Yt.shape[1]):
            yj_t = Yt[:, j][M[:, j]]
            yj_p = Yp[:, j][M[:, j]]
            if yj_t.size == 0:
                continue

            diff = yj_p - yj_t
            mae_j = float(np.mean(np.abs(diff)))
            mse_j = float(np.mean(diff ** 2))
            rmse_j = float(np.sqrt(mse_j))

            # avoid divide-by-zero for MAPE
            denom = np.maximum(np.abs(yj_t), 1e-8)
            mape_j = float(np.mean(np.abs(diff / denom) * 100.0))

            ss_res_j = float(np.sum(diff ** 2))
            ss_tot_j = float(np.sum((yj_t - np.mean(yj_t)) ** 2))
            r2_j = float(1.0 - ss_res_j / ss_tot_j) if ss_tot_j > 0 else np.nan

            per_mae.append(mae_j)
            per_mse.append(mse_j)
            per_rmse.append(rmse_j)
            per_mape.append(mape_j)
            per_r2.append(r2_j)

        if not per_mae:
            return ModelPerformanceMetrics(
                mae=np.nan,
                mse=np.nan,
                rmse=np.nan,
                mape=np.nan,
                r2=np.nan,
                outcome_diffs=errs,
            )

        mae = float(np.mean(per_mae))
        mse = float(np.mean(per_mse))
        rmse = float(np.mean(per_rmse))
        mape = float(np.mean(per_mape))
        # you could also report median R² if you want robustness
        r2 = float(np.nanmean(per_r2))

        return ModelPerformanceMetrics(
            mae=mae, mse=mse, rmse=rmse, mape=mape, r2=r2, outcome_diffs=errs
        )
