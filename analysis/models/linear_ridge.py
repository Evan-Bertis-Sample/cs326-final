# analysis/models/linear_ridge.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
from scipy import signal, linalg

from analysis.predict import ModelInputs, ModelOutput
from analysis.models.base import BasePredictorModel


class _Scaler:
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean
        self.std = std

    @classmethod
    def fit(cls, X: np.ndarray) -> "_Scaler":
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0.0] = 1.0
        return cls(mean, std)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std


class LinearWindowRegressor(BasePredictorModel):
    def __init__(self, **params: Any):
        super().__init__()
        self.l2: float = 1e-3
        self.use_meta: bool = True

        # filtering hyperparameters for Y targets
        self.filter_type: str = "none"       # "none" | "ema" | "boxcar"
        self.filter_alpha: float = 0.3       # for EMA
        self.filter_window: int = 7          # for boxcar

        super().set_hyperparameters(
            l2=self.l2,
            use_meta=self.use_meta,
            filter_type=self.filter_type,
            filter_alpha=self.filter_alpha,
            filter_window=self.filter_window,
        )
        if params:
            self.set_hyperparameters(**params)

        self._scaler: Optional[_Scaler] = None
        self._W: Optional[np.ndarray] = None
        self._out_dim: Optional[int] = None

    def name(self) -> str:
        return "linear_window_ridge"

    # filtering applied to outputs Y
    def _filter_targets_1geo(self, Y_geo: np.ndarray) -> np.ndarray:
        if Y_geo.size == 0:
            return Y_geo

        ftype = self.filter_type.lower()
        if ftype == "none":
            return Y_geo

        if ftype == "ema":
            alpha = max(1e-4, min(float(self.filter_alpha), 1.0))
            b = [alpha]
            a = [1.0, -(1.0 - alpha)]
            return signal.lfilter(b, a, Y_geo, axis=0)

        if ftype == "boxcar":
            win = max(int(self.filter_window), 1)
            win = min(win, Y_geo.shape[0])
            b = np.ones(win) / float(win)
            a = [1.0]
            return signal.lfilter(b, a, Y_geo, axis=0)

        return Y_geo

    def _featurize_one(self, x: ModelInputs) -> np.ndarray:
        feats: List[np.ndarray] = []
        if self.use_meta:
            feats.append(np.asarray(x.meta, dtype=float).ravel())
        feats.append(np.asarray(x.outcome_history, dtype=float).ravel())
        feats.append(np.asarray(x.policy_history, dtype=float).ravel())
        return np.concatenate(feats, axis=0)

    def _stack_XY_geo(
        self, batch: List[Tuple[ModelInputs, ModelOutput]]
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        X = np.stack([self._featurize_one(x) for (x, _) in batch], axis=0)
        Y = np.stack([y.outcomes for (_, y) in batch], axis=0).astype(float)
        geo_ids = [x.geo_id for (x, _) in batch]
        return X, Y, geo_ids

    def fit_batch(self, batch: List[Tuple[ModelInputs, ModelOutput]]) -> None:
        if not batch:
            return

        # get raw feature/target data first
        X, Y, geo_ids = self._stack_XY_geo(batch)
        self._out_dim = Y.shape[1]

        # per-geo filtering of Y
        if self.filter_type.lower() != "none":
            Y_filt = np.empty_like(Y)
            idx_by_geo: Dict[str, List[int]] = {}

            for i, gid in enumerate(geo_ids):
                idx_by_geo.setdefault(str(gid), []).append(i)

            for gid, idxs in idx_by_geo.items():
                idxs_arr = np.array(idxs)
                Y_geo = Y[idxs_arr]
                Y_geo_f = self._filter_targets_1geo(Y_geo)
                Y_filt[idxs_arr] = Y_geo_f

            Y = Y_filt

        # scaling and ridge regression
        self._scaler = _Scaler.fit(X)
        Z = self._scaler.transform(X)

        ones = np.ones((Z.shape[0], 1), dtype=Z.dtype)
        Zb = np.concatenate([ones, Z], axis=1)

        R = np.eye(Zb.shape[1])
        R[0, 0] = 0.0

        XtX = Zb.T @ Zb + float(self.l2) * R
        XtY = Zb.T @ Y

        self._W = linalg.solve(XtX, XtY, assume_a="sym")

    def predict(self, x: ModelInputs) -> ModelOutput:
        if self._W is None or self._scaler is None or self._out_dim is None:
            y = x.outcome_history[-1, :].astype(float, copy=False)
            return ModelOutput(
                x.end_date + np.timedelta64(x.horizon, "D"),
                y,
            )
        f = self._featurize_one(x)[None, :]
        z = self._scaler.transform(f)
        zb = np.concatenate([np.ones((1, 1), dtype=z.dtype), z], axis=1)
        y = (zb @ self._W).ravel()
        return ModelOutput(
            x.end_date + np.timedelta64(x.horizon, "D"),
            y,
        )

    def set_hyperparameters(self, **params: Any) -> None:
        if "l2" in params:
            self.l2 = float(params["l2"])
        if "use_meta" in params:
            self.use_meta = bool(params["use_meta"])
        if "filter_type" in params:
            self.filter_type = str(params["filter_type"])
        if "filter_alpha" in params:
            self.filter_alpha = float(params["filter_alpha"])
        if "filter_window" in params:
            self.filter_window = int(params["filter_window"])
        super().set_hyperparameters(**params)

    def get_hyperparameters(self) -> Dict[str, List[Any]]:
        return {
            "l2": [1e-5, 1e-4, 1e-3, 1e-2],
            "use_meta": [True, False],
            "filter_type": ["none", "ema", "boxcar"],
            "filter_alpha": [0.2, 0.5, 0.8],
            "filter_window": [3, 7, 14],
        }
