# analysis/models/linear_decay_ridge.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional
import numpy as np

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


class LinearDecayRidge(BasePredictorModel):
    def __init__(self, **params: Any):
        super().__init__()
        self.l2: float = 1e-3
        self.tau_days: float = 30.0
        self.use_meta: bool = True  # optional

        super().set_hyperparameters(
            l2=self.l2, tau_days=self.tau_days, use_meta=self.use_meta
        )
        if params:
            self.set_hyperparameters(**params)

        self._scaler: Optional[_Scaler] = None
        self._W: Optional[np.ndarray] = None
        self._out_dim: Optional[int] = None

    def name(self) -> str:
        return "linear_decay_ridge"

    def _featurize_one(self, x: ModelInputs) -> np.ndarray:
        feats: List[np.ndarray] = []
        if self.use_meta:
            feats.append(np.asarray(x.meta, dtype=float).ravel())
        feats.append(np.asarray(x.outcome_history, dtype=float).ravel())  # REQUIRED
        feats.append(np.asarray(x.policy_history, dtype=float).ravel())  # REQUIRED
        return np.concatenate(feats, axis=0)

    def _stack_XY_dates(self, batch: List[Tuple[ModelInputs, ModelOutput]]):
        X = np.stack([self._featurize_one(x) for (x, _) in batch], axis=0)
        Y = np.stack([y.outcomes for (_, y) in batch], axis=0).astype(float)
        dates = np.array([x.end_date.toordinal() for (x, _) in batch], dtype=float)
        return X, Y, dates

    def fit_batch(self, batch: List[Tuple[ModelInputs, ModelOutput]]) -> None:
        if not batch:
            return
        X, Y, dates_ord = self._stack_XY_dates(batch)
        self._out_dim = Y.shape[1]
        self._scaler = _Scaler.fit(X)
        Z = self._scaler.transform(X)

        ones = np.ones((Z.shape[0], 1), dtype=Z.dtype)
        Zb = np.concatenate([ones, Z], axis=1)

        t_ref = np.max(dates_ord)
        deltas = t_ref - dates_ord
        tau = max(float(self.tau_days), 1e-6)
        w = np.exp(-deltas / tau)  # (N,)

        Wmat = np.diag(w)
        R = np.eye(Zb.shape[1], dtype=Zb.dtype)
        R[0, 0] = 0.0

        XtWX = Zb.T @ Wmat @ Zb + float(self.l2) * R
        XtWY = Zb.T @ Wmat @ Y
        self._W = np.linalg.solve(XtWX, XtWY)

    def predict(self, x: ModelInputs) -> ModelOutput:
        if self._W is None or self._scaler is None or self._out_dim is None:
            y = x.outcome_history[-1, :].astype(float, copy=False)
            return ModelOutput(x.end_date + np.timedelta64(x.horizon, "D"), y)
        f = self._featurize_one(x)[None, :]
        z = self._scaler.transform(f)
        zb = np.concatenate([np.ones((1, 1), dtype=z.dtype), z], axis=1)
        y = (zb @ self._W).ravel()
        return ModelOutput(x.end_date + np.timedelta64(x.horizon, "D"), y)

    def set_hyperparameters(self, **params: Any) -> None:
        if "l2" in params:
            self.l2 = float(params["l2"])
        if "tau_days" in params:
            self.tau_days = float(params["tau_days"])
        if "use_meta" in params:
            self.use_meta = bool(params["use_meta"])
        super().set_hyperparameters(**params)

    def get_hyperparameters(self) -> Dict[str, List[Any]]:
        return {
            "l2": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
            "tau_days": [7.0, 14.0, 30.0, 60.0, 120.0],
            "use_meta": [True, False],
        }
