from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from analysis.predict import ModelInputs, ModelOutput
from analysis.models.base import BasePredictorModel  # put BasePredictorModel here


@dataclass
class _Scaler:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std

    @classmethod
    def fit(cls, X: np.ndarray) -> "_Scaler":
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0.0] = 1.0
        return cls(mean=mean, std=std)


class LinearWindowRegressor(BasePredictorModel):
    def __init__(self, **params: Any):
        super().__init__()
        # defaults
        self.l2: float = 1e-3
        self.use_meta: bool = True
        self.use_policy: bool = True
        self.use_outcome_hist: bool = True

        # persist defaults in base dict, then apply user overrides
        super().set_hyperparameters(
            l2=self.l2,
            use_meta=self.use_meta,
            use_policy=self.use_policy,
            use_outcome_hist=self.use_outcome_hist,
        )
        if params:
            self.set_hyperparameters(**params)

        self._scaler: Optional[_Scaler] = None
        self._W: Optional[np.ndarray] = None
        self._out_dim: Optional[int] = None

    def name(self) -> str:
        return "linear_window_ridge"

    def _featurize_one(self, x: ModelInputs) -> np.ndarray:
        feats: List[np.ndarray] = []
        if self.use_meta:
            feats.append(np.asarray(x.meta, dtype=float).ravel())
        if self.use_outcome_hist:
            feats.append(np.asarray(x.outcome_history, dtype=float).ravel())
        if self.use_policy:
            feats.append(np.asarray(x.policy_history, dtype=float).ravel())
        if not feats:
            feats.append(np.zeros(1, dtype=float))
        return np.concatenate(feats, axis=0)

    def _stack_XY(
        self, batch: List[Tuple[ModelInputs, ModelOutput]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        X = np.stack([self._featurize_one(x) for (x, _) in batch], axis=0)
        Y = np.stack([y.outcomes for (_, y) in batch], axis=0).astype(float)
        return X, Y

    def fit_batch(self, batch: List[Tuple[ModelInputs, ModelOutput]]) -> None:
        if not batch:
            return
        X, Y = self._stack_XY(batch)
        self._out_dim = Y.shape[1]
        self._scaler = _Scaler.fit(X)
        Z = self._scaler.transform(X)

        ones = np.ones((Z.shape[0], 1), dtype=Z.dtype)
        Zb = np.concatenate([ones, Z], axis=1)

        Fp1 = Zb.shape[1]
        R = np.eye(Fp1, dtype=Zb.dtype)
        R[0, 0] = 0.0  # don't regularize bias

        XtX = Zb.T @ Zb + float(self.l2) * R
        XtY = Zb.T @ Y
        self._W = np.linalg.solve(XtX, XtY)

    def predict(self, x: ModelInputs) -> ModelOutput:
        if self._W is None or self._scaler is None or self._out_dim is None:
            # fall back to persistence if not trained
            y = x.outcome_history[-1, :].astype(float, copy=False)
            return ModelOutput(
                pred_date=x.end_date + np.timedelta64(x.horizon, "D"),
                outcomes=y,
            )
        f = self._featurize_one(x)[None, :]
        z = self._scaler.transform(f)
        zb = np.concatenate([np.ones((1, 1), dtype=z.dtype), z], axis=1)
        y = (zb @ self._W).ravel()
        return ModelOutput(
            pred_date=x.end_date + np.timedelta64(x.horizon, "D"), outcomes=y
        )

    def set_hyperparameters(self, **params: Any) -> None:
        # set attributes when known, keep base dict in sync
        known = {"l2", "use_meta", "use_policy", "use_outcome_hist"}
        for k, v in params.items():
            if k in known:
                setattr(self, k, v)
        super().set_hyperparameters(**params)

    def get_hyperparameters(self) -> Dict[str, List[Any]]:
        return {
            "l2": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
            "use_meta": [True, False],
            "use_policy": [True, False],
            "use_outcome_hist": [True, False],
        }
