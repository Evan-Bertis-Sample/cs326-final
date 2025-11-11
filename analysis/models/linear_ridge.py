from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np

from analysis.predict import *


class PersistenceBaseline(PredictorModel):
    """Predict next outcomes = last observed outcomes in the window."""

    def name(self) -> str:
        return "persistence"

    def fit_batch(self, batch: List[Tuple[ModelInputs, ModelOutput]]) -> None:
        # No learnable parameters.
        return

    def predict(self, x: ModelInputs) -> ModelOutput:
        y = x.outcome_history[-1, :].astype(float, copy=False)
        return ModelOutput(pred_date=x.end_date + np.timedelta64(x.horizon, "D"),
                           outcomes=y)


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


class LinearWindowRegressor(PredictorModel):
    """
    Multi-output linear ridge regressor on flattened histories + meta.

    Feature vector:
      [ meta (M),
        flatten(outcome_history) (W*O),
        flatten(policy_history)  (W*P) ]

    Y target:
      outcomes (O,)

    Closed-form ridge (no sklearn); standardizes X; keeps a bias term.
    """

    def __init__(self, l2: float = 1e-3, use_meta: bool = True, use_policy: bool = True, use_outcome_hist: bool = True):
        self.l2 = float(l2)
        self.use_meta = bool(use_meta)
        self.use_policy = bool(use_policy)
        self.use_outcome_hist = bool(use_outcome_hist)

        self._scaler: Optional[_Scaler] = None
        self._W: Optional[np.ndarray] = None  # (F+1, O) including bias row
        self._out_dim: Optional[int] = None   # O

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
            # fall back to zeros if all disabled (shouldn't happen)
            feats.append(np.zeros(1, dtype=float))
        return np.concatenate(feats, axis=0)

    def _stack_XY(self, batch: List[Tuple[ModelInputs, ModelOutput]]) -> Tuple[np.ndarray, np.ndarray]:
        X = np.stack([self._featurize_one(x) for (x, _) in batch], axis=0)  # (N, F)
        Y = np.stack([y.outcomes for (_, y) in batch], axis=0).astype(float)  # (N, O)
        return X, Y

    def fit_batch(self, batch: List[Tuple[ModelInputs, ModelOutput]]) -> None:
        if not batch:
            return

        X, Y = self._stack_XY(batch)  # (N, F), (N, O)
        self._out_dim = Y.shape[1]

        # scale features
        self._scaler = _Scaler.fit(X)
        Z = self._scaler.transform(X)  # (N, F)

        # add bias
        ones = np.ones((Z.shape[0], 1), dtype=Z.dtype)
        Zb = np.concatenate([ones, Z], axis=1)  # (N, F+1)

        # ridge: (Zb^T Zb + λ * R) W = Zb^T Y
        # do NOT regularize bias (index 0)
        Fp1 = Zb.shape[1]
        R = np.eye(Fp1, dtype=Zb.dtype)
        R[0, 0] = 0.0

        XtX = Zb.T @ Zb + self.l2 * R          # (F+1, F+1)
        XtY = Zb.T @ Y                          # (F+1, O)
        self._W = np.linalg.solve(XtX, XtY)     # (F+1, O)

    def predict(self, x: ModelInputs) -> ModelOutput:
        if self._W is None or self._scaler is None or self._out_dim is None:
            # if untrained, fall back to persistence
            y = x.outcome_history[-1, :].astype(float, copy=False)
            return ModelOutput(pred_date=x.end_date + np.timedelta64(x.horizon, "D"),
                               outcomes=y)

        f = self._featurize_one(x)[None, :]     # (1, F)
        z = self._scaler.transform(f)           # (1, F)
        zb = np.concatenate([np.ones((1, 1), dtype=z.dtype), z], axis=1)  # (1, F+1)
        y = (zb @ self._W).ravel()              # (O,)
        return ModelOutput(pred_date=x.end_date + np.timedelta64(x.horizon, "D"),
                           outcomes=y)
