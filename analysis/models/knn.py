# analysis/models/knn.py
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


class KNNWindowRegressor(BasePredictorModel):
    def __init__(self, **params: Any):
        super().__init__()
        self.k: int = 5
        self.use_meta: bool = True  # optional; policy & outcome are mandatory
        self.metric: str = "euclidean"

        super().set_hyperparameters(
            k=self.k, use_meta=self.use_meta, metric=self.metric
        )
        if params:
            self.set_hyperparameters(**params)

        self._scaler: Optional[_Scaler] = None
        self._X: Optional[np.ndarray] = None
        self._Y: Optional[np.ndarray] = None

    def name(self) -> str:
        return "knn_window"

    def _featurize_one(self, x: ModelInputs) -> np.ndarray:
        feats: List[np.ndarray] = []
        if self.use_meta:
            feats.append(np.asarray(x.meta, dtype=float).ravel())
        feats.append(np.asarray(x.outcome_history, dtype=float).ravel())
        feats.append(np.asarray(x.policy_history, dtype=float).ravel())
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
        self._scaler = _Scaler.fit(X)
        self._X = self._scaler.transform(X)
        self._Y = Y

    def predict(self, x: ModelInputs) -> ModelOutput:
        if self._X is None or self._Y is None or self._scaler is None:
            y = x.outcome_history[-1, :].astype(float, copy=False)
            return ModelOutput(x.end_date + np.timedelta64(x.horizon, "D"), y)

        f = self._featurize_one(x)[None, :]
        z = self._scaler.transform(f)
        diff = self._X - z
        dists = np.sqrt(np.sum(diff * diff, axis=1))

        k = min(int(self.k), self._X.shape[0])
        nn_idx = np.argpartition(dists, kth=k - 1)[:k]
        y = self._Y[nn_idx].mean(axis=0)
        return ModelOutput(x.end_date + np.timedelta64(x.horizon, "D"), y)

    def set_hyperparameters(self, **params: Any) -> None:
        if "k" in params:
            self.k = int(params["k"])
        if "use_meta" in params:
            self.use_meta = bool(params["use_meta"])
        if "metric" in params:
            self.metric = str(params["metric"])
        super().set_hyperparameters(**params)

    def get_hyperparameters(self) -> Dict[str, List[Any]]:
        return {
            "k": [1, 3, 5, 10, 20],
            "use_meta": [True, False],
            "metric": ["euclidean"],
        }
