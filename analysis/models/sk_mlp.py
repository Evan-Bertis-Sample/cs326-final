# analysis/models/sk_mlp.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
from sklearn.neural_network import MLPRegressor

from analysis.predict import ModelInputs, ModelOutput, PredictorModel
from analysis.models.base import BasePredictorModel


class SKMLPRegressor(BasePredictorModel):
    def __init__(self, **params: Any):
        super().__init__()
        # defaults
        self.use_meta: bool = True
        self.hidden_layer_sizes = (128, 64)
        self.alpha = 1e-4
        self.learning_rate_init = 1e-3
        self.max_iter = 200
        self.random_state = 42
        self.batch_size = "auto"
        # store defaults, then apply overrides
        super().set_hyperparameters(
            use_meta=self.use_meta,
            hidden_layer_sizes=self.hidden_layer_sizes,
            alpha=self.alpha,
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
            random_state=self.random_state,
            batch_size=self.batch_size,
        )
        if params:
            self.set_hyperparameters(**params)

        self._mlp: Optional[MLPRegressor] = None
        self._out_dim: Optional[int] = None

    def name(self) -> str:
        return "sk_mlp"

    def _featurize_one(self, x: ModelInputs) -> np.ndarray:
        feats: List[np.ndarray] = []
        if self.use_meta:
            feats.append(np.asarray(x.meta, dtype=float).ravel())
        feats.append(np.asarray(x.outcome_history, dtype=float).ravel())  # REQUIRED
        feats.append(np.asarray(x.policy_history, dtype=float).ravel())  # REQUIRED
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
        self._mlp = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            alpha=self.alpha,
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
            random_state=self.random_state,
            batch_size=self.batch_size,
            n_iter_no_change=10,
            early_stopping=True,
            validation_fraction=0.1,
        )
        # sklearn expects 1D or 2D target; for multi-output, pass (N, O)
        self._mlp.fit(X, Y)

    def predict(self, x: ModelInputs) -> ModelOutput:
        if self._mlp is None or self._out_dim is None:
            # fallback: persistence
            y = x.outcome_history[-1, :].astype(float, copy=False)
            return ModelOutput(x.end_date + np.timedelta64(x.horizon, "D"), y)
        f = self._featurize_one(x)[None, :]
        y = self._mlp.predict(f).reshape(-1)
        return ModelOutput(x.end_date + np.timedelta64(x.horizon, "D"), y)

    def set_hyperparameters(self, **params: Any) -> None:
        known = {
            "use_meta",
            "hidden_layer_sizes",
            "alpha",
            "learning_rate_init",
            "max_iter",
            "random_state",
            "batch_size",
        }
        for k, v in params.items():
            if k in known:
                setattr(self, k, v)
        super().set_hyperparameters(**params)

    def get_hyperparameters(self) -> Dict[str, List[Any]]:
        return {
            "use_meta": [True, False],
            "hidden_layer_sizes": [(128, 64), (256, 128), (256, 128, 64)],
            "alpha": [1e-5, 1e-4, 1e-3],
            "learning_rate_init": [1e-3, 3e-4],
            "max_iter": [200, 400],
        }
