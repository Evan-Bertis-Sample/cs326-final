from __future__ import annotations
from typing import List, Tuple, Optional
from analysis.predict import *
import pandas as pd

class PersistenceBaseline(PredictorModel):
    def __init__(self, **params: Any):
        self._params: Dict[str, Any] = {}
        if params:
            self.set_hyperparameters(**params)

    def name(self) -> str:
        return "persistence"

    def fit_batch(self, batch: List[Tuple[ModelInputs, ModelOutput]]) -> None:
        return

    def predict(self, x: ModelInputs) -> ModelOutput:
        y = x.outcome_history[-1, :].astype(float, copy=False)
        return ModelOutput(
            pred_date=x.end_date + np.timedelta64(x.horizon, "D"),
            outcomes=y,
        )

    def set_hyperparameters(self, **params: Any) -> None:
        self._params.update(params)

    def get_hyperparameters(self) -> Dict[str, List[Any]]:
        # No tunable hyperparams, but keep consistent interface
        return {"noop": [None]}