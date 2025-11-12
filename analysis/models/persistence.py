from __future__ import annotations
from typing import Any, Dict, List, Tuple

import numpy as np

from analysis.predict import ModelInputs, ModelOutput, PredictorModel
from analysis.models.base import BasePredictorModel  # put BasePredictorModel here


class PersistenceBaseline(BasePredictorModel):
    def __init__(self, **params: Any):
        super().__init__()
        # no learnable params, but keep interface consistent
        if params:
            self.set_hyperparameters(**params)

    def name(self) -> str:
        return "persistence"

    def fit_batch(self, batch: List[Tuple[ModelInputs, ModelOutput]]) -> None:
        return  # nothing to learn

    def predict(self, x: ModelInputs) -> ModelOutput:
        y = x.outcome_history[-1, :].astype(float, copy=False)
        return ModelOutput(
            pred_date=x.end_date + np.timedelta64(x.horizon, "D"),
            outcomes=y,
        )

    def get_hyperparameters(self) -> Dict[str, List[Any]]:
        # No tunables; keep a stable API for search code
        return {"noop": [None]}
