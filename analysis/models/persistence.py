from __future__ import annotations
from typing import List, Tuple, Optional
from analysis.predict import *
import pandas as pd

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