# analysis/predict_api.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Iterable, Optional, Protocol
import numpy as np
import pandas as pd

from analysis.config import AnalysisConfig
from analysis.oxcgrt_data import OxCGRTData


@dataclass(frozen=True)
class ModelInputs:
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
    query : np.array
    outcome_error : pd.errors

@dataclass(frozen=True)
class ModelPerformanceMetrics:
    mae : float
    mse : float
    rmse : float
    mape : float
    r2 : float
    outcome_diffs : np.ndarray # array of Model_Errors for each prediction


class PredictorModel(Protocol):
    def name(self) -> str: ...
    def fit_batch(self, batch: List[Tuple[ModelInputs, ModelOutput]]) -> None: ...
    def predict(self, x: ModelInputs) -> ModelOutput: ...


class OutcomePredictor:
    def __init__(self, model: PredictorModel):
        self.model = model

    def _ensure_dt(self, df: pd.DataFrame) -> pd.DataFrame:
        if not pd.api.types.is_datetime64_any_dtype(df[self.date_col]):
            df = df.copy()
            df[self.date_col] = pd.to_datetime(df[self.date_col], errors="coerce")
        return df

    def predict(self, input : ModelInputs) -> ModelOutput:
        return self.model.predict(input)

    def evaluate(self, tests : List[ModelInputs, ModelOutput]) -> ModelPerformanceMetrics:
        pass


class PersistenceModel:
    def name(self) -> str:
        return "persistence"

    def train(self, training_data : List[Tuple[ModelInputs, ModelOutput]]) -> None:
        return  # no learnable params

    def predict(self, x: ModelInputs) -> ModelOutput:
        # last row of outcome_history
        y = x.outcome_history[-1, :].astype(float)
        return ModelOutput(
            pred_date=x.end_date + pd.Timedelta(days=x.horizon), outcomes=y
        )