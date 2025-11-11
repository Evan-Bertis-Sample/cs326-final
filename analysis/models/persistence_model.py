from analysis.predict import *
import pandas as pd

class PersistenceModel:
    def name(self) -> str:
        return "persistence"

    def fit_batch(self, batch: List[Tuple[ModelInputs, ModelOutput]]) -> None:
        return  # no learnable params

    def predict(self, x: ModelInputs) -> ModelOutput:
        y = x.outcome_history[-1, :].astype(float)
        return ModelOutput(pred_date=x.end_date + pd.Timedelta(days=x.horizon), outcomes=y)