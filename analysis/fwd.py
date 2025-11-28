# fwd.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple

import json
from joblib import load
import numpy as np
import pandas as pd

from analysis.config import AnalysisConfig
from analysis.oxcgrt_data import OxCGRTData
from analysis.predict import (  # adjust import path if needed
    ModelInputs,
    OutcomePredictor,
    ModelIOPairBuilder,
    PredictorModel,
)


@dataclass(frozen=True)
class LoadedModelInfo:
    model: PredictorModel
    predictor: OutcomePredictor
    hyperparams: Dict[str, Any]


class ModelForwarder:
    def __init__(self, data: OxCGRTData, model_map_path: Path | None = None):
        self.data = data
        if model_map_path is None:
            model_map_path = AnalysisConfig.paths.output / "model_map.json"

        self.model_map_path = model_map_path
        self._model_map: Dict[str, Dict[str, str]] = self._load_model_map()
        self._cache: Dict[Tuple[str, str], LoadedModelInfo] = {}

    def _load_model_map(self) -> Dict[str, Dict[str, str]]:
        if not self.model_map_path.exists():
            raise FileNotFoundError(f"model_map.json not found at {self.model_map_path}")
        with self.model_map_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        return data

    @staticmethod
    def _load_hyperparams_for_model_file(model_file: Path) -> Dict[str, Any]:
        stem = model_file.stem
        parts = stem.split("_")
        if len(parts) < 2:
            return {}

        h = parts[-1]
        hp_path = model_file.parent / f"hyperparameters_{h}.json"
        if not hp_path.exists():
            return {}

        with hp_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _load_model_for_geo(self, geo_id: str) -> LoadedModelInfo:
        rec = self._model_map.get(geo_id)
        if rec is None:
            raise KeyError(f"GeoID {geo_id!r} not found in model_map.json")

        model_name = rec["model_name"]
        model_file = Path(rec["model_file"])
        key = (model_name, str(model_file))

        cached = self._cache.get(key)
        if cached is not None:
            return cached

        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")

        model: PredictorModel = load(model_file)
        hp = self._load_hyperparams_for_model_file(model_file)
        predictor = OutcomePredictor(model=model)
        info = LoadedModelInfo(model=model, predictor=predictor, hyperparams=hp)
        self._cache[key] = info
        return info

    def get_predictor_for_geo(self, geo_id: str) -> Tuple[OutcomePredictor, Dict[str, Any]]:
        info = self._load_model_for_geo(geo_id)
        return info.predictor, info.hyperparams

    def build_initial_window_inputs(
        self,
        geo_id: str,
        window_size: int,
        end_index: int,
        horizon: int = 1,
    ) -> Tuple[ModelInputs, pd.DataFrame]:
        geo_df = self.data.get_timeseries(geo_id)
        if geo_df.empty:
            raise ValueError(f"No data available for GeoID {geo_id!r}")

        geo_df = geo_df.reset_index(drop=True)
        if end_index < window_size - 1:
            raise ValueError(
                f"end_index {end_index} too small for window_size {window_size}"
            )
        if end_index >= len(geo_df):
            raise ValueError(
                f"end_index {end_index} out of bounds for geo {geo_id} "
                f"(len={len(geo_df)})"
            )

        start = end_index - window_size + 1
        window_df = geo_df.iloc[start : end_index + 1].copy()

        # Use ModelIOPairBuilder internals to encode meta/policies/outcomes consistently
        builder = ModelIOPairBuilder(window_size=window_size, horizon=horizon, max_per_geo=None)

        md = AnalysisConfig.metadata
        date_col = md.date_column

        # meta uses full geo_df
        encoded_meta = builder._encode_meta(geo_df)

        policy_history = builder._encode_policies(window_df, 0, window_size)
        outcome_history = builder._encode_outcomes(window_df, 0, window_size)
        if outcome_history is None:
            raise ValueError("Outcome encoding returned None for initial window.")

        end_date = pd.to_datetime(window_df.iloc[-1][date_col])

        inputs = ModelInputs(
            geo_id=geo_id,
            meta=encoded_meta,
            policy_history=policy_history,
            outcome_history=outcome_history,
            end_date=end_date,
            horizon=horizon,
        )
        return inputs, window_df

    @staticmethod
    def apply_policy_override(
        inputs: ModelInputs,
        new_policy_last_row: np.ndarray,
    ) -> ModelInputs:
        """
        Return a new ModelInputs where the *last* policy row is replaced by new_policy_last_row.
        Does not modify the original inputs.
        """
        if new_policy_last_row.shape[0] != inputs.policy_history.shape[1]:
            raise ValueError(
                f"Policy override dimension mismatch: expected {inputs.policy_history.shape[1]}, "
                f"got {new_policy_last_row.shape[0]}"
            )

        ph = inputs.policy_history.copy()
        ph[-1, :] = new_policy_last_row.astype(float)

        return ModelInputs(
            geo_id=inputs.geo_id,
            meta=inputs.meta,
            policy_history=ph,
            outcome_history=inputs.outcome_history,
            end_date=inputs.end_date,
            horizon=inputs.horizon,
        )
