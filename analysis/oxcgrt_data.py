from analysis.config import *

import pandas as pd
import typing as tp
from pathlib import Path

class OxCGRTData:
    """
    An interface to work with the Oxford COVID-19 Government Response Tracker (OxCGRT) data.
    """

    def __init__(self, data_path: tp.Union[str, Path]):
        self.data = self._load_data(data_path)

    def __init__(self, frame : pd.DataFrame):
        self.data = frame

    def _load_data(self, data_path: tp.Union[str, Path]) -> pd.DataFrame:
        return pd.read_csv(data_path)
    
    @staticmethod
    def get_column_names() -> tp.Set[str]:
        return set(AnalysisConfig.metadata.id_columns +
                   AnalysisConfig.metadata.policy_columns + 
                   AnalysisConfig.metadata.outcome_columns)

    @classmethod
    def get_outcome_columns(cls, data: pd.DataFrame) -> tp.Set[str]:
        return set(AnalysisConfig.metadata.outcome_columns)
    
