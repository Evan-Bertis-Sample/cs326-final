from analysis.config import AnalysisConfig
from analysis.oxcgrt_data import OxCGRTData, GeoID
from analysis.predict import OutcomePredictor, ModelIOPairBuilder
from analysis.models.persistence import PersistenceModel
from analysis.cache import Cache, CacheConfig
from analysis.io import timed_banner

from typing import Optional

@timed_banner
def print_data_summary():
    all_data = OxCGRTData(AnalysisConfig.paths.data)
    # Print basic info
    print("Overview:")
    print(f"Rows: {len(all_data.data)}, Columns: {len(all_data.data.columns)}")
    print("Date column:", all_data.date_col)
    print()

    # Show first few metadata-defined columns
    print("ID columns:", AnalysisConfig.metadata.id_columns)
    print("Policy columns (sample):", AnalysisConfig.metadata.policy_columns[:5])
    print("Outcome columns:", AnalysisConfig.metadata.outcome_columns)
    print()

    countries = all_data.geo_id_strings(True)
    print(F"Number of geos: {len(countries)}")


# A collection of functions to run with caching
@timed_banner
def build_training_pairs(window : int, horizon : int, max_per_geo : str):
    all_data = OxCGRTData(AnalysisConfig.paths.data)
    # Build pairs (small sample)
    builder = ModelIOPairBuilder(window_size=window, horizon=horizon, max_per_geo=max_per_geo,
                                policy_missing="ffill_then_zero", outcome_missing="ffill_bfill", verbose=True)
    
    train_pairs = builder.get_pairs(all_data)
    return train_pairs

@timed_banner
def train_baseline():
    pass