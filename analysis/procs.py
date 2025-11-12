from analysis.config import AnalysisConfig
from analysis.oxcgrt_data import OxCGRTData, GeoID
from analysis.predict import OutcomePredictor, ModelIOPairBuilder, PredictorModel, ModelTrainingPairSet, ModelPerformanceMetrics
from analysis.cache import Cache, CacheConfig
from analysis.io import timed_banner

from pathlib import Path
from typing import Optional, List, Union, Tuple, Dict, Any

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
def build_training_pairs(cluster_file : Union[str, Path], window : int, horizon : int, max_per_geo : str) -> ModelTrainingPairSet:
    all_data = OxCGRTData(AnalysisConfig.paths.data)

    # cluster the data into regions, then split into train, test, validation datasets
    # implement
    training_data =
    testing_data = 
    validation_data = 

    # Build pairs (small sample)
    builder = ModelIOPairBuilder(window_size=window, horizon=horizon, max_per_geo=max_per_geo,
                                policy_missing="ffill_then_zero", outcome_missing="ffill_bfill", verbose=True)
    
    # generate 
    train_pairs = builder.get_pairs(training_data)
    test_pairs = builder.get_pairs(testing_data)
    validation_pairs = builder.get_pairs(validation_pairs)

    return ModelTrainingPairSet(
        training=train_pairs,
        testing=test_pairs,
        validation=test_pairs
    )

def _get_cluster_from_path(cluster_file : Union[str, Path]) -> List[GeoID]:
    # Deserialize the list of GeoIDs
    # This is simple, it is a list geoids in English, each sperated by a new line
    pass

@timed_banner
def handle_models(cluster_file : Union[str, Path]):
    # wrapper function

    # ideally this should be in the args, but it might break the caching
    models_to_train : List[PredictorModel] = [

    ]

    # train each model
    # then graph performance of the best model

def _search_model_hyperspace(training_pair_set : ModelTrainingPairSet, model : PredictorModel) -> Tuple[Dict[str, Any], ModelPerformanceMetrics]:
    # get training data, it should be in a cache

    # do a search of the hyperparameter space

    # returns the best model's performance metrics and parameters

    pass

def _train_model(training_pair_set : ModelTrainingPairSet, model : PredictorModel):
    pass

def _evaluate_model(training_pair_set : ModelTrainingPairSet, model : PredictorModel) -> ModelPerformanceMetrics:
    pass

def _graph_model_perfomance(model_names, model_params, model_metrics):
    pass