from analysis.config import AnalysisConfig
from analysis.oxcgrt_data import OxCGRTData, GeoID

from analysis.predict import OutcomePredictor, ModelIOPairBuilder
from analysis.models.persistence_model import PersistenceModel
from analysis.cache import Cache, CacheConfig

from pathlib import Path

def build_pairs(window : int, horizon : int, max_per_geo : str):
    all_data = OxCGRTData(AnalysisConfig.paths.data)
    # Build pairs (small sample)
    builder = ModelIOPairBuilder(window_size=window, horizon=horizon, max_per_geo=max_per_geo,
                                policy_missing="ffill_then_zero", outcome_missing="ffill_bfill", verbose=True)
    
    train_pairs = builder.get_pairs(all_data)
    return train_pairs

def main():
    # Load dataset
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

    Cache.init(CacheConfig(root=Path("models/.cache"), compress=3, default_verbose=True))

    Cache.Begin("build_pairs_all", ["window", "horizon", "max_per_geo"])
    train_pairs = Cache.call(build_pairs, window=14, horizon=1, max_per_geo=None)
    have_same = Cache.exists(build_pairs, window=14, horizon=1, cluster="ALL")

    print(train_pairs, have_same)
    Cache.End()

    # Train and evaluate persistence
    model = PersistenceModel()
    model.fit_batch(train_pairs)

    predictor = OutcomePredictor(model)
    metrics = predictor.evaluate(train_pairs[:2])  # quick sanity slice
    print(metrics)


if __name__ == "__main__":
    main()
