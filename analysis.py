from analysis.config import AnalysisConfig
from analysis.oxcgrt_data import OxCGRTData, GeoID

from analysis.predict import OutcomePredictor, ModelIOPairBuilder
from analysis.models.persistence_model import PersistenceModel

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
    
    # Build pairs (small sample)
    builder = ModelIOPairBuilder(window_size=14, horizon=1, max_per_geo=20,
                                policy_missing="ffill_then_zero", outcome_missing="ffill_bfill", verbose=True)
    
    train_pairs = builder.get_pairs(all_data)

    # Train and evaluate persistence
    model = PersistenceModel()
    model.fit_batch(train_pairs)

    predictor = OutcomePredictor(model)
    metrics = predictor.evaluate(train_pairs[:2])  # quick sanity slice
    print(metrics)


if __name__ == "__main__":
    main()
