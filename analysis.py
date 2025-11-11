from analysis.config import AnalysisConfig
from analysis.oxcgrt_data import OxCGRTData, GeoID


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

    # Test geo IDs
    geo_series = all_data.geo_id_strings(True)
    print("GeoID Sample")
    print(geo_series)
    print(f"Unique GeoIDs: {len(geo_series)}")
    print()

    # Test region split
    print("Region split:")
    splits = all_data.split_by_region(ratio=(0.8, 0.1, 0.1))
    print(f"Train size: {len(splits.training)}, Test size: {len(splits.testing)}, Val size: {len(splits.validation) if splits.validation is not None else 0}")
    print()

    # Example: get one region's time series
    first_geo = geo_series.dropna().unique()[0]
    print(f"Timeseries for {first_geo}")
    ts = all_data.get_timeseries(first_geo)
    print(ts.head())


if __name__ == "__main__":
    main()
