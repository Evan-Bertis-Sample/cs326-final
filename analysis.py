from analysis.config import AnalysisConfig
from analysis.oxcgrt_data import OxCGRTData

def main():
    all_data = OxCGRTData(AnalysisConfig.paths.data)
    
    print(all_data.get_column_names())


if __name__ == "__main__":
    main()