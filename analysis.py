import pandas as pd

def load_data(file_path):
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith(('.xls', '.xlsx')):
        return pd.read_excel(file_path)
    else:
        raise ValueError("File format not supported. Use CSV or Excel.")

def analyze_data(data):
    """Analyze data properties and return the analysis results."""
    analysis_result = {}
    analysis_result['shape'] = data.shape
    analysis_result['columns'] = data.columns.tolist()
    analysis_result['missing_values'] = data.isnull().sum().to_dict()

    descriptive_statistics = data.describe(include='all')
    processed_stats = []
    for stat_name in descriptive_statistics.index:  # Use index to get statistic names
        row = {'statistic': stat_name}
        for col in descriptive_statistics.columns:
            row[col] = descriptive_statistics.at[stat_name, col]  # Use .at to access specific values
        processed_stats.append(row)

    analysis_result['descriptive_statistics'] = processed_stats

    return analysis_result
