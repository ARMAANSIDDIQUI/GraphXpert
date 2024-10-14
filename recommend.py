import pandas as pd

def analyze_data(df):
    """Analyze the data and recommend graph types based on column types."""
    column_info = df.dtypes

    recommendations = []
    explanations = []
    categorical_columns = []
    numerical_columns = []
    datetime_columns = []

    # Analyze columns by type
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            numerical_columns.append(column)
        elif pd.api.types.is_categorical_dtype(df[column]) or df[column].nunique() < 10:
            categorical_columns.append(column)
        elif pd.api.types.is_datetime64_any_dtype(df[column]):
            datetime_columns.append(column)

    # Check for missing data
    missing_data = df.isnull().sum()
    if missing_data.any():
        recommendations.append("Handle missing data by either filling or dropping missing values.")
        explanations.append("Some columns have missing values. You can either fill missing data using mean, median, or mode, or drop rows with missing data.")

    # Handle numerical columns
    if numerical_columns:
        if len(numerical_columns) == 1:
            recommendations.append(f"Histogram for '{numerical_columns[0]}' to show its distribution.")
            explanations.append(f"A histogram is recommended for numerical data like '{numerical_columns[0]}' to display the distribution of data values over a continuous range.")
            
            recommendations.append(f"Box plot for '{numerical_columns[0]}' to identify outliers.")
            explanations.append(f"A box plot is recommended for '{numerical_columns[0]}' to show the distribution of data and detect any potential outliers.")

        elif len(numerical_columns) >= 2:
            recommendations.append(f"Scatter plot to show the relationship between numerical columns: {', '.join(numerical_columns)}.")
            explanations.append("A scatter plot is useful to visualize potential correlations or relationships between two or more numerical variables.")

    # Handle categorical columns
    if categorical_columns:
        recommendations.append(f"Bar chart for categorical columns like '{', '.join(categorical_columns)}'.")
        explanations.append(f"Bar charts are recommended for categorical data, such as '{categorical_columns[0]}', to show the frequency or count of categories.")
        
        if numerical_columns:
            recommendations.append(f"Box plot of '{numerical_columns[0]}' grouped by '{categorical_columns[0]}'.")
            explanations.append(f"Box plots can be used to compare the distribution of numerical data ('{numerical_columns[0]}') across different categories in '{categorical_columns[0]}'.")
        
        if len(categorical_columns) == 1 and df[categorical_columns[0]].nunique() <= 5:
            recommendations.append(f"Pie chart for '{categorical_columns[0]}' to show category proportions.")
            explanations.append(f"A pie chart is recommended for '{categorical_columns[0]}' when there are few categories (less than 5) to display proportions effectively.")

    # Handle datetime columns
    if datetime_columns:
        if numerical_columns:
            recommendations.append(f"Line plot for time series analysis of '{numerical_columns[0]}' over '{datetime_columns[0]}'.")
            explanations.append(f"A line plot is recommended for time series analysis to show how '{numerical_columns[0]}' changes over time ('{datetime_columns[0]}').")

    return recommendations, explanations
