import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
import joblib
import os
import numpy as np

# Directories for loading data and saving models
csv_directory = r"C:\Users\dell\Downloads\GraphXpert\data\csv"
xls_directory = r"C:\Users\dell\Downloads\GraphXpert\data\xls"
model_directory = r"C:\Users\dell\Downloads\GraphXpert\models"

# Specify the directories for different plot data
directories = {
    'scatter': os.path.join(csv_directory, 'scatter_data'),
    'line': os.path.join(csv_directory, 'line_data'),
    'histogram': os.path.join(csv_directory, 'histogram_data'),
    'boxplot': os.path.join(csv_directory, 'boxplot_data'),
    'bar': os.path.join(csv_directory, 'bar_data'),
    'pie': os.path.join(csv_directory, 'pie_data')
}

def load_data(file_path):
    """Load the synthetic data from a CSV or XLS file."""
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        return pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")

def preprocess_data(df, target_column):
    """Prepare the data for model training."""
    print(f"Initial Data Shape: {df.shape}")
    print(f"Data Loaded:\n{df.head()}")  # Debug print

    # Check if the target column exists
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in the dataset.")

    # Select only numerical columns
    df_numerical = df.select_dtypes(include=[np.number])
    print(f"Data after selecting numerical columns:\n{df_numerical.head()}")  # Debug print

    # Handle categorical columns
    df_categorical = df.select_dtypes(include=['object'])
    if not df_categorical.empty:
        # One-Hot Encoding for categorical columns
        encoder = OneHotEncoder()
        encoded_features = encoder.fit_transform(df_categorical).toarray()
        df_encoded = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(df_categorical.columns))
        df_numerical = pd.concat([df_numerical, df_encoded], axis=1)

    # Prepare features (X) and target (y)
    X = df_numerical.drop(columns=[target_column], errors='ignore')  # Independent variables
    y = df_numerical[target_column]  # Dependent variable

    print(f"Features (X) shape: {X.shape}, Target (y) shape: {y.shape}")  # Debug print

    if X.empty or y.empty:
        raise ValueError("Features or target variable is empty.")

    return X, y

def train_model(X, y):
    """Train a Multilinear Regression model."""
    # Ensure we have enough samples to train and test
    if len(y) < 2:
        print("Not enough data to train the model.")
        return None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Ensure there are enough samples for testing
    if len(X_test) < 2:
        print("Not enough test data to evaluate the model.")
        return None
    
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Calculate accuracy metrics
    accuracy_metrics(y_test, y_pred)

    return model

def accuracy_metrics(y_true, y_pred):
    """Calculate and print the accuracy metrics for predictions."""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R-squared: {r2:.4f}")

def save_model(model, filename):
    """Save the trained model to the specified directory."""
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)

    model_path = os.path.join(model_directory, filename)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

def main():
    # Loop through each plot type
    for plot_type in directories.keys():
        # Determine file path and target column based on plot type
        if plot_type == 'scatter':
            file_path = os.path.join(directories['scatter'], "scatter_data_1.csv")
            target_column = "Y Values"
        elif plot_type == 'line':
            file_path = os.path.join(directories['line'], "line_data_1.csv")
            target_column = "Stock Prices"
        elif plot_type == 'histogram':
            file_path = os.path.join(directories['histogram'], "histogram_data_1.csv")
            target_column = "Histogram Data"
        elif plot_type == 'boxplot':
            file_path = os.path.join(directories['boxplot'], "boxplot_data_1.csv")
            target_column = "Box Plot Data"
        elif plot_type == 'bar':
            file_path = os.path.join(directories['bar'], "bar_data_1.csv")
            target_column = "Bar Values"
        elif plot_type == 'pie':
            # Pie chart does not have a meaningful prediction target
            print("Skipping model training for pie chart data.")
            continue
        else:
            print(f"Warning: Invalid plot type specified: {plot_type}. Skipping this plot type.")
            continue

        # Load the data
        try:
            df = load_data(file_path)
            print(f"Loaded Data for {plot_type}:\n{df.head()}")  # Debug print

            # Preprocess the data
            X, y = preprocess_data(df, target_column)

            # Train the model
            model = train_model(X, y)

            # Save the model if training was successful
            if model is not None:
                save_model(model, filename=f'trained_model_{plot_type}.pkl')

        except Exception as e:
            print(f"Error processing {plot_type}: {e}")

if __name__ == "__main__":
    main()
