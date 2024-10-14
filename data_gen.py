import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

# Directories for saving generated data
csv_directory = r"C:\Users\dell\Downloads\GraphXpert\data\csv"
xls_directory = r"C:\Users\dell\Downloads\GraphXpert\data\xls"

# Ensure the directories exist
os.makedirs(os.path.join(csv_directory, 'scatter_data'), exist_ok=True)
os.makedirs(os.path.join(csv_directory, 'line_data'), exist_ok=True)
os.makedirs(os.path.join(csv_directory, 'histogram_data'), exist_ok=True)
os.makedirs(os.path.join(csv_directory, 'boxplot_data'), exist_ok=True)
os.makedirs(os.path.join(csv_directory, 'bar_data'), exist_ok=True)
os.makedirs(os.path.join(csv_directory, 'pie_data'), exist_ok=True)

# Ensure XLS directories exist
os.makedirs(os.path.join(xls_directory, 'scatter_data'), exist_ok=True)
os.makedirs(os.path.join(xls_directory, 'line_data'), exist_ok=True)
os.makedirs(os.path.join(xls_directory, 'histogram_data'), exist_ok=True)
os.makedirs(os.path.join(xls_directory, 'boxplot_data'), exist_ok=True)
os.makedirs(os.path.join(xls_directory, 'bar_data'), exist_ok=True)
os.makedirs(os.path.join(xls_directory, 'pie_data'), exist_ok=True)

def generate_synthetic_scatter_data(n_rows=100):
    """Generate data for scatter plot: two numerical columns."""
    np.random.seed(42)
    x_values = np.random.uniform(1, 100, size=n_rows)
    y_values = x_values * np.random.uniform(0.5, 1.5, size=n_rows)
    
    df = pd.DataFrame({
        'X Values': x_values,
        'Y Values': y_values
    })
    return df

def generate_synthetic_line_data(n_rows=100):
    """Generate data for line plot with date-based stock or sales values."""
    np.random.seed(42)
    start_date = datetime(2010, 1, 1)
    date_column = [start_date + timedelta(days=i) for i in range(n_rows)]
    stock_prices = np.cumsum(np.random.uniform(-10, 10, size=n_rows)) + 100  # Synthetic stock price changes
    
    df = pd.DataFrame({
        'Date': date_column,
        'Stock Prices': stock_prices
    })
    return df

def generate_synthetic_histogram_data(n_rows=100):
    """Generate data for histogram plot: a single numerical column."""
    np.random.seed(42)
    hist_data = np.random.normal(loc=50, scale=15, size=n_rows)
    
    df = pd.DataFrame({
        'Histogram Data': hist_data
    })
    return df

def generate_synthetic_boxplot_data(n_rows=100):
    """Generate data for box plot: one numerical and one categorical column."""
    np.random.seed(42)
    box_data = np.random.normal(50, 10, size=n_rows)
    categories = np.random.choice(['A', 'B', 'C'], size=n_rows)
    
    df = pd.DataFrame({
        'Box Plot Data': box_data,
        'Categories': categories
    })
    return df

def generate_synthetic_bar_data(n_rows=100):
    """Generate data for bar plot: one categorical and one numerical column."""
    np.random.seed(42)
    categories = np.random.choice(['Category 1', 'Category 2', 'Category 3'], size=n_rows)
    values = np.random.randint(10, 100, size=n_rows)
    
    df = pd.DataFrame({
        'Bar Categories': categories,
        'Bar Values': values
    })
    return df

def generate_synthetic_pie_data():
    """Generate data for pie chart: categorical data with corresponding values."""
    categories = ['Slice A', 'Slice B', 'Slice C', 'Slice D']
    values = np.random.randint(50, 150, size=4)
    
    df = pd.DataFrame({
        'Pie Categories': categories,
        'Pie Values': values
    })
    return df

def save_data_to_files(num_files):
    """Generate and save data for all plot types into separate files."""
    
    for i in range(num_files):
        # Generate and save scatter data
        scatter_df = generate_synthetic_scatter_data(n_rows=100)
        scatter_df.to_csv(os.path.join(csv_directory, 'scatter_data', f'scatter_data_{i+1}.csv'), index=False)
        scatter_df.to_excel(os.path.join(xls_directory, 'scatter_data', f'scatter_data_{i+1}.xlsx'), index=False)
        
        # Generate and save line data (with date)
        line_df = generate_synthetic_line_data(n_rows=100)
        line_df.to_csv(os.path.join(csv_directory, 'line_data', f'line_data_{i+1}.csv'), index=False)
        line_df.to_excel(os.path.join(xls_directory, 'line_data', f'line_data_{i+1}.xlsx'), index=False)
        
        # Generate and save histogram data
        hist_df = generate_synthetic_histogram_data(n_rows=100)
        hist_df.to_csv(os.path.join(csv_directory, 'histogram_data', f'histogram_data_{i+1}.csv'), index=False)
        hist_df.to_excel(os.path.join(xls_directory, 'histogram_data', f'histogram_data_{i+1}.xlsx'), index=False)
        
        # Generate and save box plot data
        box_df = generate_synthetic_boxplot_data(n_rows=100)
        box_df.to_csv(os.path.join(csv_directory, 'boxplot_data', f'boxplot_data_{i+1}.csv'), index=False)
        box_df.to_excel(os.path.join(xls_directory, 'boxplot_data', f'boxplot_data_{i+1}.xlsx'), index=False)
        
        # Generate and save bar plot data
        bar_df = generate_synthetic_bar_data(n_rows=100)
        bar_df.to_csv(os.path.join(csv_directory, 'bar_data', f'bar_data_{i+1}.csv'), index=False)
        bar_df.to_excel(os.path.join(xls_directory, 'bar_data', f'bar_data_{i+1}.xlsx'), index=False)
        
        # Generate and save pie chart data
        pie_df = generate_synthetic_pie_data()
        pie_df.to_csv(os.path.join(csv_directory, 'pie_data', f'pie_data_{i+1}.csv'), index=False)
        pie_df.to_excel(os.path.join(xls_directory, 'pie_data', f'pie_data_{i+1}.xlsx'), index=False)

if __name__ == "__main__":
    # Ask the user for the number of files to create
    num_files = int(input("Enter the number of synthetic data files to create for each plot type: "))
    
    # Generate and save all plot type data to separate files
    save_data_to_files(num_files)
