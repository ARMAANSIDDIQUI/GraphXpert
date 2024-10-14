import os
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, render_template, request, redirect, url_for, flash, send_file
from analysis import load_data, analyze_data  # Import analysis functions
from recommend import analyze_data as recommend_analysis  # Import recommendation functions
import joblib
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'  # Use relative path for file uploads
app.config['MAX_CONTENT_LENGTH'] = 128 * 1024 * 1024  # 128MB limit for file uploads
app.secret_key = 'supersecretkey'  # Required for flash messaging

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    allowed_extensions = {'csv', 'xls', 'xlsx'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def create_plot(data, plot_type, predictions=None):
    """Create a plot based on the selected plot type."""
    plt.figure(figsize=(10, 6))
    print(f"Creating plot type: {plot_type}")

    # Create the plot
    if plot_type == 'scatter':
        data.plot(kind='scatter', x=data.columns[0], y=data.columns[1], alpha=0.5, color='blue')
    elif plot_type == 'line':
        data.plot(kind='line', x=data.columns[0], y=data.columns[1], color='blue')
    elif plot_type == 'bar':
        data[data.columns[0]].value_counts().plot(kind='bar', color='blue')
    elif plot_type == 'hist':
        data[data.columns[0]].hist(color='blue')
    elif plot_type == 'box':
        data.boxplot(column=data.columns[0], by=data.columns[1])
    elif plot_type == 'pie':
        data.set_index(data.columns[0]).plot.pie(y=data.columns[1], autopct='%1.1f%%', figsize=(10, 6))

    # Plot predictions if they exist
    if predictions is not None:
        if plot_type == 'line':
            plt.plot(predictions.index, predictions.values, color='red', label='Predictions')
        elif plot_type == 'scatter':
            plt.scatter(predictions.index, predictions.values, color='red', label='Predictions')
        plt.legend()

    plt.title(f'{plot_type.capitalize()} Plot')
    plt.xlabel(data.columns[0])
    plt.ylabel(data.columns[1])

    plot_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'plot.png')
    plt.savefig(plot_filename)
    plt.close()

    return plot_filename

@app.route('/')
def index():
    """Render the homepage."""
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_form():
    """Handle file upload."""
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part in the request', 'danger')
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)

        if not allowed_file(file.filename):
            flash('Invalid file type. Please upload a CSV or XLS file.', 'danger')
            return redirect(request.url)

        if file:
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Redirect to file_name.html after successful upload
            return render_template('file_name.html', filename=filename)

    return render_template('upload.html')

@app.route('/start_analysis', methods=['POST'])
def start_analysis():
    """Analyze the uploaded file and provide recommendations."""
    filename = request.form['filename']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        # Load the data for analysis and recommendations
        df = load_data(file_path)

        # Perform data analysis
        analysis_result = analyze_data(df)

        # Get graph recommendations from recommend.py
        recommendations, explanations = recommend_analysis(df)

        # Pass both analysis, recommendations, and zip to the template
        return render_template(
            'analysis.html',
            analysis_result=analysis_result,
            recommendations=recommendations,
            explanations=explanations,
            filename=filename,
            zip=zip  # Pass Python's zip function to the template
        )
    except Exception as e:
        flash(f"Error processing the file: {str(e)}", "danger")
        return redirect(url_for('upload_form'))

@app.route('/save_analysis', methods=['POST'])
def save_analysis():
    """Save the analysis and recommendations as a text file and offer it for download."""
    filename = request.form['filename']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        # Load the data for analysis and recommendations
        df = load_data(file_path)

        # Perform data analysis
        analysis_result = analyze_data(df)

        # Get graph recommendations from recommend.py
        recommendations, explanations = recommend_analysis(df)

        # Prepare the analysis and recommendation data
        content = []
        content.append(f"File analyzed: {filename}\n")
        content.append("Data Properties:\n")
        content.append(f"Shape: {analysis_result['shape']}\n")
        content.append(f"Columns: {analysis_result['columns']}\n")
        content.append("\nMissing Values:\n")
        for col, count in analysis_result['missing_values'].items():
            content.append(f"{col}: {count} missing values\n")
        content.append("\nDescriptive Statistics:\n")
        for row in analysis_result['descriptive_statistics']:
            content.append(f"{row['statistic']}: {row}\n")

        content.append("\nRecommended Graph Types:\n")
        for recommendation, explanation in zip(recommendations, explanations):
            content.append(f"{recommendation}: {explanation}\n")

        # Write the content to a text file
        save_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'analysis_recommendations.txt')
        with open(save_filename, 'w') as f:
            f.write('\n'.join(content))

        # Send the file to the user
        return send_file(save_filename, as_attachment=True)

    except Exception as e:
        flash(f"Error saving analysis: {str(e)}", "danger")
        return redirect(url_for('start_analysis'))

@app.route('/plot', methods=['POST'])
def plot():
    """Plot the graph based on the user's selection."""
    filename = request.form['filename']
    plot_type = request.form['plot_type']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        df = load_data(file_path)

        # Create the plot based on the selected type
        plot_filename = create_plot(df, plot_type)

        with open(plot_filename, "rb") as image_file:
            plot_img = base64.b64encode(image_file.read()).decode('utf-8')

        # Pass the filename to the template for downloading
        return render_template('plot_view.html', plot_img=plot_img, plot_filename='plot.png', filename=filename)

    except Exception as e:
        flash(f"Error creating plot: {str(e)}", "danger")
        return redirect(url_for('start_analysis'))

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    """Download the generated plot file."""
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        flash("File not found!", "danger")
        return redirect(url_for('upload_form'))

@app.route('/start_prediction', methods=['POST'])
def start_prediction():
    """Start the prediction process based on user input."""
    filename = request.form['filename']
    plot_type = request.form["plot_type"]

    # Check if the plot type supports predictions
    if plot_type in ['pie', 'hist']:
        flash("Prediction is not applicable for pie or histogram plots.", "danger")
        return redirect(url_for('start_analysis'))

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    model_path = os.path.join('models', f'trained_model_{plot_type}.pkl')
    
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        flash(f"No model found for plot type: {plot_type}", "danger")
        return redirect(url_for('start_analysis'))

    # Load data for predictions
    df = load_data(file_path)

    # Get the number of predictions to make from the user
    n_predictions = int(request.form['n_predictions'])

    # Make predictions
    last_row = df.iloc[-1]
    predictions = []
    for i in range(n_predictions):
        new_value = model.predict(last_row.values.reshape(1, -1))
        predictions.append(new_value[0])
        last_row = np.append(last_row[:-1], new_value)  # Update last row for next prediction

    # Create a DataFrame for predictions
    prediction_index = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=n_predictions)
    prediction_df = pd.DataFrame(predictions, index=prediction_index, columns=['Predictions'])

    # Plot the data with predictions
    plot_filename = create_plot(df, plot_type, prediction_df)

    with open(plot_filename, "rb") as image_file:
        plot_img = base64.b64encode(image_file.read()).decode('utf-8')

    return render_template('prediction_view.html', plot_img=plot_img, predictions=prediction_df)

if __name__ == "__main__":
    app.run(debug=True)
