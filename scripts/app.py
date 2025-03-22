## run with python -m scripts.app  (fixed static folder issue)
import sys
import os
import matplotlib
import matplotlib.pyplot as plt
from flask import Flask, render_template
import pandas as pd
from data.fetch_data import fetch_data_from_bybit
from strategies.visualizations import plot_moving_averages

# Set Matplotlib to use a non-interactive backend
matplotlib.use('Agg')

# Add the project root directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Initialize Flask app with proper static folder path
app = Flask(__name__, template_folder="../templates", static_folder="../static")

@app.route("/")
def home():
    """
    Fetches data from Bybit and generates the required visualizations.
    Passes a flag to the template to check if data was available.
    """
    df = fetch_data_from_bybit()  # Fetch data from Bybit
    data_available = False  # Initialize flag

    if not df.empty:  # Check if DataFrame has data
        # Save the image in the correct static folder
        static_file_path = os.path.join(app.static_folder, 'moving_averages.png')
        plot_moving_averages(df, save_path=static_file_path)
        data_available = True  # Set flag to True if data is available

    # Pass the flag to the template to conditionally render content
    return render_template('dashboard.html', data_available=data_available)

if __name__ == "__main__":
    app.run(debug=True)
