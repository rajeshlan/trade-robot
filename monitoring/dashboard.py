import sys
import os

# Add the project root directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, render_template
import pandas as pd
from data.fetch_data import fetch_data_from_bybit
from strategies.visualizations import generate_performance_heatmap, plot_moving_averages

app = Flask(__name__)

@app.route("/")
def home():
    # Fetch data and generate required visualizations
    df = fetch_data_from_bybit()
    if not df.empty:
        plot_moving_averages(df)
    
    # Serve Dashboard HTML (you'd have to set this up with Flask templates)
    return render_template('dashboard.html')

if __name__ == "__main__":
    app.run(debug=True)
