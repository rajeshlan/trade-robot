## run with python -m scripts.app  (need checking) as gives quite some errors on http://127.0.0.1:5000/

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

app = Flask(__name__)

@app.route("/")
def home():
    # Fetch data and generate required visualizations
    df = fetch_data_from_bybit()
    if not df.empty:
        # Plot moving averages (now saved to a file instead of trying to display interactively)
        plot_moving_averages(df)

    # Serve Dashboard HTML (you'd have to set this up with Flask templates)
    return render_template('dashboard.html')

if __name__ == "__main__":
    app.run(debug=True)
