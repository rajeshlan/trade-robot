# python -m monitoring.dashboard (improvise the dashboard according to your project)
import os
import sys
import logging
from flask import Flask, render_template
from data.fetch_data import fetch_data_from_bybit
from strategies.visualizations import generate_performance_heatmap, plot_moving_averages

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Add the project root directory to the system path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

# Specify the template folder
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')
STATIC_DIR = os.path.join(BASE_DIR, 'static')

# Initialize Flask app
app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)

@app.route("/")
def home():
    """Render the home route with data visualizations."""
    try:
        logging.info("Fetching data from Bybit...")
        df = fetch_data_from_bybit()

        if df is not None and not df.empty:
            logging.info("Data fetched successfully. Generating visualizations...")
            # Generate moving averages plot
            plot_moving_averages(df)

            logging.info("Visualizations generated successfully.")
        else:
            logging.warning("No data available to generate visualizations.")

        # Render dashboard template
        return render_template('dashboard.html', data_available=not df.empty)
    except Exception as e:
        logging.error(f"Error in home route: {e}")
        return render_template('error.html', error_message=str(e)), 500

if __name__ == "__main__":
    os.chdir(BASE_DIR)  # Ensure the project runs from the base directory
    logging.info(f"Starting Flask server... Templates directory: {TEMPLATE_DIR}")
    app.run(debug=True)
