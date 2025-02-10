import os
from flask import Flask, render_template, request, jsonify
import pandas as pd
import subprocess
import RPi.GPIO as GPIO
import time
import sys
from evaluate import ARIMAForecaster


# GPIO pin for the LED
LED1_PIN = 17
# Setup
GPIO.setmode(GPIO.BCM)  # Use Broadcom pin numbering
GPIO.setup(LED1_PIN, GPIO.OUT)  # Set the pin as an output


# Correct path to the templates folder
app = Flask(__name__,template_folder='/home/pi4/Desktop/server_5/template')

# Path to the other Python script
script_path = 'Desktop/server_5/update.py'
# Run the script in parallel (asynchronously)
try:
    print(f"Running {script_path} in parallel...")
    subprocess.Popen(['python3', script_path])
except FileNotFoundError:
    print(f"The script {script_path} was not found.")

# get boolean value
@app.route('/send_boolean', methods=['POST'])
def handle_boolean():
    data = request.get_json()  # Retrieve the JSON data
    boolean_value = data.get('value')  # Extract the boolean value
    if boolean_value == True:
        print("Boolean is True!")
        GPIO.output(LED1_PIN, GPIO.HIGH)
    else:
        print("Boolean is False!")
        GPIO.output(LED1_PIN, GPIO.LOW)

    return jsonify(message="Boolean received!")  # Return a JSON response


@app.route('/')
def index():
    import datetime
    sys.path.append('/home/pi4/Desktop/server_5')
    from evaluate import ARIMAForecaster

    data_path = "/home/pi4/Desktop/server_5/data/bangkok-air-quality_raw.csv"

    # Create an ARIMAForecaster instance
    arima_forecaster = ARIMAForecaster(data_path)

    # Run the forecast to get the DataFrame
    forecast_df = arima_forecaster.run_forecast()

    # Get the original data from preprocessing
    original_data = arima_forecaster.preprocess_data()

    # Filter original data to include only the last year
    current_year = datetime.datetime.now().year
    last_year = current_year - 1
    original_data_last_year = original_data[original_data['Year'] == last_year]

    # Prepare chart data for both original and forecasted data
    chart_data = {
        "original": {
            "labels": original_data_last_year['date'].dt.strftime('%Y-%m-%d').tolist(),
            "values": original_data_last_year['bangkok_aqi'].tolist(),
        },
        "forecasted": {
            "labels": forecast_df['date'].dt.strftime('%Y-%m-%d').tolist(),
            "values": forecast_df['forecasted_aqi'].tolist(),
        },
    }
    return render_template('index.html', chart_data=chart_data)



if __name__ == '__main__':
    app.run(debug=True)     

    