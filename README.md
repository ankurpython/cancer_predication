# Cancer Prediction

A machine learning project for predicting cancer outcomes using Python, featuring intuitive web interfaces powered by Streamlit and Flask.

## Overview
This repository provides a robust machine learning model to predict cancer outcomes from medical data. It includes user-friendly web applications built with Streamlit for interactive exploration and Flask for API-driven access.

## Features
- **Data Preprocessing**: Clean and prepare medical datasets for analysis.
- **Machine Learning Model**: Accurate cancer prediction using advanced algorithms.
- **Streamlit Interface**: Interactive dashboard for real-time predictions and visualizations.
- **Flask API**: Programmatic access for integrating predictions into other applications.
- **Model Evaluation**: Comprehensive metrics to assess performance.

## Prerequisites
- Python 3.8+
- Git
- Virtual environment (recommended)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/ankurpython/cancer_predication.git
   cd cancer_predication
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Streamlit App
1. Ensure dependencies are installed.
2. Launch the Streamlit app:
   ```bash
   streamlit run app_streamlit.py
   ```
3. Open `http://localhost:8501` in your browser to explore predictions and visualizations.

### Running the Flask API
1. Ensure dependencies are installed.
2. Start the Flask server:
   ```bash
   python app_flask.py
   ```
3. Access the API at `http://localhost:5000`. Example endpoint:
   ```bash
   curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"data": [value1, value2, ...]}'
   ```

## Project Structure
- `app_streamlit.py`: Streamlit app for interactive UI.
- `app_flask.py`: Flask app for API endpoints.
- `requirements.txt`: List of dependencies.
- `data/`: Directory for input datasets (add your data here).
- `models/`: Directory for trained models.
