import datetime
import os
import pandas as pd
import sqlite3
import numpy as np
import joblib
from flask import Flask, render_template, request

# Load the trainedd model
try:
    model = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trained_model.pkl'))
    scaler = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scaler.pkl'))
except FileNotFoundError as e:
    print(f"Error loading model or scaler: {e}")
    exit(1)
# initialization of Flassk app
app = Flask(__name__)


# Function to connect to the SQLite database
def get_db_connection():
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'heart_disease.db')
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # For accessing columns by name
    return conn


# Function to initialize the database
def init_db():
    with get_db_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS Patients(
        patient_id INTEGER PRIMARY KEY AUTOINCREMENT,
        age INTEGER NOT NULL,
        sex INTEGER NOT NULL,
        cp INTEGER NOT NULL,
        trestbps INTEGER NOT NULL,
        chol INTEGER NOT NULL,
        fbs INTEGER NOT NULL,
        restecg INTEGER NOT NULL,
        thalach INTEGER NOT NULL,
        exang INTEGER NOT NULL,
        oldpeak REAL NOT NULL,
        slope INTEGER NOT NULL,
        ca INTEGER NOT NULL,
        thal INTEGER NOT NULL
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS Predictions(
        prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id INTEGER NOT NULL,
        prediction_result TEXT NOT NULL,
        confidence_score REAL NOT NULL,
        timestamp DATETIME NOT NULL,
        FOREIGN KEY (patient_id) REFERENCES Patients(patient_id)

        )
        """)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS Errors(
        error_id INTEGER PRIMARY KEY AUTOINCREMENT,
        error_message TEXT NOT NULL,
        timestamp DATETIME NOT NULL
        )
        """)
        conn.commit()


# call the initialization function when the app starts
init_db()


# Home page route
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/prediction', methods=['POST'])
def predict():
    try:
        # data collection form
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        cp = int(request.form['cp'])
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = int(request.form['fbs'])
        restecg = int(request.form.get('restecg', 0))
        thalach = int(request.form['thalach'])
        exang = int(request.form.get('exang', 0))
        oldpeak = float(request.form['oldpeak'])
        slope = int(request.form.get('slope', 0))
        ca = int(request.form.get('ca', 0))
        thal = int(request.form.get('thal', 0))

        # For DataFrame and preprocess
        input_data = pd.DataFrame({
            'age': [age],
            'sex': [sex],
            'cp': [cp],
            'trestbps': [trestbps],
            'chol': [chol],
            'fbs': [fbs],
            'restecg': [restecg],
            'thalach': [thalach],
            'exang': [exang],
            'oldpeak': [oldpeak],
            'slope': [slope],
            'ca': [ca],
            'thal': [thal]
        })

        # Validate features
        if hasattr(model, 'feature_name_in_'):
            missing = set(model.feature_names_in_) - set(input_data.columns)
            if missing:
                raise ValueError(f"Missing required features: {missing}")

        # input data scalling
        input_data_scaled = scaler.transform(input_data)

        # prediction
        prediction = model.predict(input_data_scaled)[0]
        probability = model.predict_proba(input_data_scaled)[0][1] * 100

        # load data in database
        with get_db_connection() as conn:
            cursor = conn.cursor()

            # insert patient data
            cursor.execute("""
            INSERT INTO Patients(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal))
            patient_id = cursor.lastrowid

            # insert prediction result
            cursor.execute("""
            INSERT INTO Predictions (patient_id, prediction_result, confidence_score, timestamp )
            VALUES(?, ?, ?, ?)

            """, (patient_id, str(prediction), float(probability), datetime.datetime.now()))
            conn.commit()

            if prediction == 1:
                result = "The patient is have heart disease."
            else:
                result = "The patient is not have heart disease"
            return render_template('index.html', result=result, probability=f"{probability:.2f}%")
    except Exception as e:
        # log error to the database
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO Errors(error_message, timestamp)
            VALUES(?, ?)
            """, (str(e), datetime.datetime.now()))
            conn.commit()
        print(f"Error: {str(e)}")
        return render_template('index.html', result="Error in prediction. Kindly check your inputs.", probability="N/A")


if __name__ == '__main__':
    app.run(debug=True)

