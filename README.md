# Heart Disease Detection Using Machine Learning

I built this project as a learning-focused machine-learning application that explores how patient health measurements can be used to estimate the likelihood of heart disease. It includes scripts for preparing data and training models, plus a Flask web app for entering patient information and viewing a prediction.

> **Important:** This project is for educational and demonstration purposes only. It is not a medical device and must not be used to diagnose, treat, or rule out heart disease. Please consult a qualified healthcare professional for medical advice.

## Project structure

```text
Heart-disease-detection-Using-ml/
├── .github/                          # GitHub repository configuration
├── AllModelsTraining.py               # Train and evaluate multiple classifiers
├── Data_Spliting.py                   # Split the Excel data and compare classifiers
├── Heart_disease_cleveland_new.xlsx    # Dataset used by the training scripts
├── app.py                             # Flask application and prediction/database logic
├── index.html                         # Web form template (see Flask note below)
├── logistic_regression_model.pkl      # Saved model artifact
├── scaler.pkl                         # Saved feature scaler
├── style.css                          # Web app stylesheet (see Flask note below)
├── trainmodel.py                      # Train and save the logistic regression model
└── trained_model.pkl                  # Saved model used by the Flask app
```

The Flask app also creates `heart_disease.db` locally to store submitted patient data, predictions, and errors. The database is generated at runtime and is not included in the repository.

## What the project does

- Loads heart-disease data from an Excel workbook.
- Splits data into training and test sets and scales input features.
- Trains a logistic regression model; a separate script also compares logistic regression, random forest, and support vector machine models.
- Runs a Flask web interface that accepts patient measurements and returns a model prediction and probability.
- Records submitted values and prediction results in a local SQLite database.

## Requirements

- Python 3.9 or newer
- pip

Install the Python packages used by the project:

```bash
python -m pip install Flask pandas numpy scikit-learn joblib openpyxl
```

## Setup and run

1. Clone the repository and enter its directory:

   ```bash
   git clone https://github.com/asifm6414/Heart-disease-detection-Using-ml.git
   cd Heart-disease-detection-Using-ml
   ```

2. Install the requirements shown above.

3. **Set up Flask's template and static folders.** Flask looks for the HTML template in `templates/` and static files in `static/`, but these files are currently at the repository root. Create the folders and move the files:

   ```bash
   mkdir -p templates static
   mv index.html templates/index.html
   mv style.css static/style.css
   ```

4. Start the application:

   ```bash
   python app.py
   ```

5. Open the local address printed by Flask (typically `http://127.0.0.1:5000`) in your browser.

The app loads `trained_model.pkl` and `scaler.pkl` when it starts. If you need to regenerate them, run:

```bash
python trainmodel.py
```

The training script reads `Heart_disease_cleveland_new.xlsx` from the current directory and saves updated model artifacts there.

## Training and evaluation scripts

- `trainmodel.py` trains a logistic regression classifier, saves `trained_model.pkl` and `scaler.pkl`, and prints evaluation metrics.
- `Data_Spliting.py` creates train/test Excel files under `Train_data/` and `Test_data/` and compares several classifiers.
- `AllModelsTraining.py` is an additional script for training and evaluating multiple classifiers. Check its expected input file locations before running it; its paths may need to match the output produced by the data-splitting script.

Model scores depend on the dataset, split, and training configuration. I have not included fixed performance claims here; run the scripts to see the metrics produced in your environment.

## Input features

The web form collects age, sex, chest-pain category, resting blood pressure, cholesterol, fasting blood sugar, resting ECG, maximum heart rate, exercise-induced angina, ST depression (`oldpeak`), ST-segment slope, number of major vessels (`ca`), and thalassemia category (`thal`). These values must use the same feature definitions and encoding expected by the dataset and saved model.

## Data and privacy

The app stores submitted patient measurements and prediction records in a local SQLite database. Do not enter real or identifiable health information into this demo. If adapting the project for any real-world use, review privacy, security, clinical validation, and applicable regulatory requirements first.

## License

No license is currently specified for this repository. Please contact the repository owner before reusing or distributing the project.
