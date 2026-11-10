# Heart Disease Detection — CardioLens Demo

I built this repository as an educational machine-learning project. It includes a Flask web interface that applies a saved scaler and classifier to sample heart-health measurements.

> **Safety notice:** This is a learning demo, not a medical device or clinical decision-support system. It has not been validated for diagnosis, screening, treatment, or patient care. Its output and probability must not guide medical decisions. Do not enter real or identifiable health information. Consult a qualified healthcare professional for health concerns; contact local emergency services for emergencies.

## What is included

- Flask app with server-side input validation, CSRF protection, security response headers, and readiness/liveness endpoints.
- Logistic regression model and scaler artifacts expected at `trained_model.pkl` and `scaler.pkl`.
- Responsive web form served from Flask's `templates/` and `static/` directories.
- Training and model exploration scripts for the included Excel dataset.
- Docker and Render configuration for an example web deployment.

The web app does **not** persist submitted values or predictions. Application logs should also never be configured to include request bodies or health form values.

## Repository structure

```text
.
├── app.py
├── templates/index.html
├── static/style.css
├── requirements.txt
├── Dockerfile
├── render.yaml
├── trainmodel.py
├── Data_Spliting.py
├── AllModelsTraining.py
├── Heart_disease_cleveland_new.xlsx
├── trained_model.pkl
└── scaler.pkl
```

Root-level `index.html` and `style.css` are legacy copies from the original project; the running Flask app uses `templates/index.html` and `static/style.css`.

## Run locally

Requires Python 3.11 and pip.

```bash
git clone https://github.com/asifm6414/Heart-disease-detection-Using-ml.git
cd Heart-disease-detection-Using-ml
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Set a local secret and run with Flask's development server:

```bash
# macOS/Linux
export SECRET_KEY="$(python -c 'import secrets; print(secrets.token_hex(32))')"
export APP_ENV=development
flask --app app run
```

```powershell
# Windows PowerShell
$env:SECRET_KEY = python -c "import secrets; print(secrets.token_hex(32))"
$env:APP_ENV = "development"
flask --app app run
```

Browse to `http://127.0.0.1:5000`. The development server is for local use only. Production is served through Gunicorn or the container configuration.

## Deploy with Docker

Build and run locally:

```bash
docker build -t cardiolens .
docker run --rm -p 8000:8000 \
  -e APP_ENV=production \
  -e SECRET_KEY="$(python -c 'import secrets; print(secrets.token_hex(32))')" \
  cardiolens
```

The service listens on port 8000. For a real deployment, provide the secret through your hosting provider's secret manager—never commit it or put it in an image. Terminate HTTPS at a trusted hosting proxy/load balancer and use a managed platform with health checks, monitoring, backups for any separately added data, and an incident-response plan.

## Deploy to Render

The included `render.yaml` is a starting point for a Render Blueprint deployment. Connect the repository in Render and deploy the Blueprint; Render generates `SECRET_KEY`. Confirm the Python version, service plan, health check, and current platform settings in the Render dashboard before launching. The web service is stateless and does not require persistent disks for the included app.

The configuration is a deployment example, not a guarantee of uptime, security compliance, clinical readiness, or suitability for handling protected health information.

## Model artifacts and training

The app loads the classifier and scaler at startup. By default, it expects `trained_model.pkl` and `scaler.pkl` beside `app.py`. You can override these paths with `MODEL_PATH` and `SCALER_PATH`.

To regenerate the artifacts from the workbook:

```bash
python trainmodel.py
```

The training script writes the two artifacts into the current working directory. Retrain and validate the artifacts in the same pinned Python/scikit-learn environment used for deployment. Pickle/joblib files can execute code when loaded: only load artifacts from a trusted source, and keep the training and serving dependency versions compatible. The model's input feature names and order must match the application schema in `app.py`.

The older `Data_Spliting.py` and `AllModelsTraining.py` scripts are research/experimentation utilities and may need path or code fixes before use; they are not part of the production web startup process.

## Health endpoints

- `GET /health/live` — process liveness.
- `GET /health/ready` — reports readiness after model artifacts load successfully.

## Production-readiness limitations

This work hardens the software delivery path, but it does not make the prediction model fit for real-world medical use. Before any clinical or patient-facing use, this project would need, at minimum, independent clinical validation, calibration and subgroup/fairness analysis, documented data provenance and consent, privacy/security and regulatory review, an appropriate clinical safety process, accessibility review, and qualified clinical oversight. The dataset, model, input encoding, and displayed probability must be evaluated by domain experts. Do not use this deployment for real patient care or protected health information.

## License

No license is currently specified. Contact the repository owner before reuse or redistribution.
