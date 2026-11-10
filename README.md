# Heart Disease Detection — CardioLens educational demo

I built this as a machine-learning learning project with a web interface for exploring a saved classifier. It is an educational software demo only.

> **Not a medical product:** This application is not a medical device or clinical decision-support system. It has not been clinically validated and must not be used for diagnosis, screening, treatment, triage, or patient-care decisions. Do not enter real or identifiable health information. Consult a qualified healthcare professional about health concerns; contact local emergency services in an emergency.

## Included

- Flask web application with server-side validation, CSRF protection, security headers, and readiness/liveness routes.
- Responsive CardioLens-branded demo interface. It does not create user accounts, collect payments, or persist form inputs.
- Reproducible baseline training/evaluation script with a held-out test set and stratified cross-validation model selection.
- Docker, Render, Railway, DigitalOcean App Platform, and GitHub Actions examples.

## Local setup

Requires Python 3.11.

```bash
git clone https://github.com/asifm6414/Heart-disease-detection-Using-ml.git
cd Heart-disease-detection-Using-ml
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt -r requirements-dev.txt
