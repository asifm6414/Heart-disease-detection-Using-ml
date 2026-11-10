"""Flask web application for an educational heart-disease ML demo.

This application is not a medical device and must not be used for clinical decisions.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from flask import Flask, render_template, request
from flask_wtf.csrf import CSRFError, CSRFProtect
from werkzeug.exceptions import BadRequest

BASE_DIR = Path(__file__).resolve().parent
FEATURES = (
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
)

# Bounds/allowed encoded values are input-quality checks, not medical guidance.
INTEGER_FIELDS = {
    "age": (20, 100),
    "sex": (0, 1),
    "cp": (0, 3),
    "trestbps": (50, 300),
    "chol": (50, 800),
    "fbs": (0, 1),
    "restecg": (0, 2),
    "thalach": (40, 250),
    "exang": (0, 1),
    "slope": (0, 2),
    "ca": (0, 3),
    "thal": (1, 3),
}

csrf = CSRFProtect()


def _load_artifacts() -> tuple[Any, Any]:
    model_path = Path(os.environ.get("MODEL_PATH", BASE_DIR / "trained_model.pkl"))
    scaler_path = Path(os.environ.get("SCALER_PATH", BASE_DIR / "scaler.pkl"))
    if not model_path.is_file() or not scaler_path.is_file():
        raise RuntimeError(
            "Required model artifacts are missing. Provide trained_model.pkl and scaler.pkl, "
            "or set MODEL_PATH and SCALER_PATH."
        )
    try:
        # Only load trusted artifacts created and controlled by this project.
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
    except Exception as exc:
        raise RuntimeError("Could not load the configured model artifacts.") from exc

    for artifact, label in ((model, "model"), (scaler, "scaler")):
        names = getattr(artifact, "feature_names_in_", None)
        if names is not None and tuple(names) != FEATURES:
            raise RuntimeError(
                f"The {label} feature order does not match the application input schema. "
                "Retrain artifacts with the feature order documented in app.py."
            )
    return model, scaler


def _parse_form(form: Any) -> pd.DataFrame:
    values: dict[str, int | float] = {}
    for name in FEATURES:
        raw = form.get(name, "").strip()
        if not raw:
            raise ValueError(f"Please provide a value for {name}.")
        try:
            if name == "oldpeak":
                value = float(raw)
                if not pd.notna(value) or not 0 <= value <= 10:
                    raise ValueError("ST depression must be a number from 0 to 10.")
                values[name] = value
            else:
                value = int(raw)
                minimum, maximum = INTEGER_FIELDS[name]
                if not minimum <= value <= maximum:
                    raise ValueError(f"{name} must be between {minimum} and {maximum}.")
                values[name] = value
        except (TypeError, OverflowError):
            raise ValueError(f"Enter a valid number for {name}.") from None
        except ValueError as exc:
            if str(exc).startswith(("ST depression", f"{name} must be")):
                raise
            raise ValueError(f"Enter a valid number for {name}.") from None

    return pd.DataFrame([values], columns=FEATURES)


def create_app(test_config: dict[str, Any] | None = None) -> Flask:
    app = Flask(__name__, instance_relative_config=True)
    environment = os.environ.get("APP_ENV", "development").lower()
    secret_key = os.environ.get("SECRET_KEY")
    if environment == "production" and not secret_key:
        raise RuntimeError("Set a strong SECRET_KEY environment variable in production.")

    app.config.from_mapping(
        SECRET_KEY=secret_key or "local-development-only-change-me",
        MAX_CONTENT_LENGTH=16 * 1024,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=environment == "production",
        WTF_CSRF_TIME_LIMIT=3600,
        TESTING=False,
    )
    if test_config:
        app.config.update(test_config)

    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO").upper())
    csrf.init_app(app)

    try:
        model, scaler = _load_artifacts()
    except RuntimeError:
        app.logger.exception("Application startup failed while loading model artifacts")
        raise
    app.extensions["prediction_model"] = model
    app.extensions["prediction_scaler"] = scaler

    @app.after_request
    def add_security_headers(response):
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; style-src 'self'; script-src 'self'; "
            "img-src 'self' data:; form-action 'self'; base-uri 'self'; frame-ancestors 'none'",
        )
        response.headers.setdefault("Cache-Control", "no-store")
        if environment == "production":
            response.headers.setdefault(
                "Strict-Transport-Security", "max-age=31536000; includeSubDomains"
            )
        return response

    @app.get("/")
    def home():
        return render_template("index.html")

    @app.post("/prediction")
    def predict():
        try:
            data = _parse_form(request.form)
        except (ValueError, BadRequest) as exc:
            return render_template("index.html", error=str(exc)), 400

        try:
            scaler = app.extensions["prediction_scaler"]
            model = app.extensions["prediction_model"]
            transformed = scaler.transform(data)
            prediction = model.predict(transformed)[0]
            classes = list(getattr(model, "classes_", []))
            if not hasattr(model, "predict_proba") or 1 not in classes:
                raise RuntimeError("Model must support probabilities and include positive class 1.")
            probability = float(model.predict_proba(transformed)[0][classes.index(1)]) * 100
            result = (
                "The model estimates an elevated likelihood in this example."
                if int(prediction) == 1
                else "The model does not flag an elevated likelihood in this example."
            )
            return render_template(
                "index.html", result=result, probability=f"{probability:.1f}%"
            )
        except Exception:
            # Never log submitted form data: it may contain sensitive health information.
            app.logger.exception("Prediction failed")
            return render_template(
                "index.html",
                error="The prediction could not be completed. Please try again later.",
            ), 503

    @app.get("/health/live")
    def health_live():
        return {"status": "ok"}, 200

    @app.get("/health/ready")
    def health_ready():
        ready = bool(
            app.extensions.get("prediction_model")
            and app.extensions.get("prediction_scaler")
        )
        return {"status": "ready" if ready else "not_ready"}, 200 if ready else 503

    @app.errorhandler(CSRFError)
    def handle_csrf_error(_error):
        return render_template(
            "index.html", error="Your form session expired. Reload the page and try again."
        ), 400

    @app.errorhandler(413)
    def handle_payload_too_large(_error):
        return {"error": "Request is too large."}, 413

    return app


app = create_app()
