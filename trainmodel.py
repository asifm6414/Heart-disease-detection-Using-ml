"""Train and evaluate a small set of educational baseline classifiers.

This project is a demo only and not a clinical system.
The purpose is to train models and save artifacts for use by the app.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent
DATA_PATH = Path(os.environ.get("DATA_PATH", ROOT / "Heart_disease_cleveland_new.xlsx"))
FEATURES = (
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
)
RANDOM_STATE = 42


def atomic_joblib_dump(value, destination: Path) -> None:
    """Save a model artifact safely using a temporary file swap.

    This avoids partial writes and keeps the final file consistent.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)

    try:
        joblib.dump(value, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    """Train baseline models and save the selected model + scaler."""

    data = pd.read_excel(DATA_PATH)

    missing = set((*FEATURES, "target")) - set(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    # Keep only model features and target
    data = data.loc[:, [*FEATURES, "target"]].copy()

    # Convert columns to numeric to reduce type-related errors
    data[FEATURES] = data[FEATURES].apply(pd.to_numeric, errors="coerce")
    data["target"] = pd.to_numeric(data["target"], errors="coerce")

    # Remove rows with missing target label
    data = data.dropna(subset=["target"])

    if set(data["target"].unique()) != {0, 1}:
        raise ValueError("Expected a binary target encoded as 0 and 1.")

    X, y = data.loc[:, FEATURES], data["target"].astype(int)

    counts = y.value_counts()
    if counts.min() < 3:
        raise ValueError("At least three observations per class are needed for stratified evaluation.")

    # Split into train and test before model tuning
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Use a reasonable number of folds if enough data exists
    folds = min(5, int(y_train.value_counts().min()))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)

    candidates = {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "support_vector_machine": SVC(
            C=1.0,
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }

    scores = {}

    for name, estimator in candidates.items():
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", estimator),
            ]
        )

        result = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring={"roc_auc": "roc_auc", "balanced_accuracy": "balanced_accuracy"},
            n_jobs=1,
            error_score="raise",
        )

        scores[name] = {
            "mean_cv_roc_auc": float(result["test_roc_auc"].mean()),
            "std_cv_roc_auc": float(result["test_roc_auc"].std()),
            "mean_cv_balanced_accuracy": float(result["test_balanced_accuracy"].mean()),
        }

        print(f"{name}: {scores[name]}")

    # Choose model based only on training data cross-validation
    best_name = max(scores, key=lambda key: scores[key]["mean_cv_roc_auc"])

    final_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", candidates[best_name]),
        ]
    )

    final_pipeline.fit(X_train, y_train)

    predictions = final_pipeline.predict(X_test)
    probabilities = final_pipeline.predict_proba(X_test)[:, list(final_pipeline.classes_).index(1)]

    test_metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "classification_report": classification_report(
            y_test,
            predictions,
            zero_division=0,
            output_dict=True,
        ),
    }

    # Save the model and scaler to disk
    atomic_joblib_dump(final_pipeline.named_steps["classifier"], ROOT / "trained_model.pkl")
    atomic_joblib_dump(final_pipeline.named_steps["scaler"], ROOT / "scaler.pkl")

    metrics = {
        "selected_model": best_name,
        "selection_metric": "mean stratified CV ROC-AUC on training partition",
        "cross_validation": scores,
        "held_out_test": test_metrics,
        "feature_order": list(FEATURES),
        "warning": "Exploratory metrics only; not evidence of clinical validity or suitability for care.",
    }

    (ROOT / "model_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Selected model: {best_name}")
    print(
        "Held-out metrics: "
        f"{json.dumps({k: v for k, v in test_metrics.items() if k != 'classification_report'}, indent=2)}"
    )
    print("Artifacts saved. These scores are not clinical validation.")

if __name__ == "__main__":
    main()
