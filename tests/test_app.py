import numpy as np
import pytest

import app as app_module


class FakeScaler:
    def transform(self, values):
        return values.to_numpy(dtype=float)


class FakeModel:
    classes_ = np.array([0, 1])

    def predict(self, values):
        return np.array([1 if values[0, 0] > 50 else 0])

    def predict_proba(self, values):
        return np.array([[0.2, 0.8]])


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(app_module, "_load_artifacts", lambda: (FakeModel(), FakeScaler()))
    test_app = app_module.create_app({"TESTING": True, "WTF_CSRF_ENABLED": False})
    test_app.extensions["prediction_model"] = FakeModel()
    test_app.extensions["prediction_scaler"] = FakeScaler()
    return test_app.test_client()


def sample_form(**overrides):
    data = {
        "age": "55", "sex": "1", "cp": "0", "trestbps": "120", "chol": "200",
        "fbs": "0", "restecg": "1", "thalach": "150", "exang": "0",
        "oldpeak": "1.2", "slope": "1", "ca": "0", "thal": "2",
    }
    return data | overrides


def test_home_and_security_headers(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Not for medical use" in response.data
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["Content-Security-Policy"]


def test_health_endpoints(client):
    assert client.get("/health/live").json == {"status": "ok"}
    assert client.get("/health/ready").json == {"status": "ready"}


def test_prediction_returns_educational_estimate(client):
    response = client.post("/prediction", data=sample_form())
    assert response.status_code == 200
    assert b"flags this sample as positive" in response.data
    assert b"80.0%" in response.data


def test_missing_or_out_of_range_input_is_rejected(client):
    assert client.post("/prediction", data=sample_form(age="")).status_code == 400
    assert client.post("/prediction", data=sample_form(age="999")).status_code == 400


def test_non_finite_float_is_rejected(client):
    response = client.post("/prediction", data=sample_form(oldpeak="nan"))
    assert response.status_code == 400
