from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data

def test_ready_endpoint():
    response = client.get("/ready")
    assert response.status_code in (200, 503)

def test_predict_no_model():
    response = client.post("/predict", json={
        "Pclass": 3,
        "Sex": "male",
        "Age": 25.0,
        "SibSp": 0,
        "Parch": 0,
        "Fare": 7.25,
        "Embarked": "S",
    })
    assert response.status_code in (200, 503)

def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200


def test_predict_batch_no_model():
    response = client.post("/predict/batch", json=[
        {"Pclass": 3, "Sex": "male", "Age": 25.0, "SibSp": 0, "Parch": 0,
         "Fare": 7.25, "Embarked": "S"},
        {"Pclass": 1, "Sex": "female", "Age": 30.0, "SibSp": 1, "Parch": 0,
         "Fare": 100.0, "Embarked": "C"},
    ])
    assert response.status_code in (200, 503)
    if response.status_code == 200:
        data = response.json()
        assert "пассажиры" in data
        assert len(data["пассажиры"]) == 2
