import os
import sys
from fastapi.testclient import TestClient
from app.predict import app

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] in ["healthy", "unhealthy"]

def test_predict_valid_input():
    sample = [13.74, 1.67, 2.25, 18.6, 103.0, 2.6, 2.8, 0.26, 1.28, 4.18, 1.06, 3.4, 980]
    response = client.post("/predict", json={"features": sample})
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in [0, 1, 2]
    assert len(data["probabilities"]) == 3
    assert len(data["feature_names"]) == 13