import pytest
from fastapi.testclient import TestClient
from execution.api import app

# Need to run tests under lifespan manager to load models properly
client = TestClient(app)

def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_search_endpoint_success():
    # Use TestClient with lifespan context manager manually or just invoke GET
    with TestClient(app) as client_with_lifespan:
        response = client_with_lifespan.get("/api/search?q=test&mode=bm25")
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert data["mode"] == "bm25"
        assert data["query"] == "test"

def test_search_endpoint_validation_error():
    # Invalid mode should be rejected by Pydantic
    with TestClient(app) as client_with_lifespan:
        response = client_with_lifespan.get("/api/search?q=test&mode=invalid_mode")
        assert response.status_code == 422  # Validation Error

def test_stats_endpoint():
    response = client.get("/api/stats")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
