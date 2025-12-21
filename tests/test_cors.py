"""
Tests for CORS middleware.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastapi.middleware.cors import CORSMiddleware
from app.middleware.logging_middleware import LoggingMiddleware


# Create test app with CORS
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://example.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(LoggingMiddleware)


@app.get("/test")
def cors_test_endpoint():
    return {"status": "ok"}


@pytest.fixture
def client():
    """Test client fixture."""
    with TestClient(app) as c:
        yield c


def test_cors_allows_origin(client):
    """Test that CORS allows requests from allowed origins."""
    response = client.options(
        "/test",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code == 200
    assert "Access-Control-Allow-Origin" in response.headers
    assert response.headers["Access-Control-Allow-Origin"] == "http://localhost:3000"


def test_cors_includes_headers(client):
    """Test that CORS includes necessary headers."""
    response = client.get("/test", headers={"Origin": "http://localhost:3000"})

    assert response.status_code == 200
    assert "Access-Control-Allow-Origin" in response.headers
    assert "Access-Control-Allow-Credentials" in response.headers


def test_cors_preflight_request(client):
    """Test CORS preflight OPTIONS request."""
    response = client.options(
        "/test",
        headers={
            "Origin": "https://example.com",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "Authorization",
        },
    )

    assert response.status_code == 200
    assert "Access-Control-Allow-Origin" in response.headers
    assert "Access-Control-Allow-Methods" in response.headers
