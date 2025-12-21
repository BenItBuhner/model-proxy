"""
Tests for logging middleware.
"""

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from app.middleware.logging_middleware import LoggingMiddleware


@pytest.fixture
def app_with_middleware():
    """Create FastAPI app with logging middleware."""
    app = FastAPI()
    app.add_middleware(LoggingMiddleware)

    @app.get("/test")
    async def test_endpoint(request: Request):
        return {
            "request_id": getattr(request.state, "request_id", None),
            "start_time": getattr(request.state, "start_time", None),
        }

    return app


def test_logging_middleware_adds_request_id(app_with_middleware):
    """Test that middleware adds request ID to request state."""
    client = TestClient(app_with_middleware)
    response = client.get("/test")
    assert response.status_code == 200
    data = response.json()
    assert data["request_id"] is not None
    assert isinstance(data["request_id"], str)


def test_logging_middleware_adds_start_time(app_with_middleware):
    """Test that middleware adds start time to request state."""
    client = TestClient(app_with_middleware)
    response = client.get("/test")
    assert response.status_code == 200
    data = response.json()
    assert data["start_time"] is not None


def test_logging_middleware_adds_x_request_id_header(app_with_middleware):
    """Test that middleware adds X-Request-ID header to response."""
    client = TestClient(app_with_middleware)
    response = client.get("/test")
    assert response.status_code == 200
    assert "X-Request-ID" in response.headers
    assert response.headers["X-Request-ID"] is not None


def test_logging_middleware_unique_request_ids(app_with_middleware):
    """Test that middleware generates unique request IDs."""
    client = TestClient(app_with_middleware)
    response1 = client.get("/test")
    response2 = client.get("/test")

    assert response1.headers["X-Request-ID"] != response2.headers["X-Request-ID"]
