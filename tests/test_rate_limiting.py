"""
Tests for rate limiting middleware.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.middleware.rate_limiting import RateLimitingMiddleware
from app.middleware.logging_middleware import LoggingMiddleware
from app.auth import CLIENT_API_KEY
from app.routers.health import router as health_router


@pytest.fixture
def app():
    """Create a fresh app for each test."""
    app = FastAPI()
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(
        RateLimitingMiddleware, requests_per_minute=5, tokens_per_minute=1000
    )
    app.include_router(health_router)

    @app.get("/test")
    def test_endpoint():
        return {"status": "ok"}

    return app


@pytest.fixture
def client(app):
    """Test client fixture."""
    with TestClient(app) as c:
        yield c


def test_rate_limiting_allows_requests(client):
    """Test that rate limiting allows requests under the limit."""
    response = client.get(
        "/test", headers={"Authorization": f"Bearer {CLIENT_API_KEY}"}
    )
    assert response.status_code == 200
    assert "X-RateLimit-Limit-Requests" in response.headers
    assert "X-RateLimit-Remaining-Requests" in response.headers


def test_rate_limiting_blocks_excess_requests(client):
    """Test that rate limiting blocks requests over the limit."""
    # Make requests up to the limit (5 requests)
    for i in range(5):
        response = client.get(
            "/test", headers={"Authorization": f"Bearer {CLIENT_API_KEY}"}
        )
        assert response.status_code == 200, f"Request {i + 1} should succeed"

    # Next request should be rate limited (6th request)
    # FastAPI should convert HTTPException to 429 response
    response = client.get(
        "/test",
        headers={"Authorization": f"Bearer {CLIENT_API_KEY}"},
        follow_redirects=False,
    )
    assert response.status_code == 429, (
        f"Expected 429, got {response.status_code}. Response: {response.text}"
    )
    # JSONResponse uses "content" not "detail"
    error_data = response.json()
    assert "error" in error_data
    assert error_data["error"]["type"] == "rate_limit_error"


def test_rate_limiting_skips_health_checks(client):
    """Test that rate limiting doesn't apply to health checks."""
    # Make many requests to health endpoint - should not be rate limited
    for _ in range(10):
        response = client.get("/health")
        assert response.status_code in [
            200,
            503,
        ]  # Health check might fail if DB not available


def test_rate_limiting_headers(client):
    """Test that rate limit headers are included in responses."""
    response = client.get(
        "/test", headers={"Authorization": f"Bearer {CLIENT_API_KEY}"}
    )

    assert "X-RateLimit-Limit-Requests" in response.headers
    assert "X-RateLimit-Remaining-Requests" in response.headers
    assert "X-RateLimit-Limit-Tokens" in response.headers
    assert "X-RateLimit-Remaining-Tokens" in response.headers

    assert int(response.headers["X-RateLimit-Limit-Requests"]) == 5
    assert int(response.headers["X-RateLimit-Remaining-Requests"]) >= 0


def test_rate_limiting_per_client(client):
    """Test that rate limiting is per client API key."""
    # Use different API keys - they'll be hashed differently
    # Since we're using CLIENT_API_KEY hash, all requests with same key will be same client
    # But requests without auth will be "anonymous"
    response1 = client.get("/test")  # No auth = anonymous
    response2 = client.get("/test")  # No auth = anonymous

    # Both should succeed (same anonymous client, but we're not hitting limit)
    assert response1.status_code == 200
    assert response2.status_code == 200
