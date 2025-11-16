"""
Tests for health check endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from app.routers.health import (
    check_database,
    check_providers,
    check_model_config,
    check_provider_configs,
    get_overall_status
)


@pytest.fixture
def client():
    """Test client fixture."""
    from app.main import app
    with TestClient(app) as c:
        yield c


def test_health_check_basic(client):
    """Test basic health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "timestamp" in data
    assert data["status"] == "healthy"


def test_health_check_detailed(client):
    """Test detailed health check endpoint."""
    response = client.get("/health/detailed")
    # In test environment without API keys, system may be unhealthy (503)
    # but the endpoint should still return data
    assert response.status_code in [200, 503]
    data = response.json()

    # If healthy (200), response is direct
    if response.status_code == 200:
        assert "status" in data
        assert "timestamp" in data
        assert "uptime_seconds" in data
        assert "components" in data
        health_data = data
    else:
        # If unhealthy (503), response is wrapped in FastAPI error format
        assert "detail" in data
        health_data = data["detail"]
        assert "status" in health_data

    assert "timestamp" in health_data
    assert "uptime_seconds" in health_data
    assert "components" in health_data
    assert "database" in health_data["components"]
    assert "providers" in health_data["components"]
    assert "model_config" in health_data["components"]
    assert "provider_configs" in health_data["components"]
    # In test environment, status may be "unhealthy" due to missing API keys
    assert health_data["status"] in ["healthy", "degraded", "unhealthy"]


def test_check_database_healthy():
    """Test database check when healthy."""
    result = check_database()
    assert result["status"] == "healthy"
    assert "response_time_ms" in result


def test_check_providers():
    """Test provider check."""
    result = check_providers()
    assert isinstance(result, dict)
    # Should have at least openai and anthropic
    assert "openai" in result or "anthropic" in result


def test_check_model_config():
    """Test model config check."""
    result = check_model_config()
    assert result["status"] == "healthy"
    assert "models_count" in result
    assert result["models_count"] > 0


def test_check_provider_configs():
    """Test provider configs check."""
    result = check_provider_configs()
    assert result["status"] == "healthy"
    assert "providers_loaded" in result
    assert result["providers_loaded"] >= 2  # At least openai and anthropic


def test_get_overall_status_healthy():
    """Test overall status calculation when healthy."""
    components = {
        "database": {"status": "healthy"},
        "providers": {
            "openai": {"status": "healthy", "keys_available": 1},
            "anthropic": {"status": "healthy", "keys_available": 1}
        },
        "model_config": {"status": "healthy"},
        "provider_configs": {"status": "healthy"}
    }
    status = get_overall_status(components)
    assert status == "healthy"


def test_get_overall_status_degraded():
    """Test overall status calculation when degraded."""
    components = {
        "database": {"status": "healthy"},
        "providers": {
            "openai": {"status": "healthy", "keys_available": 0},
            "anthropic": {"status": "healthy", "keys_available": 1}
        },
        "model_config": {"status": "healthy"},
        "provider_configs": {"status": "healthy"}
    }
    status = get_overall_status(components)
    assert status == "degraded"


def test_get_overall_status_unhealthy():
    """Test overall status calculation when unhealthy."""
    components = {
        "database": {"status": "unhealthy"},
        "providers": {
            "openai": {"status": "healthy", "keys_available": 1}
        },
        "model_config": {"status": "healthy"},
        "provider_configs": {"status": "healthy"}
    }
    status = get_overall_status(components)
    assert status == "unhealthy"

