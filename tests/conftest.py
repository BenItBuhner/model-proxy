import os

import pytest
from fastapi.testclient import TestClient

# Set environment variables before importing app modules
os.environ.setdefault("CLIENT_API_KEY", "test_client_key_123")
os.environ.setdefault("OPENAI_API_KEY_1", "test_openai_key_1")
os.environ.setdefault("ANTHROPIC_API_KEY_1", "test_anthropic_key_1")


# Legacy model_resolver removed; tests should use routing.config_loader or
# app.routing.config_loader.config_loader where appropriate.
from app.core.api_key_manager import (
    reset_failed_keys,
)
from app.main import app


@pytest.fixture(scope="module")
def client():
    """Test client fixture."""
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def reset_api_keys():
    """Reset failed keys before each test."""
    reset_failed_keys()
    yield
    reset_failed_keys()


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock environment variables for testing."""
    monkeypatch.setenv("CLIENT_API_KEY", "test_client_key_123")
    monkeypatch.setenv("OPENAI_API_KEY", "test_openai_key_1")
    monkeypatch.setenv("OPENAI_API_KEY_1", "test_openai_key_1")
    monkeypatch.setenv("OPENAI_API_KEY_2", "test_openai_key_2")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test_anthropic_key_1")
    monkeypatch.setenv("ANTHROPIC_API_KEY_1", "test_anthropic_key_1")
    monkeypatch.setenv("ANTHROPIC_API_KEY_2", "test_anthropic_key_2")
    yield
    # Cleanup
    monkeypatch.delenv("CLIENT_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY_1", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY_2", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY_1", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY_2", raising=False)
