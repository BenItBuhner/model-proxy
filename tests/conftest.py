import pytest
import os
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient

# Set environment variables before importing app modules
os.environ.setdefault("CLIENT_API_KEY", "test_client_key_123")
os.environ.setdefault("OPENAI_API_KEY_1", "test_openai_key_1")
os.environ.setdefault("ANTHROPIC_API_KEY_1", "test_anthropic_key_1")

from app.main import app
from app.auth import verify_client_api_key, CLIENT_API_KEY
from app.core.model_resolver import resolve_model, get_available_models
from app.core.api_key_manager import (
    get_api_key,
    mark_key_failed,
    get_available_keys,
    reset_failed_keys,
    _parse_provider_keys
)
from app.core.format_converters import (
    anthropic_to_openai_request,
    openai_to_anthropic_request,
    anthropic_to_openai_response,
    openai_to_anthropic_response
)
from app.providers.openai_provider import OpenAIProvider
from app.providers.anthropic_provider import AnthropicProvider

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
