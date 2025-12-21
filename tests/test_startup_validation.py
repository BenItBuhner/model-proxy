"""
Tests for startup validation functionality.
"""

import pytest
import os
from app.core.startup_validation import (
    validate_database,
    validate_client_api_key,
    validate_provider_configs,
    validate_provider_api_keys,
    validate_model_config,
    validate_startup,
    StartupValidationError,
)


def test_validate_database_healthy():
    """Test database validation when healthy."""
    is_valid, error = validate_database()
    assert is_valid is True
    assert error == ""


def test_validate_client_api_key_present(monkeypatch):
    """Test client API key validation when present."""
    monkeypatch.setenv("CLIENT_API_KEY", "test_key")
    is_valid, error = validate_client_api_key()
    assert is_valid is True
    assert error == ""


def test_validate_client_api_key_missing_not_required(monkeypatch):
    """Test client API key validation when missing but not required."""
    monkeypatch.delenv("CLIENT_API_KEY", raising=False)
    monkeypatch.setenv("REQUIRE_CLIENT_API_KEY", "false")
    is_valid, error = validate_client_api_key()
    assert is_valid is True


def test_validate_client_api_key_missing_required(monkeypatch):
    """Test client API key validation when missing and required."""
    monkeypatch.delenv("CLIENT_API_KEY", raising=False)
    monkeypatch.setenv("REQUIRE_CLIENT_API_KEY", "true")
    is_valid, error = validate_client_api_key()
    assert is_valid is False
    assert "CLIENT_API_KEY" in error


def test_validate_provider_configs():
    """Test provider config validation."""
    is_valid, errors = validate_provider_configs()
    assert is_valid is True
    assert len(errors) == 0


def test_validate_provider_api_keys(monkeypatch):
    """Test provider API key validation."""
    monkeypatch.setenv("OPENAI_API_KEY_1", "test_key_1")
    monkeypatch.setenv("ANTHROPIC_API_KEY_1", "test_key_2")

    is_valid, warnings = validate_provider_api_keys()
    assert is_valid is True


def test_validate_provider_api_keys_none(monkeypatch):
    """Test provider API key validation when none are configured."""
    # Clear all API keys
    for key in os.environ.keys():
        if "API_KEY" in key:
            monkeypatch.delenv(key, raising=False)

    is_valid, warnings = validate_provider_api_keys()
    # Should fail if no keys are configured
    assert is_valid is False
    assert len(warnings) > 0


def test_validate_model_config():
    """Test model config validation."""
    is_valid, error = validate_model_config()
    assert is_valid is True
    assert error == ""


def test_validate_startup_success(monkeypatch):
    """Test full startup validation when everything is configured."""
    monkeypatch.setenv("CLIENT_API_KEY", "test_key")
    monkeypatch.setenv("OPENAI_API_KEY_1", "test_key_1")
    monkeypatch.setenv("ANTHROPIC_API_KEY_1", "test_key_2")

    # Should not raise
    try:
        validate_startup()
    except StartupValidationError:
        pytest.fail("Startup validation should pass")


def test_validate_startup_failure(monkeypatch):
    """Test startup validation failure."""
    # Clear all API keys
    for key in list(os.environ.keys()):
        if "API_KEY" in key:
            monkeypatch.delenv(key, raising=False)

    # validate_startup() always raises if there are errors
    # The FAIL_ON_STARTUP_VALIDATION check is in main.py, not here
    with pytest.raises(StartupValidationError):
        validate_startup()
