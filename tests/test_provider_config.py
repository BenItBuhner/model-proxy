"""
Tests for provider configuration system.
"""

import pytest
from app.core.provider_config import (
    load_provider_config,
    get_all_provider_configs,
    get_provider_config,
    reload_provider_config,
    validate_provider_config,
    get_provider_endpoint,
    get_provider_auth_headers,
    get_provider_env_var_patterns,
    is_provider_enabled,
)


def test_load_provider_config_openai():
    """Test loading OpenAI provider config."""
    config = load_provider_config("openai")
    assert config["name"] == "openai"
    assert config["enabled"]
    assert "endpoints" in config
    assert "authentication" in config
    assert config["endpoints"]["base_url"] == "https://api.openai.com/v1"


def test_load_provider_config_anthropic():
    """Test loading Anthropic provider config."""
    config = load_provider_config("anthropic")
    assert config["name"] == "anthropic"
    assert config["enabled"]
    assert "endpoints" in config
    assert config["endpoints"]["base_url"] == "https://api.anthropic.com/v1"


def test_load_provider_config_not_found():
    """Test loading non-existent provider config."""
    with pytest.raises(FileNotFoundError):
        load_provider_config("nonexistent")


def test_get_all_provider_configs():
    """Test getting all provider configs."""
    configs = get_all_provider_configs()
    assert isinstance(configs, dict)
    assert "openai" in configs
    assert "anthropic" in configs


def test_get_provider_config():
    """Test getting provider config."""
    config = get_provider_config("openai")
    assert config is not None
    assert config["name"] == "openai"


def test_get_provider_config_not_found():
    """Test getting non-existent provider config."""
    config = get_provider_config("nonexistent")
    assert config is None


def test_reload_provider_config():
    """Test reloading provider config."""
    config1 = get_provider_config("openai")
    config2 = reload_provider_config("openai")
    assert config1["name"] == config2["name"]


def test_validate_provider_config():
    """Test validating provider config."""
    config = {
        "name": "test",
        "enabled": True,
        "endpoints": {"base_url": "https://api.test.com"},
        "authentication": {"type": "bearer", "header_name": "Authorization"},
    }
    result = validate_provider_config(config)
    assert result


def test_validate_provider_config_invalid():
    """Test validating invalid provider config."""
    config = {
        "name": "test"
        # Missing required fields
    }
    with pytest.raises(ValueError):
        validate_provider_config(config)


def test_get_provider_endpoint():
    """Test getting provider endpoint URL."""
    endpoint = get_provider_endpoint("openai", "completions")
    assert endpoint.startswith("https://")
    assert "chat/completions" in endpoint


def test_get_provider_endpoint_streaming():
    """Test getting streaming endpoint URL."""
    endpoint = get_provider_endpoint("openai", "streaming")
    assert endpoint.startswith("https://")


def test_get_provider_auth_headers_openai():
    """Test getting OpenAI auth headers."""
    headers = get_provider_auth_headers("openai", "test_key")
    assert "Authorization" in headers
    assert headers["Authorization"] == "Bearer test_key"


def test_get_provider_auth_headers_anthropic():
    """Test getting Anthropic auth headers."""
    headers = get_provider_auth_headers("anthropic", "test_key")
    assert "x-api-key" in headers
    assert headers["x-api-key"] == "test_key"
    assert "anthropic-version" in headers


def test_get_provider_env_var_patterns():
    """Test getting environment variable patterns."""
    patterns = get_provider_env_var_patterns("openai")
    assert isinstance(patterns, list)
    assert len(patterns) > 0
    assert any("OPENAI_API_KEY" in pattern for pattern in patterns)


def test_is_provider_enabled():
    """Test checking if provider is enabled."""
    enabled = is_provider_enabled("openai")
    assert enabled


def test_is_provider_enabled_not_found():
    """Test checking non-existent provider."""
    enabled = is_provider_enabled("nonexistent")
    assert not enabled


def test_provider_config_proxy_support():
    """Test provider config with proxy support."""
    config = get_provider_config("openai")
    assert "proxy_support" in config
    assert "enabled" in config["proxy_support"]
    assert "base_url_override" in config["proxy_support"]
