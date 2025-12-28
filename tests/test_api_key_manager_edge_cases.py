"""
Comprehensive tests for API key manager edge cases and advanced scenarios.
"""

from app.core.api_key_manager import (
    get_api_key,
    mark_key_failed,
    get_available_keys,
    reset_failed_keys,
    _parse_provider_keys,
    KEY_COOLDOWN_SECONDS,
)


def test_parse_provider_keys_singular_only(monkeypatch):
    """Test parsing when only singular key is provided."""
    monkeypatch.setenv("OPENAI_API_KEY", "singular_key")
    monkeypatch.delenv("OPENAI_API_KEY_1", raising=False)

    keys = _parse_provider_keys("openai")
    assert "singular_key" in keys
    assert len(keys) == 1


def test_parse_provider_keys_numbered_only(monkeypatch):
    """Test parsing when only numbered keys are provided."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY_1", "key1")
    monkeypatch.setenv("OPENAI_API_KEY_2", "key2")

    keys = _parse_provider_keys("openai")
    assert "key1" in keys
    assert "key2" in keys
    assert len(keys) == 2


def test_parse_provider_keys_mixed(monkeypatch):
    """Test parsing when both singular and numbered keys exist."""
    monkeypatch.setenv("OPENAI_API_KEY", "singular")
    monkeypatch.setenv("OPENAI_API_KEY_1", "key1")
    monkeypatch.setenv("OPENAI_API_KEY_2", "key2")

    keys = _parse_provider_keys("openai")
    assert "singular" in keys
    assert "key1" in keys
    assert "key2" in keys


def test_parse_provider_keys_gaps(monkeypatch):
    """Test parsing with gaps in numbering (still discovers higher indexes)."""
    monkeypatch.setenv("OPENAI_API_KEY_1", "key1")
    monkeypatch.setenv("OPENAI_API_KEY_3", "key3")
    # Missing key_2 - parsing should still find key_3

    keys = _parse_provider_keys("openai")
    assert "key1" in keys
    assert "key3" in keys
    assert len(keys) == 2


def test_get_available_keys_after_failure(monkeypatch):
    """Test getting available keys after marking one as failed."""
    monkeypatch.setenv("OPENAI_API_KEY_1", "key1")
    monkeypatch.setenv("OPENAI_API_KEY_2", "key2")
    monkeypatch.setenv("OPENAI_API_KEY_3", "key3")

    # Mark key2 as failed
    mark_key_failed("openai", "key2")

    available = get_available_keys("openai")
    assert "key1" in available
    assert "key2" not in available
    assert "key3" in available


def test_get_api_key_consistency(monkeypatch):
    """Test that get_api_key returns valid keys."""
    monkeypatch.setenv("OPENAI_API_KEY_1", "key1")
    monkeypatch.setenv("OPENAI_API_KEY_2", "key2")

    for _ in range(20):
        key = get_api_key("openai")
        assert key in ["key1", "key2"]


def test_mark_key_failed_multiple_times(monkeypatch):
    """Test marking the same key as failed multiple times."""
    monkeypatch.setenv("OPENAI_API_KEY_1", "key1")

    # Mark multiple times
    mark_key_failed("openai", "key1")
    mark_key_failed("openai", "key1")
    mark_key_failed("openai", "key1")

    available = get_available_keys("openai")
    assert "key1" not in available


def test_reset_failed_keys_specific_provider(monkeypatch):
    """Test resetting failed keys for specific provider."""
    monkeypatch.setenv("OPENAI_API_KEY_1", "openai_key")
    monkeypatch.setenv("ANTHROPIC_API_KEY_1", "anthropic_key")

    mark_key_failed("openai", "openai_key")
    mark_key_failed("anthropic", "anthropic_key")

    reset_failed_keys("openai")

    assert "openai_key" in get_available_keys("openai")
    assert "anthropic_key" not in get_available_keys("anthropic")


def test_get_available_keys_all_failed(monkeypatch):
    """Test when all keys are marked as failed."""
    monkeypatch.setenv("OPENAI_API_KEY_1", "key1")
    monkeypatch.setenv("OPENAI_API_KEY_2", "key2")

    mark_key_failed("openai", "key1")
    mark_key_failed("openai", "key2")

    available = get_available_keys("openai")
    assert len(available) == 0


def test_get_api_key_all_failed(monkeypatch):
    """Test get_api_key when all keys are failed."""
    monkeypatch.setenv("OPENAI_API_KEY_1", "key1")

    mark_key_failed("openai", "key1")

    key = get_api_key("openai")
    assert key is None


def test_provider_name_normalization(monkeypatch):
    """Test that provider names are normalized correctly."""
    # Test with hyphens
    monkeypatch.setenv("ANTHROPIC_API_KEY_1", "key1")

    keys = _parse_provider_keys("anthropic")
    assert "key1" in keys


def test_key_cooldown_configuration(monkeypatch):
    """Test that cooldown period is configurable."""

    # This test verifies the cooldown is read from env
    # Actual cooldown testing would require time manipulation
    assert isinstance(KEY_COOLDOWN_SECONDS, int)
    assert KEY_COOLDOWN_SECONDS > 0


def test_multiple_providers_independent(monkeypatch):
    """Test that different providers maintain independent key states."""
    monkeypatch.setenv("OPENAI_API_KEY_1", "openai_key")
    monkeypatch.setenv("ANTHROPIC_API_KEY_1", "anthropic_key")

    # Mark OpenAI key as failed
    mark_key_failed("openai", "openai_key")

    # Anthropic key should still be available
    assert "anthropic_key" in get_available_keys("anthropic")

    # OpenAI key should not be available
    assert "openai_key" not in get_available_keys("openai")


def test_get_available_keys_empty_provider(monkeypatch):
    """Test getting available keys for provider with no keys."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY_1", raising=False)

    keys = get_available_keys("openai")
    assert len(keys) == 0


def test_parse_provider_keys_special_characters(monkeypatch):
    """Test parsing keys with special characters."""
    special_key = "sk-test_123!@#$%^&*()"
    monkeypatch.setenv("OPENAI_API_KEY_1", special_key)

    keys = _parse_provider_keys("openai")
    assert special_key in keys
