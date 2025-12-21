"""
Tests for API key manager module.
"""

from app.core.api_key_manager import (
    get_api_key,
    mark_key_failed,
    get_available_keys,
    reset_failed_keys,
    _parse_provider_keys,
)


def test_parse_provider_keys_openai(monkeypatch):
    """Test parsing OpenAI API keys."""
    monkeypatch.setenv("OPENAI_API_KEY", "key1")
    monkeypatch.setenv("OPENAI_API_KEY_1", "key1")
    monkeypatch.setenv("OPENAI_API_KEY_2", "key2")

    keys = _parse_provider_keys("openai")
    assert "key1" in keys
    assert "key2" in keys
    assert len(keys) >= 2


def test_parse_provider_keys_anthropic(monkeypatch):
    """Test parsing Anthropic API keys."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anth_key1")
    monkeypatch.setenv("ANTHROPIC_API_KEY_1", "anth_key1")
    monkeypatch.setenv("ANTHROPIC_API_KEY_2", "anth_key2")

    keys = _parse_provider_keys("anthropic")
    assert "anth_key1" in keys
    assert "anth_key2" in keys


def test_get_available_keys(monkeypatch):
    """Test getting available keys."""
    monkeypatch.setenv("OPENAI_API_KEY_1", "key1")
    monkeypatch.setenv("OPENAI_API_KEY_2", "key2")

    keys = get_available_keys("openai")
    assert len(keys) >= 2
    assert "key1" in keys
    assert "key2" in keys


def test_get_api_key_random_selection(monkeypatch):
    """Test random API key selection."""
    monkeypatch.setenv("OPENAI_API_KEY_1", "key1")
    monkeypatch.setenv("OPENAI_API_KEY_2", "key2")

    # Get multiple keys to ensure randomness
    keys_selected = set()
    for _ in range(10):
        key = get_api_key("openai")
        assert key is not None
        keys_selected.add(key)

    # Should have selected at least one key
    assert len(keys_selected) > 0


def test_mark_key_failed(monkeypatch):
    """Test marking a key as failed."""
    monkeypatch.setenv("OPENAI_API_KEY_1", "key1")
    monkeypatch.setenv("OPENAI_API_KEY_2", "key2")

    # Mark a key as failed
    mark_key_failed("openai", "key1")

    # Check that key1 is not in available keys
    available = get_available_keys("openai")
    assert "key1" not in available
    assert "key2" in available


def test_failed_key_cooldown(monkeypatch):
    """Test that failed keys become available after cooldown."""
    monkeypatch.setenv("OPENAI_API_KEY_1", "key1")

    # Mark key as failed
    mark_key_failed("openai", "key1")

    # Should not be available immediately
    available = get_available_keys("openai")
    assert "key1" not in available

    # Reset failed keys to simulate cooldown passing
    reset_failed_keys("openai")

    # Now should be available
    available = get_available_keys("openai")
    assert "key1" in available


def test_reset_failed_keys(monkeypatch):
    """Test resetting failed keys."""
    monkeypatch.setenv("OPENAI_API_KEY_1", "key1")
    monkeypatch.setenv("ANTHROPIC_API_KEY_1", "anth_key1")

    # Mark keys as failed
    mark_key_failed("openai", "key1")
    mark_key_failed("anthropic", "anth_key1")

    # Reset OpenAI keys
    reset_failed_keys("openai")

    # OpenAI key should be available, Anthropic should not
    assert "key1" in get_available_keys("openai")
    assert "anth_key1" not in get_available_keys("anthropic")

    # Reset all
    reset_failed_keys()
    assert "anth_key1" in get_available_keys("anthropic")


def test_get_available_keys_empty(monkeypatch):
    """Test getting available keys when none configured."""
    # Remove all OpenAI keys
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY_1", raising=False)

    keys = get_available_keys("openai")
    assert len(keys) == 0


def test_get_api_key_no_keys(monkeypatch):
    """Test getting API key when none available."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY_1", raising=False)

    key = get_api_key("openai")
    assert key is None


def test_multiple_failed_keys(monkeypatch):
    """Test handling multiple failed keys."""
    monkeypatch.setenv("OPENAI_API_KEY_1", "key1")
    monkeypatch.setenv("OPENAI_API_KEY_2", "key2")
    monkeypatch.setenv("OPENAI_API_KEY_3", "key3")

    # Mark two keys as failed
    mark_key_failed("openai", "key1")
    mark_key_failed("openai", "key2")

    # Only key3 should be available
    available = get_available_keys("openai")
    assert "key1" not in available
    assert "key2" not in available
    assert "key3" in available
