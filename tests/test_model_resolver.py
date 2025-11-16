"""
Tests for model resolver module.
"""
import pytest
from fastapi import HTTPException
from app.core.model_resolver import resolve_model, get_available_models


def test_resolve_model_cerebras():
    """Test resolving Cerebras GLM-4.6 model."""
    config = resolve_model("cerebras/glm-4.6")
    assert config["provider"] == "cerebras"
    assert config["provider_model"] == "zai-glm-4.6"


def test_resolve_model_groq():
    """Test resolving Groq Kimi K2 model."""
    config = resolve_model("groq/kimi-k2")
    assert config["provider"] == "groq"
    assert config["provider_model"] == "moonshotai/kimi-k2-instruct-0905"


def test_resolve_model_not_found():
    """Test resolving non-existent model."""
    with pytest.raises(HTTPException) as exc_info:
        resolve_model("non-existent-model")
    
    assert exc_info.value.status_code == 400
    assert "not found" in exc_info.value.detail.lower()


def test_get_available_models():
    """Test getting list of available models."""
    models = get_available_models()
    assert isinstance(models, list)
    assert len(models) == 19  # We have 19 models in the new config
    assert "cerebras/glm-4.6" in models
    assert "groq/kimi-k2" in models
    assert "nahcrof/qwen3-coder" in models


def test_resolve_model_nahcrof():
    """Test resolving Nahcrof GLM-4.6 model."""
    config = resolve_model("nahcrof/glm-4.6")
    assert config["provider"] == "nahcrof"
    assert config["provider_model"] == "glm-4.6"


def test_resolve_model_longcat():
    """Test resolving LongCat Flash model."""
    config = resolve_model("longcat/longcat-flash")
    assert config["provider"] == "longcat"
    assert config["provider_model"] == "LongCat-Flash-Chat"

