"""
Tests for error formatters.
"""
import pytest
from fastapi import HTTPException
from app.core.error_formatters import (
    format_openai_error,
    format_anthropic_error,
    create_provider_error_response
)


def test_format_openai_error():
    """Test OpenAI error formatting."""
    error = format_openai_error(400, "Invalid request", "invalid_request_error")
    
    assert "error" in error
    assert error["error"]["message"] == "Invalid request"
    assert error["error"]["type"] == "invalid_request_error"
    assert error["error"]["code"] == "invalid_request_error"


def test_format_openai_error_default_type():
    """Test OpenAI error formatting with default type."""
    error = format_openai_error(500, "Server error")
    
    assert "error" in error
    assert error["error"]["message"] == "Server error"
    assert error["error"]["type"] == "internal_server_error"
    assert error["error"]["code"] == "internal_server_error"


def test_format_anthropic_error():
    """Test Anthropic error formatting."""
    error = format_anthropic_error(400, "Invalid request", "invalid_request_error")
    
    assert "error" in error
    assert error["error"]["message"] == "Invalid request"
    assert error["error"]["type"] == "invalid_request_error"


def test_format_anthropic_error_default_type():
    """Test Anthropic error formatting with default type."""
    error = format_anthropic_error(500, "Server error")
    
    assert "error" in error
    assert error["error"]["message"] == "Server error"
    assert error["error"]["type"] == "internal_server_error"


def test_create_provider_error_response_openai():
    """Test creating OpenAI error response."""
    exc = create_provider_error_response("openai", 400, "Invalid request", "invalid_request_error")
    
    assert isinstance(exc, HTTPException)
    assert exc.status_code == 400
    assert isinstance(exc.detail, dict)
    assert "error" in exc.detail
    assert exc.detail["error"]["message"] == "Invalid request"


def test_create_provider_error_response_anthropic():
    """Test creating Anthropic error response."""
    exc = create_provider_error_response("anthropic", 400, "Invalid request", "invalid_request_error")
    
    assert isinstance(exc, HTTPException)
    assert exc.status_code == 400
    assert isinstance(exc.detail, dict)
    assert "error" in exc.detail
    assert exc.detail["error"]["message"] == "Invalid request"


def test_create_provider_error_response_unknown_provider():
    """Test creating error response for unknown provider."""
    exc = create_provider_error_response("unknown", 500, "Error", "api_error")
    
    assert isinstance(exc, HTTPException)
    assert exc.status_code == 500
    assert isinstance(exc.detail, dict)
    assert "error" in exc.detail

