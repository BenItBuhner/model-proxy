"""
Tests for structured logging core utilities.
"""

import uuid
from app.core.logging import (
    generate_request_id,
    hash_api_key,
    extract_parameters_from_request,
    extract_usage_from_response,
    extract_response_content,
)


def test_generate_request_id():
    """Test request ID generation."""
    request_id = generate_request_id()
    assert isinstance(request_id, str)
    # Should be a valid UUID
    uuid.UUID(request_id)


def test_generate_request_id_unique():
    """Test that request IDs are unique."""
    id1 = generate_request_id()
    id2 = generate_request_id()
    assert id1 != id2


def test_hash_api_key():
    """Test API key hashing."""
    api_key = "test_key_123"
    hashed = hash_api_key(api_key)
    assert isinstance(hashed, str)
    assert len(hashed) == 64  # SHA-256 hex digest length
    assert hashed != api_key


def test_hash_api_key_consistent():
    """Test that hashing is consistent."""
    api_key = "test_key_123"
    hashed1 = hash_api_key(api_key)
    hashed2 = hash_api_key(api_key)
    assert hashed1 == hashed2


def test_extract_parameters_from_request():
    """Test parameter extraction from request."""
    request_dict = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0.7,
        "max_tokens": 100,
        "top_p": 0.9,
        "n": 1,
        "stop": ["\n"],
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "user": "test_user",
        "stream": False,
    }
    parameters = extract_parameters_from_request(request_dict)
    assert parameters["temperature"] == 0.7
    assert parameters["max_tokens"] == 100
    assert parameters["top_p"] == 0.9
    assert parameters["n"] == 1
    assert parameters["stop"] == ["\n"]
    assert "model" not in parameters
    assert "messages" not in parameters


def test_extract_parameters_from_request_partial():
    """Test parameter extraction with partial parameters."""
    request_dict = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0.7,
    }
    parameters = extract_parameters_from_request(request_dict)
    assert parameters["temperature"] == 0.7
    assert "max_tokens" not in parameters


def test_extract_usage_from_response_openai():
    """Test usage extraction from OpenAI format."""
    response = {
        "id": "chatcmpl-123",
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }
    usage = extract_usage_from_response(response)
    assert usage["prompt_tokens"] == 10
    assert usage["completion_tokens"] == 20
    assert usage["total_tokens"] == 30


def test_extract_usage_from_response_anthropic():
    """Test usage extraction from Anthropic format."""
    response = {"id": "msg_123", "usage": {"input_tokens": 10, "output_tokens": 20}}
    usage = extract_usage_from_response(response)
    assert usage["prompt_tokens"] == 10
    assert usage["completion_tokens"] == 20
    assert usage["total_tokens"] == 30


def test_extract_usage_from_response_none():
    """Test usage extraction when missing."""
    response = {"id": "test"}
    usage = extract_usage_from_response(response)
    assert usage is None


def test_extract_response_content_openai():
    """Test content extraction from OpenAI format."""
    response = {
        "choices": [{"message": {"role": "assistant", "content": "Hello there!"}}]
    }
    content = extract_response_content(response, is_openai_format=True)
    assert content == "Hello there!"


def test_extract_response_content_anthropic():
    """Test content extraction from Anthropic format."""
    response = {"content": [{"type": "text", "text": "Hello there!"}]}
    content = extract_response_content(response, is_openai_format=False)
    assert content == "Hello there!"


def test_extract_response_content_none():
    """Test content extraction when missing."""
    response = {"id": "test"}
    content = extract_response_content(response, is_openai_format=True)
    assert content is None
