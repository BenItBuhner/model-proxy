"""
Core logging utilities for structured request/response logging.
"""

import uuid
import hashlib
from typing import Optional, Dict, Any


def generate_request_id() -> str:
    """
    Generate a unique request ID (UUID v4).

    Returns:
        UUID string
    """
    return str(uuid.uuid4())


def hash_api_key(api_key: str) -> str:
    """
    Hash an API key using SHA-256 for secure storage.

    Args:
        api_key: API key to hash

    Returns:
        SHA-256 hash hex digest
    """
    return hashlib.sha256(api_key.encode()).hexdigest()


def extract_parameters_from_request(request_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract all relevant parameters from a request dictionary.

    Args:
        request_dict: Request dictionary

    Returns:
        Dictionary with extracted parameters
    """
    parameters = {}

    # Common parameters
    param_keys = [
        "temperature",
        "max_tokens",
        "top_p",
        "n",
        "stop",
        "presence_penalty",
        "frequency_penalty",
        "logit_bias",
        "user",
        "tools",
        "tool_choice",
        "stream",
        "top_k",  # Anthropic specific
    ]

    for key in param_keys:
        if key in request_dict:
            parameters[key] = request_dict[key]

    return parameters


def extract_usage_from_response(
    response_dict: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Extract token usage information from response.
    Handles both OpenAI and Anthropic response formats.

    Args:
        response_dict: Response dictionary

    Returns:
        Usage dictionary or None
    """
    if "usage" not in response_dict:
        return None

    usage = response_dict["usage"]

    # Check if it's Anthropic format (has input_tokens/output_tokens)
    if isinstance(usage, dict) and (
        "input_tokens" in usage or "output_tokens" in usage
    ):
        # Anthropic format
        return {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("input_tokens", 0)
            + usage.get("output_tokens", 0),
        }
    elif isinstance(usage, dict) and "prompt_tokens" in usage:
        # OpenAI format - return as is
        return usage

    return None


def extract_response_content(
    response_dict: Dict[str, Any], is_openai_format: bool = True
) -> Optional[str]:
    """
    Extract response content text from response dictionary.

    Args:
        response_dict: Response dictionary
        is_openai_format: Whether response is in OpenAI format

    Returns:
        Response content text or None
    """
    if is_openai_format:
        # OpenAI format: choices[0].message.content
        choices = response_dict.get("choices", [])
        if choices and len(choices) > 0:
            message = choices[0].get("message", {})
            return message.get("content")
    else:
        # Anthropic format: content[0].text
        content = response_dict.get("content", [])
        if content and len(content) > 0:
            if isinstance(content[0], dict) and content[0].get("type") == "text":
                return content[0].get("text")

    return None
