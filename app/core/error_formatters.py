"""
Error formatters for standardizing error responses to match provider formats.
"""

from typing import Dict, Any
from fastapi import HTTPException


def format_openai_error(
    status_code: int, message: str, error_type: str = None
) -> Dict[str, Any]:
    """
    Format an error response in OpenAI API format.

    Args:
        status_code: HTTP status code
        message: Error message
        error_type: Optional error type

    Returns:
        OpenAI-formatted error response
    """
    error_code_map = {
        400: "invalid_request_error",
        401: "authentication_error",
        403: "permission_error",
        404: "not_found_error",
        429: "rate_limit_error",
        500: "internal_server_error",
        502: "bad_gateway_error",
        503: "service_unavailable_error",
        504: "gateway_timeout_error",
    }

    error_type = error_type or error_code_map.get(status_code, "api_error")

    return {
        "error": {
            "message": message,
            "type": error_type,
            "code": error_code_map.get(status_code, "unknown_error"),
        }
    }


def format_anthropic_error(
    status_code: int, message: str, error_type: str = None
) -> Dict[str, Any]:
    """
    Format an error response in Anthropic API format.

    Args:
        status_code: HTTP status code
        message: Error message
        error_type: Optional error type

    Returns:
        Anthropic-formatted error response
    """
    error_type_map = {
        400: "invalid_request_error",
        401: "authentication_error",
        403: "permission_error",
        404: "not_found_error",
        429: "rate_limit_error",
        500: "internal_server_error",
        502: "bad_gateway_error",
        503: "service_unavailable_error",
        504: "gateway_timeout_error",
    }

    error_type = error_type or error_type_map.get(status_code, "api_error")

    return {"error": {"message": message, "type": error_type}}


def create_provider_error_response(
    provider_name: str, status_code: int, message: str, error_type: str = None
) -> HTTPException:
    """
    Create an HTTPException with provider-formatted error response.

    Args:
        provider_name: Provider name ("openai" or "anthropic")
        status_code: HTTP status code
        message: Error message
        error_type: Optional error type

    Returns:
        HTTPException with formatted error
    """
    if provider_name == "openai":
        error_detail = format_openai_error(status_code, message, error_type)
    elif provider_name == "anthropic":
        error_detail = format_anthropic_error(status_code, message, error_type)
    else:
        # Default format
        error_detail = {
            "error": {"message": message, "type": error_type or "api_error"}
        }

    return HTTPException(status_code=status_code, detail=error_detail)
