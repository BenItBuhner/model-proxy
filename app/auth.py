"""
Client API key authentication module.
Loads CLIENT_API_KEY from environment and validates incoming requests.
"""
import os
from fastapi import HTTPException, Header
from typing import Optional

# Load client API key from environment at module import time
CLIENT_API_KEY = os.getenv("CLIENT_API_KEY")

# Only raise error if CLIENT_API_KEY is required but not set
# This allows tests to set it via monkeypatch
if not CLIENT_API_KEY and os.getenv("REQUIRE_CLIENT_API_KEY", "false").lower() == "true":
    raise ValueError("CLIENT_API_KEY environment variable is required")


async def verify_client_api_key(
    authorization: Optional[str] = Header(None, alias="Authorization"),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
) -> bool:
    """
    FastAPI dependency to verify client API key.
    Expects Authorization header in format: "Bearer <api_key>" or just "<api_key>"
    
    Args:
        authorization: Authorization header value
        
    Returns:
        True if valid
        
    Raises:
        HTTPException: 401 if key is missing or invalid
    """
    # Accept either Authorization or X-API-Key for client authentication
    if not authorization and not x_api_key:
        raise HTTPException(status_code=401, detail="Missing API key")
    
    # Prefer Authorization if present, else X-API-Key
    if authorization:
        # Handle both "Bearer <key>" and "<key>" formats
        # Case-insensitive Bearer prefix, handle multiple "Bearer" prefixes
        api_key = authorization.strip()
        while api_key.lower().startswith("bearer "):
            api_key = api_key[7:].strip()
    else:
        api_key = (x_api_key or "").strip()
    
    if api_key != CLIENT_API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    
    return True

