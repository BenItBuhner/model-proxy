"""
Comprehensive tests for authentication edge cases.
"""
import pytest
from fastapi import HTTPException
from app.auth import verify_client_api_key, CLIENT_API_KEY


@pytest.mark.asyncio
async def test_verify_client_api_key_with_whitespace():
    """Test API key with whitespace handling."""
    result = await verify_client_api_key(authorization=f"Bearer  {CLIENT_API_KEY}  ")
    assert result is True


@pytest.mark.asyncio
async def test_verify_client_api_key_case_sensitive():
    """Test that API key is case-sensitive."""
    if CLIENT_API_KEY:
        modified_key = CLIENT_API_KEY.swapcase()
        if modified_key != CLIENT_API_KEY:
            with pytest.raises(HTTPException) as exc_info:
                await verify_client_api_key(authorization=f"Bearer {modified_key}")
            assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_verify_client_api_key_partial_match():
    """Test that partial key match fails."""
    if CLIENT_API_KEY and len(CLIENT_API_KEY) > 1:
        partial_key = CLIENT_API_KEY[:-1]
        with pytest.raises(HTTPException) as exc_info:
            await verify_client_api_key(authorization=f"Bearer {partial_key}")
        assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_verify_client_api_key_extra_chars():
    """Test that key with extra characters fails."""
    with pytest.raises(HTTPException) as exc_info:
        await verify_client_api_key(authorization=f"Bearer {CLIENT_API_KEY}extra")
    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_verify_client_api_key_bearer_lowercase():
    """Test Bearer prefix case insensitivity."""
    result = await verify_client_api_key(authorization=f"bearer {CLIENT_API_KEY}")
    assert result is True


@pytest.mark.asyncio
async def test_verify_client_api_key_multiple_bearer():
    """Test handling of multiple Bearer prefixes."""
    result = await verify_client_api_key(authorization=f"Bearer Bearer {CLIENT_API_KEY}")
    assert result is True  # Should strip all Bearer prefixes

