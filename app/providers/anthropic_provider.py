"""
Anthropic provider implementation.
Handles Anthropic API calls with key rotation and error handling.
Uses provider configuration for endpoints, authentication, and settings.
Enhanced with route configuration support for fallback routing.
"""

from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx

from app.core.api_key_manager import get_available_keys
from app.core.provider_config import (
    get_provider_auth_headers,
    get_provider_config,
)
from app.providers.base import BaseProvider


class AnthropicProvider(BaseProvider):
    """Provider for Anthropic API."""

    def __init__(self):
        super().__init__("anthropic")
        self.config = get_provider_config("anthropic")
        if not self.config:
            raise ValueError("Anthropic provider config not found")

        # Get base URL and endpoint from config
        self._default_base_url = self.config["endpoints"]["base_url"]
        # Check for proxy override
        if self.config.get("proxy_support", {}).get("enabled", False):
            override_url = self.config["proxy_support"].get("base_url_override")
            if override_url:
                self._default_base_url = override_url

        self.completions_endpoint = self.config["endpoints"]["completions"]
        self.timeout = self.config.get("request_config", {}).get("timeout_seconds", 60)

    @property
    def base_url(self) -> str:
        """Get the effective base URL (route-specific or default)."""
        return self._get_effective_base_url(self._default_base_url)

    def _get_endpoint_url(self) -> str:
        """Get the full endpoint URL."""
        endpoint_path = self.completions_endpoint
        if endpoint_path.startswith("/"):
            endpoint_path = endpoint_path[1:]

        base = self.base_url
        if base.endswith("/"):
            return f"{base}{endpoint_path}"
        else:
            return f"{base}/{endpoint_path}"

    def _get_available_api_keys(self) -> List[str]:
        """
        Get list of API keys to try.

        If route-specific API key is set, return only that key.
        Otherwise, return all available keys from environment.
        """
        if self._route_api_key:
            return [self._route_api_key]
        return get_available_keys(self.provider_name)

    async def call(
        self,
        model: str,
        messages: list,
        max_tokens: int,
        stream: Optional[bool] = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        tools: Optional[List[Dict]] = None,
        system: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Make a synchronous API call to Anthropic.

        Args:
            model: Model name
            messages: List of message dicts
            max_tokens: Maximum tokens to generate
            stream: Whether to stream (should be False for this method)
            temperature: Sampling temperature
            top_p: Top-p sampling
            top_k: Top-k sampling
            tools: List of tool definitions
            system: System prompt
            **kwargs: Additional Anthropic parameters

        Returns:
            Anthropic API response

        Raises:
            Exception: If all API keys fail
        """
        available_keys = self._get_available_api_keys()
        if not available_keys:
            raise Exception(f"No {self.provider_name} API keys available")

        last_error = None
        retry_count = 0

        # Try each available key until one succeeds
        for api_key in available_keys:
            try:
                endpoint_url = self._get_endpoint_url()
                auth_headers = get_provider_auth_headers("anthropic", api_key)

                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    payload = {
                        "model": model,
                        "messages": messages,
                        "max_tokens": max_tokens,
                        "stream": stream,
                    }

                    if temperature is not None:
                        payload["temperature"] = temperature
                    if top_p is not None:
                        payload["top_p"] = top_p
                    if top_k is not None:
                        payload["top_k"] = top_k
                    if tools is not None:
                        payload["tools"] = tools
                    if system is not None:
                        payload["system"] = system

                    headers = {
                        **auth_headers,
                        "Content-Type": "application/json",
                        "Accept": "text/event-stream",
                        "Connection": "keep-alive",
                        "Cache-Control": "no-cache",
                    }

                    response = await client.post(
                        endpoint_url, headers=headers, json=payload
                    )

                    # Check for errors
                    if response.status_code >= 400:
                        # If the router injected a specific key for this route,
                        # do NOT mark it failed here. The fallback router owns
                        # cooldown decisions (including fallback_no_cooldown).
                        if not self._route_api_key:
                            self._mark_key_failed(api_key)
                        retry_count += 1
                        error_msg = f"Anthropic API error {response.status_code}: {response.text}"
                        last_error = Exception(error_msg)
                        # Attach status code for fallback decision
                        last_error.status = response.status_code
                        continue

                    return response.json()

            except httpx.TimeoutException as e:
                if not self._route_api_key:
                    self._mark_key_failed(api_key)
                retry_count += 1
                last_error = e
                continue
            except Exception as e:
                if not self._route_api_key:
                    self._mark_key_failed(api_key)
                retry_count += 1
                last_error = e
                continue

        # All keys failed
        raise last_error or Exception("All Anthropic API keys failed")

    async def call_stream(
        self,
        model: str,
        messages: list,
        max_tokens: int,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        tools: Optional[List[Dict]] = None,
        system: Optional[str] = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """
        Make a streaming API call to Anthropic.

        Args:
            model: Model name
            messages: List of message dicts
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            top_k: Top-k sampling
            tools: List of tool definitions
            system: System prompt
            **kwargs: Additional Anthropic parameters

        Yields:
            SSE formatted chunks

        Raises:
            Exception: If all API keys fail
        """
        available_keys = self._get_available_api_keys()
        if not available_keys:
            raise Exception(f"No {self.provider_name} API keys available")

        last_error = None
        retry_count = 0

        # Try each available key until one succeeds
        for api_key in available_keys:
            try:
                endpoint_url = self._get_endpoint_url()
                auth_headers = get_provider_auth_headers("anthropic", api_key)

                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    payload = {
                        "model": model,
                        "messages": messages,
                        "max_tokens": max_tokens,
                        "stream": True,
                    }

                    if temperature is not None:
                        payload["temperature"] = temperature
                    if top_p is not None:
                        payload["top_p"] = top_p
                    if top_k is not None:
                        payload["top_k"] = top_k
                    if tools is not None:
                        payload["tools"] = tools
                    if system is not None:
                        payload["system"] = system

                    headers = {
                        **auth_headers,
                        "Content-Type": "application/json",
                    }

                    async with client.stream(
                        "POST", endpoint_url, headers=headers, json=payload
                    ) as response:
                        if response.status_code >= 400:
                            error_text = await response.aread()
                            if not self._route_api_key:
                                self._mark_key_failed(api_key)
                            retry_count += 1
                            error_msg = f"Anthropic API error {response.status_code}: {error_text.decode()}"
                            last_error = Exception(error_msg)
                            last_error.status = response.status_code
                            continue

                        async for line in response.aiter_lines():
                            if line:
                                # Ensure proper SSE event separation for SDKs
                                yield f"{line}\n\n"

                        return  # Success, exit retry loop

            except httpx.TimeoutException as e:
                if not self._route_api_key:
                    self._mark_key_failed(api_key)
                retry_count += 1
                last_error = e
                continue
            except Exception as e:
                if not self._route_api_key:
                    self._mark_key_failed(api_key)
                retry_count += 1
                last_error = e
                continue

        # All keys failed
        raise last_error or Exception("All Anthropic API keys failed")
