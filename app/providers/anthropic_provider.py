"""
Anthropic provider implementation.
Handles Anthropic API calls with key rotation and error handling.
Uses provider configuration for endpoints, authentication, and settings.
"""
import httpx
import json
from typing import Dict, Any, List, AsyncGenerator, Optional
from app.providers.base import BaseProvider
from app.core.api_key_manager import get_available_keys
from app.core.provider_config import get_provider_config, get_provider_endpoint, get_provider_auth_headers


class AnthropicProvider(BaseProvider):
    """Provider for Anthropic API."""
    
    def __init__(self):
        super().__init__("anthropic")
        self.config = get_provider_config("anthropic")
        if not self.config:
            raise ValueError("Anthropic provider config not found")
        
        # Get base URL and endpoint from config
        self.base_url = self.config["endpoints"]["base_url"]
        # Check for proxy override
        if self.config.get("proxy_support", {}).get("enabled", False):
            override_url = self.config["proxy_support"].get("base_url_override")
            if override_url:
                self.base_url = override_url
        
        self.completions_endpoint = self.config["endpoints"]["completions"]
        self.timeout = self.config.get("request_config", {}).get("timeout_seconds", 60)
    
    def _get_endpoint_url(self) -> str:
        """Get the full endpoint URL."""
        endpoint_path = self.completions_endpoint
        if endpoint_path.startswith("/"):
            endpoint_path = endpoint_path[1:]
        
        if self.base_url.endswith("/"):
            return f"{self.base_url}{endpoint_path}"
        else:
            return f"{self.base_url}/{endpoint_path}"
    
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
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make a synchronous API call to Anthropic.
        
        Args:
            model: Model name
            messages: List of message dicts
            max_tokens: Maximum tokens to generate
            **kwargs: Additional Anthropic parameters
            
        Returns:
            Anthropic API response
            
        Raises:
            Exception: If all API keys fail
        """
        available_keys = get_available_keys(self.provider_name)
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
                    
                    headers = {
                    **auth_headers,
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                    "Connection": "keep-alive",
                    "Cache-Control": "no-cache",
                    }
                    
                    response = await client.post(
                        endpoint_url,
                        headers=headers,
                        json=payload
                    )
                    
                    # Check for errors
                    if response.status_code >= 400:
                        self._mark_key_failed(api_key)
                        retry_count += 1
                        last_error = Exception(
                            f"Anthropic API error {response.status_code}: {response.text}"
                        )
                        continue
                    
                    return response.json()
                    
            except httpx.TimeoutException as e:
                self._mark_key_failed(api_key)
                retry_count += 1
                last_error = e
                continue
            except Exception as e:
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
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Make a streaming API call to Anthropic.
        
        Args:
            model: Model name
            messages: List of message dicts
            max_tokens: Maximum tokens to generate
            **kwargs: Additional Anthropic parameters
            
        Yields:
            SSE formatted chunks
            
        Raises:
            Exception: If all API keys fail
        """
        available_keys = get_available_keys(self.provider_name)
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
                    
                    headers = {
                        **auth_headers,
                        "Content-Type": "application/json"
                    }
                    
                    async with client.stream(
                        "POST",
                        endpoint_url,
                        headers=headers,
                        json=payload
                    ) as response:
                        if response.status_code >= 400:
                            error_text = await response.aread()
                            self._mark_key_failed(api_key)
                            retry_count += 1
                            last_error = Exception(
                                f"Anthropic API error {response.status_code}: {error_text.decode()}"
                            )
                            continue
                        
                        async for line in response.aiter_lines():
                            if line:
                                # Ensure proper SSE event separation for SDKs
                                yield f"{line}\n\n"
                        
                        return  # Success, exit retry loop
                        
            except httpx.TimeoutException as e:
                self._mark_key_failed(api_key)
                retry_count += 1
                last_error = e
                continue
            except Exception as e:
                self._mark_key_failed(api_key)
                retry_count += 1
                last_error = e
                continue
        
        # All keys failed
        raise last_error or Exception("All Anthropic API keys failed")


