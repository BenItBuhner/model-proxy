"""
OpenAI provider implementation.
Handles OpenAI API calls with key rotation and error handling.
Uses provider configuration for endpoints, authentication, and settings.
Enhanced with route configuration support for fallback routing.
"""

import json
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx

from app.core.api_key_manager import get_available_keys
from app.core.provider_config import (
    get_provider_auth_headers,
    get_provider_config,
)
from app.providers.base import BaseProvider


class OpenAIProvider(BaseProvider):
    """Provider for OpenAI-compatible APIs (OpenAI, Nahcrof, Groq, etc.)."""

    def __init__(self, provider_name: str = "openai"):
        super().__init__(provider_name)
        self.config = get_provider_config(provider_name)
        if not self.config:
            raise ValueError(f"{provider_name} provider config not found")

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
        temperature: Optional[float] = 1.0,
        top_p: Optional[float] = 1.0,
        n: Optional[int] = 1,
        stream: Optional[bool] = False,
        stop: Optional[List[str]] = None,
        max_tokens: Optional[int] = None,
        presence_penalty: Optional[float] = 0.0,
        frequency_penalty: Optional[float] = 0.0,
        logit_bias: Optional[Dict[str, float]] = None,
        user: Optional[str] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Make a synchronous API call to OpenAI.

        Args:
            model: Model name
            messages: List of message dicts
            **kwargs: Additional OpenAI parameters

        Returns:
            OpenAI API response

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
                auth_headers = get_provider_auth_headers(self.provider_name, api_key)

                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    payload = {
                        "model": model,
                        "messages": messages,
                        "temperature": temperature,
                        "top_p": top_p,
                        "n": n,
                        "stream": stream,
                    }

                    if stop is not None:
                        payload["stop"] = stop
                    if max_tokens is not None:
                        payload["max_tokens"] = max_tokens
                    if presence_penalty is not None:
                        payload["presence_penalty"] = presence_penalty
                    if frequency_penalty is not None:
                        payload["frequency_penalty"] = frequency_penalty
                    if logit_bias is not None:
                        payload["logit_bias"] = logit_bias
                    if user is not None:
                        payload["user"] = user
                    if tools is not None:
                        payload["tools"] = tools
                    if tool_choice is not None:
                        payload["tool_choice"] = tool_choice

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
                        # Mark key as failed and try next one
                        self._mark_key_failed(api_key)
                        retry_count += 1
                        error_msg = (
                            f"OpenAI API error {response.status_code}: {response.text}"
                        )
                        last_error = Exception(error_msg)
                        # Attach status code for fallback decision
                        last_error.status = response.status_code
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
        raise last_error or Exception("All OpenAI API keys failed")

    async def call_stream(
        self,
        model: str,
        messages: list,
        temperature: Optional[float] = 1.0,
        top_p: Optional[float] = 1.0,
        n: Optional[int] = 1,
        stop: Optional[List[str]] = None,
        max_tokens: Optional[int] = None,
        presence_penalty: Optional[float] = 0.0,
        frequency_penalty: Optional[float] = 0.0,
        logit_bias: Optional[Dict[str, float]] = None,
        user: Optional[str] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """
        Make a streaming API call to OpenAI.

        Args:
            model: Model name
            messages: List of message dicts
            **kwargs: Additional OpenAI parameters

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
                auth_headers = get_provider_auth_headers(self.provider_name, api_key)

                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    payload = {
                        "model": model,
                        "messages": messages,
                        "temperature": temperature,
                        "top_p": top_p,
                        "n": n,
                        "stream": True,
                    }

                    if stop is not None:
                        payload["stop"] = stop
                    if max_tokens is not None:
                        payload["max_tokens"] = max_tokens
                    if presence_penalty is not None:
                        payload["presence_penalty"] = presence_penalty
                    if frequency_penalty is not None:
                        payload["frequency_penalty"] = frequency_penalty
                    if logit_bias is not None:
                        payload["logit_bias"] = logit_bias
                    if user is not None:
                        payload["user"] = user
                    if tools is not None:
                        payload["tools"] = tools
                    if tool_choice is not None:
                        payload["tool_choice"] = tool_choice

                    headers = {
                        **auth_headers,
                        "Content-Type": "application/json",
                        "Accept": "text/event-stream",
                        "Connection": "keep-alive",
                        "Cache-Control": "no-cache",
                    }

                    try:
                        async with client.stream(
                            "POST", endpoint_url, headers=headers, json=payload
                        ) as response:
                            if response.status_code >= 400:
                                error_text = await response.aread()
                                self._mark_key_failed(api_key)
                                retry_count += 1
                                error_msg = f"OpenAI API error {response.status_code}: {error_text.decode()}"
                                last_error = Exception(error_msg)
                                last_error.status = response.status_code
                                continue

                            # If upstream didn't return SSE, fall back by emitting a single chunk
                            content_type = (
                                response.headers.get("content-type") or ""
                            ).lower()
                            if "text/event-stream" not in content_type:
                                body = await response.aread()
                                try:
                                    data = json.loads(
                                        body.decode()
                                        if isinstance(body, (bytes, bytearray))
                                        else str(body)
                                    )
                                except Exception:
                                    data = {}
                                # Extract content
                                content = ""
                                try:
                                    choices = data.get("choices", [])
                                    if choices:
                                        msg = choices[0].get("message", {})
                                        content = msg.get("content") or ""
                                except Exception:
                                    content = ""
                                chunk = {
                                    "id": f"chatcmpl-{int(time.time())}",
                                    "object": "chat.completion.chunk",
                                    "created": int(time.time()),
                                    "model": model,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {
                                                "role": "assistant",
                                                "content": content,
                                            },
                                        }
                                    ],
                                }
                                yield f"data: {json.dumps(chunk)}\n\n"
                                yield "data: [DONE]\n\n"
                                return

                            async for line in response.aiter_lines():
                                if line:
                                    # Ensure SSE event separation with a blank line
                                    yield f"{line}\n\n"
                            return
                    except Exception as stream_err:
                        # Fallback: non-stream call, convert to a single streamed chunk
                        try:
                            fallback_payload = dict(payload)
                            fallback_payload["stream"] = False
                            resp = await client.post(
                                endpoint_url, headers=headers, json=fallback_payload
                            )
                            if resp.status_code >= 400:
                                error_text = await resp.aread()
                                self._mark_key_failed(api_key)
                                retry_count += 1
                                error_msg = f"OpenAI API error {resp.status_code}: {error_text.decode()}"
                                last_error = Exception(error_msg)
                                last_error.status = resp.status_code
                                continue
                            data = resp.json()
                            # Extract content safely
                            content = ""
                            try:
                                choices = data.get("choices", [])
                                if choices:
                                    delta_msg = choices[0].get("message", {})
                                    content = delta_msg.get("content") or ""
                            except Exception:
                                content = ""
                            # Emit one chunk and DONE
                            chunk = {
                                "id": f"chatcmpl-{int(time.time())}",
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {
                                            "role": "assistant",
                                            "content": content,
                                        },
                                    }
                                ],
                            }
                            yield f"data: {json.dumps(chunk)}\n\n"
                            yield "data: [DONE]\n\n"
                            return
                        except Exception as fb_err:
                            self._mark_key_failed(api_key)
                            retry_count += 1
                            last_error = fb_err
                            continue

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
        raise last_error or Exception("All OpenAI API keys failed")
