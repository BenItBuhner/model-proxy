"""
OpenAI provider implementation.
Handles OpenAI API calls with key rotation and error handling.
Uses provider configuration for endpoints, authentication, and settings.
Enhanced with route configuration support for fallback routing.
"""

import json
import logging
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx

from app.core.api_key_manager import get_available_keys
from app.core.provider_config import (
    get_provider_auth_headers,
    get_provider_config,
)
from app.providers.base import BaseProvider

logger = logging.getLogger("openai_provider")


class ProviderAPIError(Exception):
    """Provider request failed with an HTTP status code (used for fallback decisions)."""

    def __init__(self, message: str, status: int, body: Optional[str] = None):
        super().__init__(message)
        self.status = status
        self.body = body


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
                    # Build payload - Some providers have limited parameter support
                    is_gemini = self.provider_name == "gemini"
                    is_cerebras = self.provider_name == "cerebras"

                    # Sanitize messages for Gemini - remove unsupported fields
                    sanitized_messages = messages
                    if is_gemini:
                        sanitized_messages = []
                        for msg in messages:
                            if isinstance(msg, dict):
                                # Only keep core message fields that Gemini supports
                                clean_msg = {
                                    "role": msg.get("role"),
                                    "content": msg.get("content"),
                                }
                                # Include tool_calls if present (Gemini does support this)
                                if msg.get("tool_calls"):
                                    clean_msg["tool_calls"] = msg.get("tool_calls")
                                # Include tool_call_id for tool responses
                                if msg.get("tool_call_id"):
                                    clean_msg["tool_call_id"] = msg.get("tool_call_id")
                                # Include name if present
                                if msg.get("name"):
                                    clean_msg["name"] = msg.get("name")
                                sanitized_messages.append(clean_msg)
                            else:
                                sanitized_messages.append(msg)

                    payload = {
                        "model": model,
                        "messages": sanitized_messages,
                        "stream": stream,
                    }

                    # Only include temperature if not default or not gemini
                    if temperature is not None and (
                        not is_gemini or temperature != 1.0
                    ):
                        payload["temperature"] = temperature

                    # Only include top_p if not default or not gemini
                    if top_p is not None and (not is_gemini or top_p != 1.0):
                        payload["top_p"] = top_p

                    # Gemini and Cerebras don't support n parameter
                    if n is not None and not is_gemini and not is_cerebras:
                        payload["n"] = n

                    # Gemini doesn't support stop sequences well, but Cerebras does
                    if stop is not None and not is_gemini:
                        payload["stop"] = stop

                    # Cerebras uses max_completion_tokens instead of max_tokens
                    if max_tokens is not None:
                        if is_cerebras:
                            payload["max_completion_tokens"] = max_tokens
                        else:
                            payload["max_tokens"] = max_tokens

                    # Cerebras and Gemini don't support presence_penalty
                    if (
                        presence_penalty is not None
                        and not is_gemini
                        and not is_cerebras
                    ):
                        payload["presence_penalty"] = presence_penalty

                    # Cerebras and Gemini don't support logit_bias
                    if logit_bias is not None and not is_gemini and not is_cerebras:
                        payload["logit_bias"] = logit_bias

                    # Gemini doesn't support user parameter, but Cerebras does
                    if user is not None and not is_gemini:
                        payload["user"] = user
                    if tools is not None:
                        payload["tools"] = tools
                    if tool_choice is not None:
                        payload["tool_choice"] = tool_choice

                    # Debug log the payload (without full message content)
                    debug_payload = {
                        k: v for k, v in payload.items() if k != "messages"
                    }
                    debug_payload["message_count"] = len(messages)
                    logger.debug(
                        f"Upstream request to {self.provider_name}: {debug_payload}"
                    )

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
                        # If the router injected a specific key for this route,
                        # do NOT mark it failed here. The fallback router owns
                        # cooldown decisions (including fallback_no_cooldown).
                        if not self._route_api_key:
                            self._mark_key_failed(api_key)
                        retry_count += 1
                        # Log detailed error for debugging
                        logger.error(
                            f"Provider {self.provider_name} returned {response.status_code}: {response.text[:500]}"
                        )
                        logger.debug(
                            f"Failed request payload: model={model}, endpoint={endpoint_url}"
                        )
                        error_msg = (
                            f"OpenAI API error {response.status_code}: {response.text}"
                        )
                        last_error = ProviderAPIError(
                            error_msg, status=response.status_code, body=response.text
                        )
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

                # Build payload - Some providers have limited parameter support
                is_gemini = self.provider_name == "gemini"
                is_cerebras = self.provider_name == "cerebras"

                # Sanitize messages for Gemini - remove unsupported fields
                sanitized_messages = messages
                if is_gemini:
                    sanitized_messages = []
                    for msg in messages:
                        if isinstance(msg, dict):
                            # Only keep core message fields that Gemini supports
                            clean_msg = {
                                "role": msg.get("role"),
                                "content": msg.get("content"),
                            }
                            # Include tool_calls if present (Gemini does support this)
                            if msg.get("tool_calls"):
                                clean_msg["tool_calls"] = msg.get("tool_calls")
                            # Include tool_call_id for tool responses
                            if msg.get("tool_call_id"):
                                clean_msg["tool_call_id"] = msg.get("tool_call_id")
                            # Include name if present
                            if msg.get("name"):
                                clean_msg["name"] = msg.get("name")
                            sanitized_messages.append(clean_msg)
                        else:
                            sanitized_messages.append(msg)

                payload = {
                    "model": model,
                    "messages": sanitized_messages,
                    "stream": True,
                }

                # Only include temperature if not default or not gemini
                if temperature is not None and (not is_gemini or temperature != 1.0):
                    payload["temperature"] = temperature

                # Only include top_p if not default or not gemini
                if top_p is not None and (not is_gemini or top_p != 1.0):
                    payload["top_p"] = top_p

                # Gemini and Cerebras don't support n parameter
                if n is not None and not is_gemini and not is_cerebras:
                    payload["n"] = n

                # Gemini doesn't support stop sequences well, but Cerebras does
                if stop is not None and not is_gemini:
                    payload["stop"] = stop

                # Cerebras uses max_completion_tokens instead of max_tokens
                if max_tokens is not None:
                    if is_cerebras:
                        payload["max_completion_tokens"] = max_tokens
                    else:
                        payload["max_tokens"] = max_tokens

                # Cerebras and Gemini don't support presence_penalty
                if presence_penalty is not None and not is_gemini and not is_cerebras:
                    payload["presence_penalty"] = presence_penalty

                # Cerebras and Gemini don't support logit_bias
                if logit_bias is not None and not is_gemini and not is_cerebras:
                    payload["logit_bias"] = logit_bias

                # Gemini doesn't support user parameter, but Cerebras does
                if user is not None and not is_gemini:
                    payload["user"] = user
                if tools is not None:
                    payload["tools"] = tools
                if tool_choice is not None:
                    payload["tool_choice"] = tool_choice

                # Debug log the payload (without full message content)
                debug_payload = {k: v for k, v in payload.items() if k != "messages"}
                debug_payload["message_count"] = len(messages)
                logger.debug(
                    f"Upstream request to {self.provider_name}: {debug_payload}"
                )

                headers = {
                    **auth_headers,
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                    "Connection": "keep-alive",
                    "Cache-Control": "no-cache",
                }

                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    try:
                        async with client.stream(
                            "POST", endpoint_url, headers=headers, json=payload
                        ) as response:
                            if response.status_code >= 400:
                                error_text = await response.aread()
                                if not self._route_api_key:
                                    self._mark_key_failed(api_key)
                                retry_count += 1
                                body = error_text.decode(errors="replace")
                                # Log detailed error for debugging
                                logger.error(
                                    f"Provider {self.provider_name} returned {response.status_code}: {body[:500]}"
                                )
                                logger.debug(
                                    f"Failed request payload: model={model}, endpoint={endpoint_url}"
                                )
                                error_msg = (
                                    f"OpenAI API error {response.status_code}: {body}"
                                )
                                last_error = ProviderAPIError(
                                    error_msg, status=response.status_code, body=body
                                )
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
                                # Extract content and tool_calls
                                content = ""
                                tool_calls = None
                                try:
                                    choices = data.get("choices", [])
                                    if choices:
                                        msg = choices[0].get("message", {})
                                        content = msg.get("content") or ""
                                        tool_calls = msg.get("tool_calls")
                                except Exception:
                                    content = ""
                                delta = {"role": "assistant", "content": content}
                                if tool_calls is not None:
                                    delta["tool_calls"] = tool_calls
                                chunk = {
                                    "id": f"chatcmpl-{int(time.time())}",
                                    "object": "chat.completion.chunk",
                                    "created": int(time.time()),
                                    "model": model,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": delta,
                                        }
                                    ],
                                }
                                yield f"data: {json.dumps(chunk)}\n\n"
                                yield "data: [DONE]\n\n"
                                return

                            async for line in response.aiter_lines():
                                if line:
                                    # Log raw SSE line for debugging
                                    logger.debug(
                                        f"SSE line from {self.provider_name}: {line[:200] if len(line) > 200 else line}"
                                    )

                                    # Skip empty data or malformed lines
                                    if line.strip() == "data:" or line.strip() == "":
                                        continue

                                    # Handle [DONE] marker
                                    if line.strip() == "data: [DONE]":
                                        yield "data: [DONE]\n\n"
                                        continue

                                    # Validate JSON in data lines before forwarding
                                    if line.startswith("data:"):
                                        json_str = line[5:].strip()
                                        if json_str and json_str != "[DONE]":
                                            try:
                                                # Parse to validate, then re-serialize to ensure clean JSON
                                                parsed = json.loads(json_str)

                                                # Check for error responses embedded in the stream
                                                if "error" in parsed:
                                                    error_info = parsed.get("error", {})
                                                    error_msg = error_info.get(
                                                        "message", str(parsed)
                                                    )
                                                    logger.error(
                                                        f"Error in SSE stream from {self.provider_name}: {error_msg}"
                                                    )
                                                    # Raise to trigger fallback
                                                    raise ProviderAPIError(
                                                        f"Stream error from {self.provider_name}: {error_msg}",
                                                        status=error_info.get(
                                                            "code", 500
                                                        ),
                                                        body=json_str,
                                                    )

                                                yield f"data: {json.dumps(parsed)}\n\n"
                                            except json.JSONDecodeError as e:
                                                # Check if raw line contains error indicators
                                                if (
                                                    "error" in json_str.lower()
                                                    or "ResponseStreamResult"
                                                    in json_str
                                                ):
                                                    logger.error(
                                                        f"Error response from {self.provider_name}: {json_str[:500]}"
                                                    )
                                                    raise ProviderAPIError(
                                                        f"Stream error from {self.provider_name}: {json_str[:200]}",
                                                        status=500,
                                                        body=json_str,
                                                    )
                                                logger.warning(
                                                    f"Malformed JSON from {self.provider_name}, skipping: {e} - data: {json_str[:100]}"
                                                )
                                                continue
                                    else:
                                        # Check for error text in non-data lines
                                        if (
                                            "error" in line.lower()
                                            or "ResponseStreamResult" in line
                                        ):
                                            logger.error(
                                                f"Error in SSE from {self.provider_name}: {line[:500]}"
                                            )
                                            raise ProviderAPIError(
                                                f"Stream error from {self.provider_name}: {line[:200]}",
                                                status=500,
                                                body=line,
                                            )
                                        # Non-data SSE line (like event:, id:, retry:), forward as-is
                                        yield f"{line}\n\n"
                            return

                    except httpx.RequestError:
                        # Fallback: non-stream call, convert to a single streamed chunk
                        try:
                            fallback_payload = dict(payload)
                            fallback_payload["stream"] = False
                            resp = await client.post(
                                endpoint_url, headers=headers, json=fallback_payload
                            )
                            if resp.status_code >= 400:
                                if not self._route_api_key:
                                    self._mark_key_failed(api_key)
                                retry_count += 1
                                body = resp.text
                                error_msg = (
                                    f"OpenAI API error {resp.status_code}: {body}"
                                )
                                last_error = ProviderAPIError(
                                    error_msg, status=resp.status_code, body=body
                                )
                                continue
                            data = resp.json()
                            # Extract content and tool_calls safely
                            content = ""
                            tool_calls = None
                            try:
                                choices = data.get("choices", [])
                                if choices:
                                    delta_msg = choices[0].get("message", {})
                                    content = delta_msg.get("content") or ""
                                    tool_calls = delta_msg.get("tool_calls")
                            except Exception:
                                content = ""
                            delta = {"role": "assistant", "content": content}
                            if tool_calls is not None:
                                delta["tool_calls"] = tool_calls
                            # Emit one chunk and DONE
                            chunk = {
                                "id": f"chatcmpl-{int(time.time())}",
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": delta,
                                    }
                                ],
                            }
                            yield f"data: {json.dumps(chunk)}\n\n"
                            yield "data: [DONE]\n\n"
                            return
                        except Exception as fb_err:
                            if not self._route_api_key:
                                self._mark_key_failed(api_key)
                            retry_count += 1
                            last_error = fb_err
                            continue

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
        raise last_error or Exception("All OpenAI API keys failed")
