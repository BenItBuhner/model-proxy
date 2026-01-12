"""
Azure provider implementation using azure-ai-inference SDK.
Handles Azure AI and GitHub Models providers.
Enhanced with route configuration support for fallback routing.
"""

import time
from typing import Any, AsyncGenerator, Dict, List, Optional

from app.core.api_key_manager import get_available_keys
from app.core.provider_config import get_provider_config
from app.providers.base import BaseProvider

try:
    from azure.ai.inference import ChatCompletionsClient
    from azure.ai.inference.models import AssistantMessage, SystemMessage, UserMessage
    from azure.core.credentials import AzureKeyCredential

    AZURE_SDK_AVAILABLE = True
except ImportError:
    AZURE_SDK_AVAILABLE = False


class AzureProvider(BaseProvider):
    """Provider for Azure AI inference (used by GitHub Models and Azure)."""

    def __init__(self, provider_name: str):
        super().__init__(provider_name)
        if not AZURE_SDK_AVAILABLE:
            raise ImportError(
                "azure-ai-inference package is required for Azure provider"
            )

        self.config = get_provider_config(provider_name)
        if not self.config:
            raise ValueError(f"Azure provider config not found for {provider_name}")

        # Get endpoint from config
        self._default_base_url = self.config["endpoints"]["base_url"]

        # Check for proxy override
        if self.config.get("proxy_support", {}).get("enabled", False):
            override_url = self.config["proxy_support"].get("base_url_override")
            if override_url:
                self._default_base_url = override_url

        self.timeout = self.config.get("request_config", {}).get("timeout_seconds", 60)

    @property
    def base_url(self) -> str:
        """Get the effective base URL (route-specific or default)."""
        return self._get_effective_base_url(self._default_base_url)

    @property
    def endpoint(self) -> str:
        """Alias for base_url for backward compatibility."""
        return self.base_url

    def _get_available_api_keys(self) -> List[str]:
        """
        Get list of API keys to try.

        If route-specific API key is set, return only that key.
        Otherwise, return all available keys from environment.
        """
        if self._route_api_key:
            return [self._route_api_key]
        return get_available_keys(self.provider_name)

    def _get_client(self, api_key: str) -> "ChatCompletionsClient":
        """Create Azure AI client with API key."""
        return ChatCompletionsClient(
            endpoint=self.endpoint,
            credential=AzureKeyCredential(api_key),
            timeout=self.timeout,
        )

    def _convert_messages_to_azure(self, messages: List[Dict[str, Any]]) -> List:
        """Convert OpenAI format messages to Azure format."""
        azure_messages = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")

            if role == "system":
                azure_messages.append(SystemMessage(content=content))
            elif role == "user":
                azure_messages.append(UserMessage(content=content))
            elif role == "assistant":
                azure_messages.append(AssistantMessage(content=content))
            else:
                # Default unknown roles to user
                azure_messages.append(UserMessage(content=content))

        return azure_messages

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
        Make a synchronous API call to Azure AI.

        Args:
            model: Model name
            messages: List of message dicts
            temperature: Sampling temperature
            top_p: Top-p sampling
            n: Number of completions
            stream: Whether to stream (should be False for this method)
            stop: Stop sequences
            max_tokens: Maximum tokens to generate
            presence_penalty: Presence penalty
            frequency_penalty: Frequency penalty
            logit_bias: Logit bias
            user: User identifier
            tools: Tool definitions
            tool_choice: Tool choice
            **kwargs: Additional parameters

        Returns:
            OpenAI-compatible response dict

        Raises:
            Exception: If all API keys fail
        """
        available_keys = self._get_available_api_keys()
        if not available_keys:
            raise Exception(
                f"No {self.provider_name} API keys available (all may be in cooldown)"
            )

        last_error = None
        retry_count = 0

        # Try each available key until one succeeds
        for api_key in available_keys:
            try:
                client = self._get_client(api_key)
                azure_messages = self._convert_messages_to_azure(messages)

                # Build parameters
                params = {
                    "model": model,
                    "messages": azure_messages,
                }

                if max_tokens is not None:
                    params["max_tokens"] = max_tokens
                if temperature is not None:
                    params["temperature"] = temperature
                if top_p is not None:
                    params["top_p"] = top_p
                if stop is not None and len(stop) > 0:
                    params["stop"] = stop
                if user is not None:
                    params["user"] = user

                # Azure SDK doesn't support all OpenAI parameters directly
                # We'll pass through what we can

                response = client.complete(**params)

                # Convert Azure response to OpenAI format
                openai_response = self._convert_azure_to_openai(response, model)
                return openai_response

            except Exception as e:
                # If the router injected a specific key for this route,
                # do NOT mark it failed here. The fallback router owns
                # cooldown decisions (including fallback_no_cooldown).
                if not self._route_api_key:
                    self._mark_key_failed(api_key)
                retry_count += 1
                last_error = e
                # Try to attach status code if available
                if hasattr(e, "status_code"):
                    last_error.status = e.status_code
                continue

        # All keys failed
        raise last_error or Exception(f"All {self.provider_name} API keys failed")

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
        Make a streaming API call to Azure AI.

        Args:
            model: Model name
            messages: List of message dicts
            temperature: Sampling temperature
            top_p: Top-p sampling
            n: Number of completions
            stop: Stop sequences
            max_tokens: Maximum tokens to generate
            presence_penalty: Presence penalty
            frequency_penalty: Frequency penalty
            logit_bias: Logit bias
            user: User identifier
            tools: Tool definitions
            tool_choice: Tool choice
            **kwargs: Additional parameters

        Yields:
            SSE formatted chunks

        Raises:
            Exception: If all API keys fail
        """
        available_keys = self._get_available_api_keys()
        if not available_keys:
            raise Exception(
                f"No {self.provider_name} API keys available (all may be in cooldown)"
            )

        last_error = None
        retry_count = 0

        # Try each available key until one succeeds
        for api_key in available_keys:
            try:
                client = self._get_client(api_key)
                azure_messages = self._convert_messages_to_azure(messages)

                # Build parameters
                params = {
                    "model": model,
                    "messages": azure_messages,
                    "stream": True,  # Enable streaming
                }

                if max_tokens is not None:
                    params["max_tokens"] = max_tokens
                if temperature is not None:
                    params["temperature"] = temperature
                if top_p is not None:
                    params["top_p"] = top_p
                if stop is not None and len(stop) > 0:
                    params["stop"] = stop
                if user is not None:
                    params["user"] = user

                # Azure SDK streaming - attempt to use streaming if available
                try:
                    response_stream = client.complete(**params)

                    # Check if response is iterable (streaming)
                    if hasattr(response_stream, "__iter__"):
                        for chunk in response_stream:
                            if hasattr(chunk, "choices") and chunk.choices:
                                choice = chunk.choices[0]
                                content = ""
                                if hasattr(choice, "delta") and choice.delta:
                                    content = getattr(choice.delta, "content", "") or ""
                                elif hasattr(choice, "message") and choice.message:
                                    content = (
                                        getattr(choice.message, "content", "") or ""
                                    )

                                chunk_data = {
                                    "id": f"chatcmpl-{model.replace('/', '-')}",
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
                                            "finish_reason": None,
                                        }
                                    ],
                                }
                                import json

                                yield f"data: {json.dumps(chunk_data)}\n\n"
                        yield "data: [DONE]\n\n"
                        return
                    else:
                        # Non-streaming response, convert to single chunk
                        content = ""
                        if (
                            hasattr(response_stream, "choices")
                            and response_stream.choices
                        ):
                            content = response_stream.choices[0].message.content or ""

                        chunk_data = {
                            "id": f"chatcmpl-{model.replace('/', '-')}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"role": "assistant", "content": content},
                                    "finish_reason": "stop",
                                }
                            ],
                        }
                        import json

                        yield f"data: {json.dumps(chunk_data)}\n\n"
                        yield "data: [DONE]\n\n"
                        return

                except Exception:
                    # Fallback: non-streaming call, convert to simulated stream
                    params_non_stream = dict(params)
                    params_non_stream["stream"] = False
                    response = client.complete(**params_non_stream)

                    content = ""
                    if hasattr(response, "choices") and response.choices:
                        content = response.choices[0].message.content or ""

                    chunk_data = {
                        "id": f"chatcmpl-{model.replace('/', '-')}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"role": "assistant", "content": content},
                                "finish_reason": "stop",
                            }
                        ],
                    }
                    import json

                    yield f"data: {json.dumps(chunk_data)}\n\n"
                    yield "data: [DONE]\n\n"
                    return

            except Exception as e:
                if not self._route_api_key:
                    self._mark_key_failed(api_key)
                retry_count += 1
                last_error = e
                if hasattr(e, "status_code"):
                    last_error.status = e.status_code
                continue

        # All keys failed
        raise last_error or Exception(f"All {self.provider_name} API keys failed")

    def _convert_azure_to_openai(self, azure_response, model: str) -> Dict[str, Any]:
        """Convert Azure response to OpenAI format."""
        content = ""
        if hasattr(azure_response, "choices") and azure_response.choices:
            choice = azure_response.choices[0]
            if hasattr(choice, "message") and choice.message:
                content = getattr(choice.message, "content", "") or ""

        return {
            "id": f"chatcmpl-{model.replace('/', '-')}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": (
                    getattr(azure_response.usage, "prompt_tokens", 0)
                    if hasattr(azure_response, "usage")
                    else 0
                ),
                "completion_tokens": (
                    getattr(azure_response.usage, "completion_tokens", 0)
                    if hasattr(azure_response, "usage")
                    else 0
                ),
                "total_tokens": (
                    getattr(azure_response.usage, "total_tokens", 0)
                    if hasattr(azure_response, "usage")
                    else 0
                ),
            },
        }
