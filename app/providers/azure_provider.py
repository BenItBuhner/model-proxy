"""
Azure provider implementation using azure-ai-inference SDK.
Handles Azure AI and GitHub Models providers.
"""
import os
from typing import Dict, Any, List, AsyncGenerator, Optional
from app.providers.base import BaseProvider
from app.core.api_key_manager import get_available_keys
from app.core.provider_config import get_provider_config, get_provider_auth_headers

try:
    from azure.ai.inference import ChatCompletionsClient
    from azure.ai.inference.models import SystemMessage, UserMessage
    from azure.core.credentials import AzureKeyCredential
    AZURE_SDK_AVAILABLE = True
except ImportError:
    AZURE_SDK_AVAILABLE = False


class AzureProvider(BaseProvider):
    """Provider for Azure AI inference (used by GitHub Models and Azure)."""

    def __init__(self, provider_name: str):
        super().__init__(provider_name)
        if not AZURE_SDK_AVAILABLE:
            raise ImportError("azure-ai-inference package is required for Azure provider")

        self.config = get_provider_config(provider_name)
        if not self.config:
            raise ValueError(f"Azure provider config not found for {provider_name}")

        # Get endpoint from config
        self.endpoint = self.config["endpoints"]["base_url"]

        # Check for proxy override
        if self.config.get("proxy_support", {}).get("enabled", False):
            override_url = self.config["proxy_support"].get("base_url_override")
            if override_url:
                self.endpoint = override_url

        self.timeout = self.config.get("request_config", {}).get("timeout_seconds", 60)

    def _get_client(self, api_key: str) -> ChatCompletionsClient:
        """Create Azure AI client with API key."""
        return ChatCompletionsClient(
            endpoint=self.endpoint,
            credential=AzureKeyCredential(api_key),
            timeout=self.timeout
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
                # For assistant messages, we need to handle the content properly
                azure_messages.append(UserMessage(content=content))  # Simplified for now
            else:
                # Skip unknown roles or convert to user
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
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make a synchronous API call to Azure AI.

        Args:
            model: Model name
            messages: List of message dicts
            **kwargs: Additional parameters

        Returns:
            OpenAI-compatible response dict

        Raises:
            Exception: If all API keys fail
        """
        available_keys = get_available_keys(self.provider_name)
        if not available_keys:
            raise Exception(f"No {self.provider_name} API keys available (all may be in cooldown)")

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
                self._mark_key_failed(api_key)
                retry_count += 1
                last_error = e
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
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Make a streaming API call to Azure AI.

        Args:
            model: Model name
            messages: List of message dicts
            **kwargs: Additional parameters

        Yields:
            SSE formatted chunks

        Raises:
            Exception: If all API keys fail
        """
        available_keys = get_available_keys(self.provider_name)
        if not available_keys:
            raise Exception(f"No {self.provider_name} API keys available (all may be in cooldown)")

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

                # Azure SDK streaming is synchronous, we'll need to adapt
                # For now, let's do a non-streaming call and simulate streaming
                response = client.complete(**params)

                # Convert to streaming format
                yield f"data: {{\"id\":\"chatcmpl-{model.replace('/', '-')}\",\"object\":\"chat.completion.chunk\",\"created\":{int(__import__('time').time())},\"model\":\"{model}\",\"choices\":[{{\"index\":0,\"delta\":{{\"role\":\"assistant\",\"content\":\"{response.choices[0].message.content}\"}},\"finish_reason\":\"stop\"}}]}}\n\n"
                yield "data: [DONE]\n\n"
                return

            except Exception as e:
                self._mark_key_failed(api_key)
                retry_count += 1
                last_error = e
                continue

        # All keys failed
        raise last_error or Exception(f"All {self.provider_name} API keys failed")

    def _convert_azure_to_openai(self, azure_response, model: str) -> Dict[str, Any]:
        """Convert Azure response to OpenAI format."""
        return {
            "id": f"chatcmpl-{model.replace('/', '-')}",
            "object": "chat.completion",
            "created": int(__import__('time').time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": azure_response.choices[0].message.content
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": getattr(azure_response.usage, 'prompt_tokens', 0) if hasattr(azure_response, 'usage') else 0,
                "completion_tokens": getattr(azure_response.usage, 'completion_tokens', 0) if hasattr(azure_response, 'usage') else 0,
                "total_tokens": getattr(azure_response.usage, 'total_tokens', 0) if hasattr(azure_response, 'usage') else 0
            }
        }
