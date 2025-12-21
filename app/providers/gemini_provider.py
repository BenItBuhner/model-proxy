"""
Native Gemini provider implementation using the Google GenAI SDK.
Handles Gemini API calls with key rotation, streaming, and tool support.
Uses the native google-genai SDK for better compatibility and reliability.
"""

import json
import logging
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

from app.core.api_key_manager import get_available_keys
from app.providers.base import BaseProvider

logger = logging.getLogger("gemini_provider")


class GeminiProviderError(Exception):
    """Gemini provider request failed with an HTTP status code."""

    def __init__(self, message: str, status: int, body: Optional[str] = None):
        super().__init__(message)
        self.status = status
        self.body = body


class GeminiProvider(BaseProvider):
    """Provider for Google Gemini API using the native GenAI SDK."""

    def __init__(self):
        super().__init__("gemini")
        self.timeout = 60

    def _get_available_api_keys(self) -> List[str]:
        """
        Get list of API keys to try.

        If route-specific API key is set, return only that key.
        Otherwise, return all available keys from environment.
        """
        if self._route_api_key:
            return [self._route_api_key]
        return get_available_keys(self.provider_name)

    def _convert_messages_to_genai_format(
        self, messages: List[Dict[str, Any]]
    ) -> tuple[Optional[str], List[Dict[str, Any]]]:
        """
        Convert OpenAI-style messages to Google GenAI format.

        Returns:
            Tuple of (system_instruction, contents)
        """
        system_instruction = None
        contents = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content")

            # Extract system message as system_instruction
            if role == "system":
                if isinstance(content, str):
                    system_instruction = content
                elif isinstance(content, list):
                    # Handle content parts
                    text_parts = [
                        p.get("text", "") for p in content if p.get("type") == "text"
                    ]
                    system_instruction = " ".join(text_parts)
                continue

            # Map OpenAI roles to Gemini roles
            gemini_role = "user" if role == "user" else "model"

            # Handle tool/function responses
            if role == "tool" or role == "function":
                msg.get("tool_call_id", "")
                function_name = msg.get("name", "")
                # Create a function response part
                contents.append(
                    {
                        "role": "user",
                        "parts": [
                            {
                                "function_response": {
                                    "name": function_name,
                                    "response": {"result": content},
                                }
                            }
                        ],
                    }
                )
                continue

            # Build parts list
            parts = []

            if isinstance(content, str):
                if content:  # Only add non-empty text
                    parts.append({"text": content})
            elif isinstance(content, list):
                # Handle multimodal content
                for item in content:
                    if isinstance(item, dict):
                        item_type = item.get("type", "text")
                        if item_type == "text":
                            text = item.get("text", "")
                            if text:
                                parts.append({"text": text})
                        elif item_type == "image_url":
                            image_url = item.get("image_url", {})
                            url = (
                                image_url.get("url", "")
                                if isinstance(image_url, dict)
                                else ""
                            )
                            if url.startswith("data:"):
                                # Base64 encoded image
                                # Format: data:image/jpeg;base64,<data>
                                try:
                                    header, data = url.split(",", 1)
                                    mime_type = header.split(":")[1].split(";")[0]
                                    parts.append(
                                        {
                                            "inline_data": {
                                                "mime_type": mime_type,
                                                "data": data,
                                            }
                                        }
                                    )
                                except (ValueError, IndexError):
                                    logger.warning(
                                        f"Failed to parse base64 image: {url[:50]}..."
                                    )
                            else:
                                # URL reference
                                parts.append(
                                    {
                                        "file_data": {
                                            "file_uri": url,
                                            "mime_type": "image/jpeg",
                                        }
                                    }
                                )

            # Handle assistant messages with tool calls
            if role == "assistant" and msg.get("tool_calls"):
                for tool_call in msg.get("tool_calls", []):
                    func = tool_call.get("function", {})
                    func_name = func.get("name", "")
                    func_args = func.get("arguments", "{}")
                    try:
                        args_dict = (
                            json.loads(func_args)
                            if isinstance(func_args, str)
                            else func_args
                        )
                    except json.JSONDecodeError:
                        args_dict = {}
                    parts.append(
                        {
                            "function_call": {
                                "name": func_name,
                                "args": args_dict,
                            }
                        }
                    )

            if parts:
                contents.append({"role": gemini_role, "parts": parts})

        return system_instruction, contents

    def _convert_tools_to_genai_format(
        self, tools: Optional[List[Dict[str, Any]]]
    ) -> Optional[List[Dict[str, Any]]]:
        """Convert OpenAI-style tools to Google GenAI format."""
        if not tools:
            return None

        function_declarations = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                declaration = {
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                }
                if func.get("parameters"):
                    declaration["parameters"] = func.get("parameters")
                function_declarations.append(declaration)

        if function_declarations:
            return [{"function_declarations": function_declarations}]
        return None

    def _convert_response_to_openai_format(
        self, response: Any, model: str, stream: bool = False
    ) -> Dict[str, Any]:
        """Convert Google GenAI response to OpenAI format."""
        response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        # Extract content from response
        text_content = ""
        tool_calls = []
        finish_reason = "stop"

        try:
            if hasattr(response, "text") and response.text:
                text_content = response.text

            # Check for function calls
            if hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, "content") and candidate.content:
                    for part in candidate.content.parts:
                        if hasattr(part, "function_call") and part.function_call:
                            fc = part.function_call
                            tool_calls.append(
                                {
                                    "id": f"call_{uuid.uuid4().hex[:8]}",
                                    "type": "function",
                                    "function": {
                                        "name": fc.name,
                                        "arguments": json.dumps(
                                            dict(fc.args)
                                            if hasattr(fc.args, "items")
                                            else fc.args
                                        ),
                                    },
                                }
                            )
                            finish_reason = "tool_calls"
                        elif hasattr(part, "text") and part.text:
                            text_content = part.text

                # Get finish reason from candidate
                if hasattr(candidate, "finish_reason"):
                    fr = candidate.finish_reason
                    if fr:
                        fr_str = str(fr).upper()
                        if "STOP" in fr_str:
                            finish_reason = "stop"
                        elif "MAX_TOKENS" in fr_str or "LENGTH" in fr_str:
                            finish_reason = "length"
                        elif "SAFETY" in fr_str:
                            finish_reason = "content_filter"
                        elif "FUNCTION" in fr_str or "TOOL" in fr_str:
                            finish_reason = "tool_calls"

            # Extract usage
            usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                um = response.usage_metadata
                usage["prompt_tokens"] = getattr(um, "prompt_token_count", 0) or 0
                usage["completion_tokens"] = (
                    getattr(um, "candidates_token_count", 0) or 0
                )
                usage["total_tokens"] = getattr(um, "total_token_count", 0) or 0

        except Exception as e:
            logger.warning(f"Error extracting response content: {e}")

        message = {
            "role": "assistant",
            "content": text_content if text_content else None,
        }
        if tool_calls:
            message["tool_calls"] = tool_calls

        return {
            "id": response_id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": usage,
        }

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
        Make a synchronous API call to Gemini.

        Args:
            model: Model name
            messages: List of message dicts
            **kwargs: Additional parameters

        Returns:
            OpenAI-compatible API response

        Raises:
            Exception: If all API keys fail
        """
        # Import here to avoid issues if SDK not installed
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise ImportError(
                "google-genai package is required for native Gemini support. "
                "Install it with: pip install google-genai"
            )

        available_keys = self._get_available_api_keys()
        if not available_keys:
            raise Exception(f"No {self.provider_name} API keys available")

        last_error = None

        # Convert messages to Gemini format
        system_instruction, contents = self._convert_messages_to_genai_format(messages)

        # Convert tools to Gemini format
        genai_tools = self._convert_tools_to_genai_format(tools)

        # Try each available key until one succeeds
        for api_key in available_keys:
            try:
                client = genai.Client(api_key=api_key)

                # Build config
                config_params = {}
                if temperature is not None:
                    config_params["temperature"] = temperature
                if top_p is not None:
                    config_params["top_p"] = top_p
                if max_tokens is not None:
                    config_params["max_output_tokens"] = max_tokens
                if stop is not None:
                    config_params["stop_sequences"] = (
                        stop if isinstance(stop, list) else [stop]
                    )
                if system_instruction:
                    config_params["system_instruction"] = system_instruction
                if genai_tools:
                    config_params["tools"] = genai_tools

                config = (
                    types.GenerateContentConfig(**config_params)
                    if config_params
                    else None
                )

                response = await client.aio.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )

                return self._convert_response_to_openai_format(response, model)

            except Exception as e:
                error_str = str(e)
                self._mark_key_failed(api_key)
                logger.warning(f"Gemini API call failed with key: {error_str[:100]}")

                # Try to extract status code
                status = 500
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    status = 429
                elif (
                    "401" in error_str
                    or "403" in error_str
                    or "PERMISSION_DENIED" in error_str
                ):
                    status = 403
                elif "400" in error_str or "INVALID_ARGUMENT" in error_str:
                    status = 400

                last_error = GeminiProviderError(
                    f"Gemini API error: {error_str}",
                    status=status,
                    body=error_str,
                )
                continue

        # All keys failed
        raise last_error or Exception("All Gemini API keys failed")

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
        Make a streaming API call to Gemini.

        Args:
            model: Model name
            messages: List of message dicts
            **kwargs: Additional parameters

        Yields:
            SSE formatted chunks

        Raises:
            Exception: If all API keys fail
        """
        # Import here to avoid issues if SDK not installed
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise ImportError(
                "google-genai package is required for native Gemini support. "
                "Install it with: pip install google-genai"
            )

        available_keys = self._get_available_api_keys()
        if not available_keys:
            raise Exception(f"No {self.provider_name} API keys available")

        last_error = None

        # Convert messages to Gemini format
        system_instruction, contents = self._convert_messages_to_genai_format(messages)

        # Convert tools to Gemini format
        genai_tools = self._convert_tools_to_genai_format(tools)

        # Try each available key until one succeeds
        for api_key in available_keys:
            try:
                client = genai.Client(api_key=api_key)

                # Build config
                config_params = {}
                if temperature is not None:
                    config_params["temperature"] = temperature
                if top_p is not None:
                    config_params["top_p"] = top_p
                if max_tokens is not None:
                    config_params["max_output_tokens"] = max_tokens
                if stop is not None:
                    config_params["stop_sequences"] = (
                        stop if isinstance(stop, list) else [stop]
                    )
                if system_instruction:
                    config_params["system_instruction"] = system_instruction
                if genai_tools:
                    config_params["tools"] = genai_tools

                config = (
                    types.GenerateContentConfig(**config_params)
                    if config_params
                    else None
                )

                response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
                created = int(time.time())

                # Track tool calls across chunks
                tool_call_index = 0

                async for chunk in await client.aio.models.generate_content_stream(
                    model=model,
                    contents=contents,
                    config=config,
                ):
                    delta = {"role": "assistant"}
                    finish_reason = None

                    # Extract text from chunk
                    chunk_text = ""
                    if hasattr(chunk, "text") and chunk.text:
                        chunk_text = chunk.text
                        delta["content"] = chunk_text

                    # Check for function calls in this chunk
                    if hasattr(chunk, "candidates") and chunk.candidates:
                        candidate = chunk.candidates[0]
                        if hasattr(candidate, "content") and candidate.content:
                            for part in candidate.content.parts:
                                if (
                                    hasattr(part, "function_call")
                                    and part.function_call
                                ):
                                    fc = part.function_call
                                    tool_call = {
                                        "index": tool_call_index,
                                        "id": f"call_{uuid.uuid4().hex[:8]}",
                                        "type": "function",
                                        "function": {
                                            "name": fc.name,
                                            "arguments": json.dumps(
                                                dict(fc.args)
                                                if hasattr(fc.args, "items")
                                                else fc.args
                                            ),
                                        },
                                    }
                                    delta["tool_calls"] = [tool_call]
                                    tool_call_index += 1
                                    finish_reason = "tool_calls"

                        # Check finish reason
                        if (
                            hasattr(candidate, "finish_reason")
                            and candidate.finish_reason
                        ):
                            fr = str(candidate.finish_reason).upper()
                            if "STOP" in fr:
                                finish_reason = "stop"
                            elif "MAX_TOKENS" in fr or "LENGTH" in fr:
                                finish_reason = "length"
                            elif "SAFETY" in fr:
                                finish_reason = "content_filter"
                            elif "FUNCTION" in fr or "TOOL" in fr:
                                finish_reason = "tool_calls"

                    # Build chunk response
                    chunk_response = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": delta,
                                "finish_reason": finish_reason,
                            }
                        ],
                    }

                    yield f"data: {json.dumps(chunk_response)}\n\n"

                # Send final done marker
                yield "data: [DONE]\n\n"
                return  # Success

            except Exception as e:
                error_str = str(e)
                self._mark_key_failed(api_key)
                logger.warning(f"Gemini streaming API call failed: {error_str[:100]}")

                # Try to extract status code
                status = 500
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    status = 429
                elif (
                    "401" in error_str
                    or "403" in error_str
                    or "PERMISSION_DENIED" in error_str
                ):
                    status = 403
                elif "400" in error_str or "INVALID_ARGUMENT" in error_str:
                    status = 400

                last_error = GeminiProviderError(
                    f"Gemini API error: {error_str}",
                    status=status,
                    body=error_str,
                )
                continue

        # All keys failed
        raise last_error or Exception("All Gemini API keys failed")
