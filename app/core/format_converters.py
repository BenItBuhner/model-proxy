"""
Format converters for converting between OpenAI and Anthropic request/response formats.
"""

import json
import time
from typing import Dict, Any, List, Tuple, Union, Iterable, Optional


TEXT_BLOCK = "text"
TOOL_USE_BLOCK = "tool_use"
TOOL_RESULT_BLOCK = "tool_result"


def _json_dumps_compact(value: Any) -> str:
    try:
        return json.dumps(value, separators=(",", ":"))
    except Exception:
        return json.dumps(str(value))


def _parse_arguments(arguments: Any) -> Dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if not arguments:
        return {}
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        return {"_raw": arguments}
    return {"value": arguments}


def _openai_text_blocks(content: Any) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    if content is None:
        return blocks
    if isinstance(content, str):
        if content:
            blocks.append({"type": TEXT_BLOCK, "text": content})
        return blocks
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == TEXT_BLOCK:
                    blocks.append({"type": TEXT_BLOCK, "text": part.get("text", "")})
                else:
                    blocks.append(
                        {"type": TEXT_BLOCK, "text": _json_dumps_compact(part)}
                    )
            elif isinstance(part, str):
                blocks.append({"type": TEXT_BLOCK, "text": part})
            elif part is not None:
                blocks.append({"type": TEXT_BLOCK, "text": str(part)})
        return blocks
    blocks.append({"type": TEXT_BLOCK, "text": str(content)})
    return blocks


def _collapse_blocks(blocks: List[Dict[str, Any]]) -> Union[str, List[Dict[str, Any]]]:
    if not blocks:
        return ""
    if all(block.get("type") == TEXT_BLOCK for block in blocks):
        return "".join(block.get("text", "") for block in blocks)
    return blocks


def _normalize_tool_result_content(content: Any) -> Any:
    if isinstance(content, list):
        normalized: List[Dict[str, Any]] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == TEXT_BLOCK:
                normalized.append({"type": TEXT_BLOCK, "text": part.get("text", "")})
            elif isinstance(part, str):
                normalized.append({"type": TEXT_BLOCK, "text": part})
            elif part is not None:
                normalized.append({"type": TEXT_BLOCK, "text": str(part)})
        return normalized or ""
    return content or ""


def _iter_anthropic_blocks(content: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                yield block
    elif isinstance(content, dict):
        yield content
    elif content not in (None, ""):
        yield {"type": TEXT_BLOCK, "text": str(content)}


def _anthropic_block_to_text(block: Dict[str, Any]) -> str:
    btype = block.get("type")
    if btype == TEXT_BLOCK:
        return block.get("text", "")
    if btype == TOOL_RESULT_BLOCK:
        inner = block.get("content")
        if isinstance(inner, list):
            return "".join(
                _anthropic_block_to_text(sub) for sub in inner if isinstance(sub, dict)
            )
        if isinstance(inner, str):
            return inner
        return str(inner or "")
    if btype == TOOL_USE_BLOCK:
        name = block.get("name") or "tool"
        return f"[tool_use:{name}]"
    return _json_dumps_compact(block)


def _split_anthropic_user_content(
    content: Any,
) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    text_chunks: List[str] = []
    tool_messages: List[Dict[str, Any]] = []
    for block in _iter_anthropic_blocks(content):
        btype = block.get("type")
        if btype == TEXT_BLOCK:
            text_chunks.append(block.get("text", ""))
        elif btype == TOOL_RESULT_BLOCK:
            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id"),
                    "content": block.get("content")
                    if isinstance(block.get("content"), str)
                    else _anthropic_block_to_text(block),
                }
            )
        else:
            text_chunks.append(_anthropic_block_to_text(block))
    text = "\n".join(filter(None, text_chunks))
    return (text or None, tool_messages)


def _split_anthropic_assistant_content(
    content: Any,
) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    text_chunks: List[str] = []
    tool_calls: List[Dict[str, Any]] = []
    for block in _iter_anthropic_blocks(content):
        btype = block.get("type")
        if btype == TEXT_BLOCK:
            text_chunks.append(block.get("text", ""))
        elif btype == TOOL_USE_BLOCK:
            args = _json_dumps_compact(block.get("input", {}))
            tool_calls.append(
                {
                    "id": block.get("id") or f"call_{len(tool_calls)}",
                    "type": "function",
                    "function": {"name": block.get("name"), "arguments": args},
                }
            )
        elif btype == TOOL_RESULT_BLOCK:
            text_chunks.append(_anthropic_block_to_text(block))
        else:
            text_chunks.append(_anthropic_block_to_text(block))
    text = "\n".join(filter(None, text_chunks))
    return (text or None, tool_calls)


def anthropic_to_openai_request(anthropic_request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert Anthropic request format to OpenAI format.
    """
    openai_messages: List[Dict[str, Any]] = []
    system_content = anthropic_request.get("system")
    if system_content:
        system_text = (
            _anthropic_block_to_text({"type": TEXT_BLOCK, "text": system_content})
            if isinstance(system_content, str)
            else ""
        )
        if not system_text and isinstance(system_content, list):
            system_text = "\n".join(
                _anthropic_block_to_text(block)
                for block in system_content
                if isinstance(block, dict)
            )
        if system_text:
            openai_messages.append({"role": "system", "content": system_text})

    for msg in anthropic_request.get("messages", []):
        role = msg.get("role")
        content = msg.get("content")
        if role == "assistant":
            text, tool_calls = _split_anthropic_assistant_content(content)
            assistant_message: Dict[str, Any] = {"role": "assistant", "content": text}
            if tool_calls:
                assistant_message["tool_calls"] = tool_calls
            openai_messages.append(assistant_message)
        elif role == "user":
            text, tool_outputs = _split_anthropic_user_content(content)
            # IMPORTANT: In OpenAI tool-calling, tool messages must come immediately
            # after the assistant message that emitted tool_calls. Some providers
            # (e.g. Cerebras) are strict about this ordering.
            #
            # So if an Anthropic user message contains tool_result blocks, emit the
            # converted OpenAI tool messages FIRST, then any user text AFTER.
            openai_messages.extend(tool_outputs)
            if text is not None:
                openai_messages.append({"role": "user", "content": text})
        else:
            openai_messages.append(
                {
                    "role": role,
                    "content": _anthropic_block_to_text(
                        {"type": TEXT_BLOCK, "text": content}
                    ),
                }
            )

    openai_request = {
        "model": anthropic_request.get("model"),
        "messages": openai_messages,
        "max_tokens": anthropic_request.get("max_tokens"),
    }

    if "temperature" in anthropic_request:
        openai_request["temperature"] = anthropic_request["temperature"]
    if "top_p" in anthropic_request:
        openai_request["top_p"] = anthropic_request["top_p"]
    if "stream" in anthropic_request:
        openai_request["stream"] = anthropic_request["stream"]
    if "tools" in anthropic_request:
        tools = []
        for tool in anthropic_request["tools"]:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.get("name"),
                        "description": tool.get("description"),
                        "parameters": tool.get("input_schema", {}),
                    },
                }
            )
        openai_request["tools"] = tools
    if "tool_choice" in anthropic_request:
        tool_choice = anthropic_request["tool_choice"]
        if isinstance(tool_choice, dict) and tool_choice.get("type") == "tool":
            openai_request["tool_choice"] = {
                "type": "function",
                "function": {"name": tool_choice.get("name")},
            }
        else:
            openai_request["tool_choice"] = tool_choice
    if anthropic_request.get("stop_sequences"):
        stop_sequences = anthropic_request["stop_sequences"]
        if len(stop_sequences) == 1:
            openai_request["stop"] = stop_sequences[0]
        else:
            openai_request["stop"] = stop_sequences

    return openai_request


def openai_to_anthropic_request(openai_request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert OpenAI request format to Anthropic format.
    """
    messages: List[Dict[str, Any]] = []
    system_segments: List[str] = []

    for msg in openai_request.get("messages", []):
        role = msg.get("role")
        content = msg.get("content")

        if role == "system":
            text = msg.get("content")
            if isinstance(text, list):
                text = "".join(
                    part.get("text", "") for part in text if isinstance(part, dict)
                )
            if text:
                system_segments.append(text)
            continue

        if role == "tool":
            block = {
                "type": TOOL_RESULT_BLOCK,
                "tool_use_id": msg.get("tool_call_id"),
                "content": _normalize_tool_result_content(content),
                "is_error": msg.get("metadata", {}).get("is_error")
                if isinstance(msg.get("metadata"), dict)
                else None,
            }
            messages.append({"role": "user", "content": [block]})
            continue

        target_role = role if role in ("assistant", "user") else "user"
        blocks = _openai_text_blocks(content)
        if role == "assistant" and msg.get("tool_calls"):
            for call in msg.get("tool_calls", []):
                parsed_args = _parse_arguments(
                    call.get("function", {}).get("arguments")
                )
                blocks.append(
                    {
                        "type": TOOL_USE_BLOCK,
                        "id": call.get("id"),
                        "name": call.get("function", {}).get("name"),
                        "input": parsed_args,
                    }
                )
        normalized_content = _collapse_blocks(blocks)
        messages.append({"role": target_role, "content": normalized_content})

    system_prompt = "\n\n".join(seg for seg in system_segments if seg)

    anthropic_request: Dict[str, Any] = {
        "model": openai_request.get("model"),
        "messages": messages,
        "max_tokens": openai_request.get("max_tokens") or 1024,
    }

    if system_prompt:
        anthropic_request["system"] = system_prompt
    if "temperature" in openai_request:
        anthropic_request["temperature"] = openai_request["temperature"]
    if "top_p" in openai_request:
        anthropic_request["top_p"] = openai_request["top_p"]
    if "stream" in openai_request:
        anthropic_request["stream"] = openai_request["stream"]
    stop = openai_request.get("stop")
    if isinstance(stop, list):
        anthropic_request["stop_sequences"] = stop
    elif isinstance(stop, str):
        anthropic_request["stop_sequences"] = [stop]
    if "tools" in openai_request:
        tools = []
        for tool in openai_request["tools"]:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                tools.append(
                    {
                        "name": func.get("name"),
                        "description": func.get("description", ""),
                        "input_schema": func.get("parameters", {}),
                    }
                )
        anthropic_request["tools"] = tools
    tool_choice = openai_request.get("tool_choice")
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        anthropic_request["tool_choice"] = {
            "type": "tool",
            "name": tool_choice.get("function", {}).get("name"),
        }
    elif tool_choice:
        anthropic_request["tool_choice"] = tool_choice

    return anthropic_request


def anthropic_to_openai_response(
    anthropic_response: Dict[str, Any], model: str
) -> Dict[str, Any]:
    """
    Convert Anthropic response format to OpenAI format.
    """
    content_blocks = anthropic_response.get("content", [])
    text_content = ""
    tool_calls = None

    for block in content_blocks:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == TEXT_BLOCK:
            text_content += block.get("text", "")
        elif btype == TOOL_USE_BLOCK:
            if tool_calls is None:
                tool_calls = []
            tool_calls.append(
                {
                    "id": block.get("id"),
                    "type": "function",
                    "function": {
                        "name": block.get("name"),
                        "arguments": _json_dumps_compact(block.get("input", {})),
                    },
                }
            )
        elif btype == TOOL_RESULT_BLOCK:
            text_content += _anthropic_block_to_text(block)

    stop_reason = anthropic_response.get("stop_reason")
    finish_reason_map = {
        "end_turn": "stop",
        "max_tokens": "length",
        "stop_sequence": "stop",
        "tool_use": "tool_calls",
        None: "stop",
    }
    # If there are tool calls, finish_reason should be "tool_calls" regardless of stop_reason
    finish_reason = (
        "tool_calls" if tool_calls else finish_reason_map.get(stop_reason, "stop")
    )

    usage = anthropic_response.get("usage", {})

    message_content = None if tool_calls else text_content

    openai_response = {
        "id": anthropic_response.get("id", f"chatcmpl-{int(time.time())}"),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": message_content,
                },
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("input_tokens", 0)
            + usage.get("output_tokens", 0),
        },
    }

    if tool_calls:
        openai_response["choices"][0]["message"]["tool_calls"] = tool_calls

    return openai_response


def openai_to_anthropic_response(
    openai_response: Dict[str, Any], model: str
) -> Dict[str, Any]:
    """
    Convert OpenAI response format to Anthropic format.
    """
    choice = openai_response.get("choices", [{}])[0]
    message = choice.get("message", {})

    content_blocks: List[Dict[str, Any]] = []
    content = message.get("content")
    if isinstance(content, str) and content:
        content_blocks.append({"type": TEXT_BLOCK, "text": content})
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == TEXT_BLOCK:
                content_blocks.append(
                    {"type": TEXT_BLOCK, "text": part.get("text", "")}
                )
            elif isinstance(part, str):
                content_blocks.append({"type": TEXT_BLOCK, "text": part})

    tool_calls = message.get("tool_calls") or []
    for tool_call in tool_calls:
        func = tool_call.get("function", {})
        input_data = _parse_arguments(func.get("arguments"))
        content_blocks.append(
            {
                "type": TOOL_USE_BLOCK,
                "id": tool_call.get("id"),
                "name": func.get("name"),
                "input": input_data,
            }
        )

    finish_reason = choice.get("finish_reason", "stop")
    stop_reason_map = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
    }
    stop_reason = stop_reason_map.get(finish_reason, "end_turn")

    usage = openai_response.get("usage", {})

    anthropic_response = {
        "id": openai_response.get("id", f"msg-{int(time.time())}"),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    }

    return anthropic_response
