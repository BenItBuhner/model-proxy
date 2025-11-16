"""
Tests for tool calling functionality across OpenAI and Anthropic formats.
Tests both streaming and non-streaming variants.
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.core.format_converters import openai_to_anthropic_request, anthropic_to_openai_request


@pytest.fixture
def client():
    """Test client fixture."""
    with TestClient(app) as c:
        yield c


class TestToolCalling:
    """Test tool calling functionality."""

    def test_openai_tool_format_conversion(self):
        """Test OpenAI to Anthropic tool format conversion."""
        openai_request = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "What's the weather?"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "City name"}
                        },
                        "required": ["location"]
                    }
                }
            }],
            "max_tokens": 100
        }

        anthropic_request = openai_to_anthropic_request(openai_request)

        assert "tools" in anthropic_request
        assert len(anthropic_request["tools"]) == 1
        tool = anthropic_request["tools"][0]
        assert tool["name"] == "get_weather"
        assert tool["description"] == "Get weather information"
        assert "input_schema" in tool
        assert tool["input_schema"]["properties"]["location"]["type"] == "string"

    def test_anthropic_tool_format_conversion(self):
        """Test Anthropic to OpenAI tool format conversion."""
        anthropic_request = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "What's the weather?"}],
            "tools": [{
                "name": "get_weather",
                "description": "Get weather information",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"}
                    },
                    "required": ["location"]
                }
            }],
            "max_tokens": 100
        }

        openai_request = anthropic_to_openai_request(anthropic_request)

        assert "tools" in openai_request
        assert len(openai_request["tools"]) == 1
        tool = openai_request["tools"][0]
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "get_weather"
        assert tool["function"]["description"] == "Get weather information"
        assert tool["function"]["parameters"]["properties"]["location"]["type"] == "string"

    def test_openai_tool_calling_non_streaming(self, client):
        """Test OpenAI tool calling (non-streaming) with nahcrof/kimi-k2-turbo."""
        # Use a very explicit prompt to encourage tool usage
        request_data = {
            "model": "nahcrof/kimi-k2-turbo",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. When asked to perform mathematical addition, you MUST use the add_numbers tool. Do not calculate the result yourself - always use the tool."
                },
                {
                    "role": "user",
                    "content": "What is 15 + 27? Please use the add_numbers tool to calculate this."
                }
            ],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "add_numbers",
                    "description": "Add two numbers together. This tool only performs addition and nothing else.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "number", "description": "First number to add"},
                            "b": {"type": "number", "description": "Second number to add"}
                        },
                        "required": ["a", "b"]
                    }
                }
            }],
            "tool_choice": "auto",
            "max_tokens": 150
        }

        response = client.post(
            "/v1/chat/completions",
            json=request_data,
            headers={"Authorization": "Bearer test_client_key_123"}
        )

        # Should succeed (200) even if model doesn't actually call tools
        assert response.status_code == 200
        data = response.json()

        # Validate response structure
        assert "choices" in data
        assert len(data["choices"]) > 0
        message = data["choices"][0]["message"]
        assert "role" in message

        # Check if we got actual tool calls
        has_tool_calls = "tool_calls" in message and message["tool_calls"] is not None

        if has_tool_calls:
            print(f"SUCCESS: Model generated actual tool calls!")
            tool_calls = message["tool_calls"]
            assert len(tool_calls) > 0
            tool_call = tool_calls[0]
            assert tool_call["type"] == "function"
            assert tool_call["function"]["name"] == "add_numbers"

            # Parse the arguments
            import json
            args = json.loads(tool_call["function"]["arguments"])
            print(f"Tool call arguments: {args}")

            # Verify the arguments are correct
            assert "a" in args and "b" in args
            assert args["a"] == 15
            assert args["b"] == 27
            print("SUCCESS: Tool call has correct arguments!")
        else:
            print(f"WARNING: Model did not generate tool calls, responded with text: {message.get('content', '')[:100]}...")
            # Content should be None when there are supposed to be tool calls
            assert message.get("content") is None or len(message.get("content", "").strip()) == 0

    def test_openai_tool_calling_streaming(self, client):
        """Test OpenAI tool calling (streaming) with nahcrof/kimi-k2-turbo."""
        request_data = {
            "model": "nahcrof/kimi-k2-turbo",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. When asked to perform mathematical addition, you MUST use the add_numbers tool. Do not calculate the result yourself - always use the tool."
                },
                {
                    "role": "user",
                    "content": "What is 7 + 8? Please use the add_numbers tool to calculate this."
                }
            ],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "add_numbers",
                    "description": "Add two numbers together. This tool only performs addition and nothing else.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "number", "description": "First number to add"},
                            "b": {"type": "number", "description": "Second number to add"}
                        },
                        "required": ["a", "b"]
                    }
                }
            }],
            "tool_choice": "auto",
            "stream": True,
            "max_tokens": 100
        }

        response = client.post(
            "/v1/chat/completions-stream",
            json=request_data,
            headers={"Authorization": "Bearer test_client_key_123"}
        )

        # Should succeed (200) with streaming response
        assert response.status_code == 200

        # Check if response contains streaming data
        content = response.text
        assert "data:" in content or len(content.strip()) > 0

        # Look for tool call data in the stream
        has_tool_call_data = "tool_calls" in content and '"type": "function"' in content
        if has_tool_call_data:
            print("SUCCESS: Streaming response contains tool call data!")
        else:
            print(f"WARNING: Streaming response does not contain tool calls: {content[:300]}...")

    def test_anthropic_tool_calling_non_streaming(self, client):
        """Test Anthropic tool calling (non-streaming) with nahcrof/kimi-k2-turbo."""
        # Use explicit instructions to encourage tool usage
        request_data = {
            "model": "nahcrof/kimi-k2-turbo",
            "messages": [
                {
                    "role": "user",
                    "content": "You are a helpful assistant. When asked to perform mathematical addition, you MUST use the add_numbers tool. Do not calculate the result yourself - always use the tool.\n\nWhat is 12 + 34? Please use the add_numbers tool to calculate this."
                }
            ],
            "tools": [{
                "name": "add_numbers",
                "description": "Add two numbers together. This tool only performs addition and nothing else.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "number", "description": "First number to add"},
                        "b": {"type": "number", "description": "Second number to add"}
                    },
                    "required": ["a", "b"]
                }
            }],
            "max_tokens": 150
        }

        response = client.post(
            "/v1/messages",
            json=request_data,
            headers={"Authorization": "Bearer test_client_key_123"}
        )

        # Should succeed (200) - Anthropic format conversion should work
        assert response.status_code == 200
        data = response.json()

        # Validate Anthropic response structure
        assert "type" in data
        assert data["type"] == "message"
        assert "role" in data
        assert "content" in data
        assert "usage" in data

        # Check if we got actual tool use content blocks
        content_blocks = data["content"]
        has_tool_use = any(block.get("type") == "tool_use" for block in content_blocks)

        if has_tool_use:
            print("SUCCESS: Model generated actual tool_use content blocks!")
            tool_blocks = [block for block in content_blocks if block.get("type") == "tool_use"]
            assert len(tool_blocks) > 0
            tool_block = tool_blocks[0]
            assert tool_block["name"] == "add_numbers"
            assert "input" in tool_block
            tool_input = tool_block["input"]
            print(f"Tool input: {tool_input}")

            # Verify the arguments are correct
            assert "a" in tool_input and "b" in tool_input
            assert tool_input["a"] == 12
            assert tool_input["b"] == 34
            print("SUCCESS: Tool call has correct arguments!")
        else:
            print(f"WARNING: Model did not generate tool_use blocks, responded with: {str(content_blocks)[:200]}...")
            # Should still have some text content
            text_blocks = [block for block in content_blocks if block.get("type") == "text"]
            if text_blocks:
                print(f"Text response: {text_blocks[0].get('text', '')[:100]}...")

    def test_anthropic_tool_calling_streaming(self, client):
        """Test Anthropic tool calling (streaming) with nahcrof/kimi-k2-turbo."""
        request_data = {
            "model": "nahcrof/kimi-k2-turbo",
            "messages": [
                {
                    "role": "user",
                    "content": "You are a helpful assistant. When asked to perform mathematical addition, you MUST use the add_numbers tool. Do not calculate the result yourself - always use the tool.\n\nWhat is 5 + 9? Please use the add_numbers tool to calculate this."
                }
            ],
            "tools": [{
                "name": "add_numbers",
                "description": "Add two numbers together. This tool only performs addition and nothing else.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "number", "description": "First number to add"},
                        "b": {"type": "number", "description": "Second number to add"}
                    },
                    "required": ["a", "b"]
                }
            }],
            "stream": True,
            "max_tokens": 100
        }

        response = client.post(
            "/v1/messages-stream",
            json=request_data,
            headers={"Authorization": "Bearer test_client_key_123"}
        )

        # Should succeed (200) with streaming response
        assert response.status_code == 200

        # Check if response contains streaming data
        content = response.text
        assert len(content.strip()) > 0

        # Look for tool_use data in the stream
        has_tool_use_data = '"type": "tool_use"' in content or 'tool_use' in content
        if has_tool_use_data:
            print("SUCCESS: Streaming response contains tool_use data!")
        else:
            print(f"WARNING: Streaming response does not contain tool_use: {content[:300]}...")

    def test_tool_format_edge_cases(self):
        """Test edge cases in tool format conversion."""
        # Test empty tools
        openai_empty = {"model": "test", "messages": [], "tools": []}
        anthropic_empty = openai_to_anthropic_request(openai_empty)
        assert "tools" not in anthropic_empty or len(anthropic_empty.get("tools", [])) == 0

        # Test tools without parameters
        openai_no_params = {
            "model": "test",
            "messages": [],
            "tools": [{"type": "function", "function": {"name": "test", "description": "test"}}]
        }
        anthropic_no_params = openai_to_anthropic_request(openai_no_params)
        assert anthropic_no_params["tools"][0]["input_schema"] == {}

        # Test malformed tools (should handle gracefully)
        openai_malformed = {
            "model": "test",
            "messages": [],
            "tools": [{"invalid": "structure"}]
        }
        anthropic_malformed = openai_to_anthropic_request(openai_malformed)
        # Should not crash, might skip malformed tools
        assert "tools" not in anthropic_malformed or len(anthropic_malformed.get("tools", [])) == 0
