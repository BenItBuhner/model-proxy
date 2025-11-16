from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union, Literal


class ContentBlock(BaseModel):
    """
    Anthropic content block.
    Supports text, tool_use, tool_result, and future block types.
    """

    type: str
    text: Optional[str] = None
    id: Optional[str] = None
    name: Optional[str] = None
    input: Optional[Dict[str, Any]] = None
    tool_use_id: Optional[str] = None
    content: Optional[Any] = None
    is_error: Optional[bool] = None


MessageContent = Union[str, List[ContentBlock]]


class AnthropicMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: MessageContent
    metadata: Optional[Dict[str, Any]] = None


class Tool(BaseModel):
    name: str
    description: str
    input_schema: Dict[str, Any]


class ToolChoice(BaseModel):
    type: str
    name: Optional[str] = None


class AnthropicMessagesRequest(BaseModel):
    model: str
    messages: List[AnthropicMessage]
    max_tokens: int = Field(default=16000, gt=0)
    system: Optional[Union[str, List[ContentBlock]]] = None
    stop_sequences: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=None, ge=0)
    stream: Optional[bool] = False
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Union[str, ToolChoice]] = None
    betas: Optional[List[str]] = None
    thinking: Optional[Dict[str, Any]] = None


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class AnthropicMessagesResponse(BaseModel):
    id: str
    type: Literal["message", "error"] = "message"
    role: Literal["assistant", "user"] = "assistant"
    model: str
    content: List[ContentBlock]
    stop_reason: Optional[str] = None
    stop_sequence: Optional[str] = None
    usage: Usage
