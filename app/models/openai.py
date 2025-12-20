from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class FunctionDefinition(BaseModel):
    """OpenAI function/tool schema."""

    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class ToolDefinition(BaseModel):
    """Tool definition wrapper."""

    type: Literal["function"] = "function"
    function: FunctionDefinition


class ToolFunctionCall(BaseModel):
    """Concrete function invocation."""

    name: str
    arguments: str


class ToolCall(BaseModel):
    """Tool call emitted by the assistant."""

    id: str
    type: Literal["function"] = "function"
    function: ToolFunctionCall


MessageContent = Optional[Union[str, List[Dict[str, Any]]]]


class ChatMessage(BaseModel):
    """OpenAI chat message."""

    role: Literal["system", "user", "assistant", "tool", "function"]
    content: MessageContent = None
    name: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class StreamOptions(BaseModel):
    include_usage: Optional[bool] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = Field(default=1.0, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    n: Optional[int] = Field(default=1, ge=1)
    stream: Optional[bool] = False
    stream_options: Optional[StreamOptions] = None
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = Field(default=None, gt=0)
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    tools: Optional[List[ToolDefinition]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    response_format: Optional[Dict[str, Any]] = None
    seed: Optional[int] = None
    parallel_tool_calls: Optional[bool] = None
    metadata: Optional[Dict[str, Any]] = None


class ChoiceLogprobs(BaseModel):
    content: Optional[List[Dict[str, Any]]] = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None
    logprobs: Optional[ChoiceLogprobs] = None


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: Optional[int] = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Optional[Usage] = None
    system_fingerprint: Optional[str] = None


class ModelPermission(BaseModel):
    id: str
    object: str = "model_permission"
    created: int
    allow_create_engine: bool = True
    allow_sampling: bool = True
    allow_logprobs: bool = True
    allow_search_indices: bool = False
    allow_view: bool = True
    allow_fine_tuning: bool = False
    organization: str = "*"
    group: Optional[str] = None
    is_blocking: bool = False


class Model(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str


class ListModelsResponse(BaseModel):
    object: str = "list"
    data: List[Model]
