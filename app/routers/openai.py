from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from app.models.openai import ChatCompletionRequest, ChatCompletionResponse, ListModelsResponse, Model
from app.auth import verify_client_api_key, CLIENT_API_KEY
from app.core.model_resolver import resolve_model
from app.routing.router import call_with_fallback
from app.routing.models import ResolvedRoute, RoutingError
from app.providers.openai_provider import OpenAIProvider
from app.providers.anthropic_provider import AnthropicProvider
from app.providers.azure_provider import AzureProvider
from app.core.format_converters import (
    anthropic_to_openai_request,
    anthropic_to_openai_response,
    openai_to_anthropic_request
)
from app.core.logging import (
    hash_api_key,
    extract_parameters_from_request,
    extract_usage_from_response,
    extract_response_content
)
from app.core.error_formatters import create_provider_error_response, format_openai_error
from app.database import logging_crud
from sqlalchemy.orm import Session
from app.database import crud
from app.database.database import SessionLocal
import json
import time

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

router = APIRouter()

# Route execution functions for fallback routing
async def _execute_openai_route(
    resolved_route: ResolvedRoute,
    messages: list,
    temperature: float = None,
    top_p: float = None,
    n: int = None,
    stream: bool = False,
    stop: list = None,
    max_tokens: int = None,
    presence_penalty: float = None,
    frequency_penalty: float = None,
    logit_bias: dict = None,
    user: str = None,
    tools: list = None,
    tool_choice: str = None,
) -> dict:
    """Execute an OpenAI-compatible route."""
    provider_name = resolved_route.provider

    # Get the appropriate provider instance
    if provider_name == "openai":
        provider = openai_provider
    elif provider_name == "nahcrof":
        provider = nahcrof_provider
    elif provider_name == "groq":
        provider = groq_provider
    elif provider_name == "cerebras":
        provider = cerebras_provider
    elif provider_name == "llama":
        provider = llama_provider
    elif provider_name == "mistral":
        provider = mistral_provider
    elif provider_name == "cloudflare":
        provider = cloudflare_provider
    elif provider_name == "gemini":
        provider = gemini_provider
    elif provider_name == "chutes":
        provider = chutes_provider
    elif provider_name == "longcat":
        provider = longcat_provider
    elif provider_name == "github":
        provider = azure_provider_github
    elif provider_name == "azure":
        provider = azure_provider_azure
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported provider in routing: {provider_name}"
        )

    # Set provider's base URL and API key if specified in route
    original_base_url = getattr(provider, '_base_url', None)
    original_api_key = getattr(provider, '_api_key', None)

    try:
        if resolved_route.base_url:
            provider._base_url = resolved_route.base_url
        if resolved_route.api_key:
            provider._api_key = resolved_route.api_key

        # Execute the call
        return await provider.call(
            model=resolved_route.model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            n=n,
            stream=stream,
            stop=stop,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            user=user,
            tools=tools,
            tool_choice=tool_choice,
        )
    finally:
        # Restore original settings
        if original_base_url is not None:
            provider._base_url = original_base_url
        if original_api_key is not None:
            provider._api_key = original_api_key


async def _execute_openai_route_stream(
    resolved_route: ResolvedRoute,
    messages: list,
    temperature: float = None,
    top_p: float = None,
    n: int = None,
    stop: list = None,
    max_tokens: int = None,
    presence_penalty: float = None,
    frequency_penalty: float = None,
    logit_bias: dict = None,
    user: str = None,
    tools: list = None,
    tool_choice: str = None,
) -> dict:
    """Execute an OpenAI-compatible streaming route."""
    provider_name = resolved_route.provider

    # Get the appropriate provider instance
    if provider_name == "openai":
        provider = openai_provider
    elif provider_name == "nahcrof":
        provider = nahcrof_provider
    elif provider_name == "groq":
        provider = groq_provider
    elif provider_name == "cerebras":
        provider = cerebras_provider
    elif provider_name == "llama":
        provider = llama_provider
    elif provider_name == "mistral":
        provider = mistral_provider
    elif provider_name == "cloudflare":
        provider = cloudflare_provider
    elif provider_name == "gemini":
        provider = gemini_provider
    elif provider_name == "chutes":
        provider = chutes_provider
    elif provider_name == "longcat":
        provider = longcat_provider
    elif provider_name == "github":
        provider = azure_provider_github
    elif provider_name == "azure":
        provider = azure_provider_azure
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported provider in routing: {provider_name}"
        )

    # Set provider's base URL and API key if specified in route
    original_base_url = getattr(provider, '_base_url', None)
    original_api_key = getattr(provider, '_api_key', None)

    try:
        if resolved_route.base_url:
            provider._base_url = resolved_route.base_url
        if resolved_route.api_key:
            provider._api_key = resolved_route.api_key

        # Execute the streaming call - return the async generator
        async for chunk in provider.call_stream(
            model=resolved_route.model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            n=n,
            stop=stop,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            user=user,
            tools=tools,
            tool_choice=tool_choice,
        ):
            yield chunk
    finally:
        # Restore original settings
        if original_base_url is not None:
            provider._base_url = original_base_url
        if original_api_key is not None:
            provider._api_key = original_api_key

# Initialize providers
openai_provider = OpenAIProvider("openai")
nahcrof_provider = OpenAIProvider("nahcrof")
groq_provider = OpenAIProvider("groq")
cerebras_provider = OpenAIProvider("cerebras")
llama_provider = OpenAIProvider("llama")
mistral_provider = OpenAIProvider("mistral")
cloudflare_provider = OpenAIProvider("cloudflare")
gemini_provider = OpenAIProvider("gemini")
chutes_provider = OpenAIProvider("chutes")
longcat_provider = OpenAIProvider("longcat")
anthropic_provider = AnthropicProvider()
azure_provider_github = AzureProvider("github")
azure_provider_azure = AzureProvider("azure")


@router.get("/v1/models", response_model=ListModelsResponse)
async def list_models(
    _: bool = Depends(verify_client_api_key)
):
    """
    List all available models from all providers.
    Returns models in OpenAI-compatible format.
    """
    from app.core.model_resolver import get_available_models
    import time

    # Get all available models
    model_names = get_available_models()

    # Convert to OpenAI model format
    models = []
    for model_name in model_names:
        try:
            config = resolve_model(model_name)
            provider = config['provider']

            model = Model(
                id=model_name,
                created=int(time.time()),  # Use current time as creation time
                owned_by=provider
            )
            models.append(model)
        except Exception:
            # Skip models that can't be resolved
            continue

    return ListModelsResponse(data=models)


@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest,
    http_request: Request,
    db: Session = Depends(get_db),
    _: bool = Depends(verify_client_api_key)
):
    """
    OpenAI-compatible chat completions endpoint.
    Routes requests to appropriate provider based on model configuration.
    """
    request_id = getattr(http_request.state, "request_id", None)
    start_time = getattr(http_request.state, "start_time", time.time())
    client_api_key_hash = hash_api_key(CLIENT_API_KEY) if CLIENT_API_KEY else None
    
    try:
        # Check if this model uses the new fallback routing system
        from app.routing.config_loader import config_loader
        try:
            routing_config = config_loader.load_config(request.model)
            use_fallback_routing = True
            logical_model = request.model
        except (FileNotFoundError, ValueError):
            # Fall back to legacy routing
            use_fallback_routing = False
            model_config = resolve_model(request.model)
            provider_name = model_config["provider"]
            provider_model = model_config["provider_model"]

        # Prepare request dict
        request_dict = request.model_dump(exclude_none=True)
        is_stream = bool(request_dict.get("stream", False))

        # Extract parameters for logging
        parameters = extract_parameters_from_request(request_dict)

        # Create initial log entry
        if request_id:
            if use_fallback_routing:
                logging_crud.create_request_log(
                    db=db,
                    request_id=request_id,
                    endpoint="/v1/chat/completions",
                    method="POST",
                    requested_model=request.model,
                    resolved_provider="fallback_router",
                    resolved_model=logical_model,
                    parameters=parameters,
                    messages=request_dict["messages"],
                    client_api_key_hash=client_api_key_hash,
                    is_streaming=is_stream
                )
            else:
                logging_crud.create_request_log(
                    db=db,
                    request_id=request_id,
                    endpoint="/v1/chat/completions",
                    method="POST",
                    requested_model=request.model,
                    resolved_provider=provider_name,
                    resolved_model=provider_model,
                    parameters=parameters,
                    messages=request_dict["messages"],
                    client_api_key_hash=client_api_key_hash,
                    is_streaming=is_stream
                )

        # If streaming requested, produce an SSE stream on this same endpoint (SDK-compatible)
        if is_stream:
            async def generate_stream():
                try:
                    if use_fallback_routing:
                        # Use fallback routing for streaming
                        async def exec_stream_route(resolved_route: ResolvedRoute):
                            # For streaming, we need to return the async generator
                            return _execute_openai_route_stream(
                                resolved_route=resolved_route,
                                messages=request_dict["messages"],
                                temperature=request_dict.get("temperature"),
                                top_p=request_dict.get("top_p"),
                                n=request_dict.get("n"),
                                stop=request_dict.get("stop"),
                                max_tokens=request_dict.get("max_tokens"),
                                presence_penalty=request_dict.get("presence_penalty"),
                                frequency_penalty=request_dict.get("frequency_penalty"),
                                logit_bias=request_dict.get("logit_bias"),
                                user=request_dict.get("user"),
                                tools=request_dict.get("tools"),
                                tool_choice=request_dict.get("tool_choice"),
                            )

                        # This will yield chunks from the successful route
                        async for chunk in await call_with_fallback(logical_model, exec_stream_route):
                            yield chunk
                    else:
                        # Legacy streaming logic
                        if provider_name == "openai":
                            async for chunk in openai_provider.call_stream(
                                model=provider_model,
                                messages=request_dict["messages"],
                                temperature=request_dict.get("temperature"),
                                top_p=request_dict.get("top_p"),
                                n=request_dict.get("n"),
                                stop=request_dict.get("stop"),
                                max_tokens=request_dict.get("max_tokens"),
                                presence_penalty=request_dict.get("presence_penalty"),
                                frequency_penalty=request_dict.get("frequency_penalty"),
                                logit_bias=request_dict.get("logit_bias"),
                                user=request_dict.get("user"),
                                tools=request_dict.get("tools"),
                                tool_choice=request_dict.get("tool_choice"),
                            ):
                                yield chunk
                        elif provider_name == "anthropic":
                            anthropic_request = openai_to_anthropic_request(request_dict)
                            async for chunk in anthropic_provider.call_stream(
                                model=provider_model,
                                messages=anthropic_request["messages"],
                                max_tokens=anthropic_request["max_tokens"],
                                temperature=anthropic_request.get("temperature"),
                                top_p=anthropic_request.get("top_p"),
                                tools=anthropic_request.get("tools"),
                            ):
                                yield chunk
                        elif provider_name == "github":
                            async for chunk in azure_provider_github.call_stream(
                                model=provider_model,
                                messages=request_dict["messages"],
                                temperature=request_dict.get("temperature"),
                                top_p=request_dict.get("top_p"),
                                n=request_dict.get("n"),
                                stop=request_dict.get("stop"),
                                max_tokens=request_dict.get("max_tokens"),
                                presence_penalty=request_dict.get("presence_penalty"),
                                frequency_penalty=request_dict.get("frequency_penalty"),
                                logit_bias=request_dict.get("logit_bias"),
                                user=request_dict.get("user"),
                                tools=request_dict.get("tools"),
                                tool_choice=request_dict.get("tool_choice"),
                            ):
                                yield chunk
                        elif provider_name == "azure":
                            async for chunk in azure_provider_azure.call_stream(
                                model=provider_model,
                                messages=request_dict["messages"],
                                temperature=request_dict.get("temperature"),
                                top_p=request_dict.get("top_p"),
                                n=request_dict.get("n"),
                                stop=request_dict.get("stop"),
                                max_tokens=request_dict.get("max_tokens"),
                                presence_penalty=request_dict.get("presence_penalty"),
                                frequency_penalty=request_dict.get("frequency_penalty"),
                                logit_bias=request_dict.get("logit_bias"),
                                user=request_dict.get("user"),
                                tools=request_dict.get("tools"),
                                tool_choice=request_dict.get("tool_choice"),
                            ):
                                yield chunk
                        elif provider_name == "nahcrof":
                            async for chunk in nahcrof_provider.call_stream(
                                model=provider_model,
                                messages=request_dict["messages"],
                                temperature=request_dict.get("temperature"),
                                top_p=request_dict.get("top_p"),
                                n=request_dict.get("n"),
                                stop=request_dict.get("stop"),
                                max_tokens=request_dict.get("max_tokens"),
                                presence_penalty=request_dict.get("presence_penalty"),
                                frequency_penalty=request_dict.get("frequency_penalty"),
                                logit_bias=request_dict.get("logit_bias"),
                                user=request_dict.get("user"),
                                tools=request_dict.get("tools"),
                                tool_choice=request_dict.get("tool_choice"),
                            ):
                                yield chunk
                        elif provider_name == "groq":
                            async for chunk in groq_provider.call_stream(
                                model=provider_model,
                                messages=request_dict["messages"],
                                temperature=request_dict.get("temperature"),
                                top_p=request_dict.get("top_p"),
                                n=request_dict.get("n"),
                                stop=request_dict.get("stop"),
                                max_tokens=request_dict.get("max_tokens"),
                                presence_penalty=request_dict.get("presence_penalty"),
                                frequency_penalty=request_dict.get("frequency_penalty"),
                                logit_bias=request_dict.get("logit_bias"),
                                user=request_dict.get("user"),
                                tools=request_dict.get("tools"),
                                tool_choice=request_dict.get("tool_choice"),
                            ):
                                yield chunk
                        elif provider_name == "cerebras":
                            async for chunk in cerebras_provider.call_stream(
                                model=provider_model,
                                messages=request_dict["messages"],
                                temperature=request_dict.get("temperature"),
                                top_p=request_dict.get("top_p"),
                                n=request_dict.get("n"),
                                stop=request_dict.get("stop"),
                                max_tokens=request_dict.get("max_tokens"),
                                presence_penalty=request_dict.get("presence_penalty"),
                                frequency_penalty=request_dict.get("frequency_penalty"),
                                logit_bias=request_dict.get("logit_bias"),
                                user=request_dict.get("user"),
                                tools=request_dict.get("tools"),
                                tool_choice=request_dict.get("tool_choice"),
                            ):
                                yield chunk
                        elif provider_name in ["llama", "mistral", "cloudflare", "gemini", "chutes", "longcat"]:
                            provider_map = {
                                "llama": llama_provider,
                                "mistral": mistral_provider,
                                "cloudflare": cloudflare_provider,
                                "gemini": gemini_provider,
                                "chutes": chutes_provider,
                                "longcat": longcat_provider
                            }
                            provider_instance = provider_map[provider_name]
                            async for chunk in provider_instance.call_stream(
                                model=provider_model,
                                messages=request_dict["messages"],
                                temperature=request_dict.get("temperature"),
                                top_p=request_dict.get("top_p"),
                                n=request_dict.get("n"),
                                stop=request_dict.get("stop"),
                                max_tokens=request_dict.get("max_tokens"),
                                presence_penalty=request_dict.get("presence_penalty"),
                                frequency_penalty=request_dict.get("frequency_penalty"),
                                logit_bias=request_dict.get("logit_bias"),
                                user=request_dict.get("user"),
                                tools=request_dict.get("tools"),
                                tool_choice=request_dict.get("tool_choice"),
                            ):
                                yield chunk
                        else:
                            err = create_provider_error_response(
                                "openai",
                                400,
                                f"Unsupported provider: {provider_name}",
                                "invalid_request_error"
                            )
                            raise err
                except HTTPException:
                    raise
                except Exception as e:
                    error_response = format_openai_error(500, f"Streaming error: {str(e)}", "internal_server_error")
                    yield f"data: {json.dumps(error_response)}\n\n"
                    yield "data: [DONE]\n\n"
                    return

            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                    "Connection": "keep-alive",
                },
            )
        
        # Route using fallback routing or legacy routing
        if use_fallback_routing:
            # Use new fallback routing system
            def exec_route(resolved_route: ResolvedRoute):
                return _execute_openai_route(
                    resolved_route=resolved_route,
                    messages=request_dict["messages"],
                    temperature=request_dict.get("temperature"),
                    top_p=request_dict.get("top_p"),
                    n=request_dict.get("n"),
                    stream=request_dict.get("stream", False),
                    stop=request_dict.get("stop"),
                    max_tokens=request_dict.get("max_tokens"),
                    presence_penalty=request_dict.get("presence_penalty"),
                    frequency_penalty=request_dict.get("frequency_penalty"),
                    logit_bias=request_dict.get("logit_bias"),
                    user=request_dict.get("user"),
                    tools=request_dict.get("tools"),
                    tool_choice=request_dict.get("tool_choice"),
                )

            response_data = await call_with_fallback(logical_model, exec_route)
            # Update model name to requested model
            response_data["model"] = request.model
        else:
            # Legacy routing logic
            if provider_name == "openai":
                # Direct OpenAI call
                response_data = await openai_provider.call(
                    model=provider_model,
                    messages=request_dict["messages"],
                    temperature=request_dict.get("temperature"),
                    top_p=request_dict.get("top_p"),
                    n=request_dict.get("n"),
                    stream=request_dict.get("stream", False),
                    stop=request_dict.get("stop"),
                    max_tokens=request_dict.get("max_tokens"),
                    presence_penalty=request_dict.get("presence_penalty"),
                    frequency_penalty=request_dict.get("frequency_penalty"),
                    logit_bias=request_dict.get("logit_bias"),
                    user=request_dict.get("user"),
                    tools=request_dict.get("tools"),
                    tool_choice=request_dict.get("tool_choice"),
                )
                # Update model name to requested model
                response_data["model"] = request.model

            elif provider_name == "anthropic":
                # Convert to Anthropic format, call, convert back
                anthropic_request = openai_to_anthropic_request(request_dict)
                anthropic_response = await anthropic_provider.call(
                    model=provider_model,
                    messages=anthropic_request["messages"],
                    max_tokens=anthropic_request["max_tokens"],
                    stream=anthropic_request.get("stream", False),
                    temperature=anthropic_request.get("temperature"),
                    top_p=anthropic_request.get("top_p"),
                    tools=anthropic_request.get("tools"),
                )
                # Convert back to OpenAI format
                response_data = anthropic_to_openai_response(anthropic_response, request.model)

            elif provider_name == "github":
                # GitHub Models uses Azure provider
                response_data = await azure_provider_github.call(
                    model=provider_model,
                    messages=request_dict["messages"],
                    temperature=request_dict.get("temperature"),
                    top_p=request_dict.get("top_p"),
                    n=request_dict.get("n"),
                    stream=request_dict.get("stream", False),
                    stop=request_dict.get("stop"),
                    max_tokens=request_dict.get("max_tokens"),
                    presence_penalty=request_dict.get("presence_penalty"),
                    frequency_penalty=request_dict.get("frequency_penalty"),
                    logit_bias=request_dict.get("logit_bias"),
                    user=request_dict.get("user"),
                    tools=request_dict.get("tools"),
                    tool_choice=request_dict.get("tool_choice"),
                )
                # Update model name to requested model
                response_data["model"] = request.model

            elif provider_name == "azure":
                # Azure AI uses Azure provider
                response_data = await azure_provider_azure.call(
                    model=provider_model,
                    messages=request_dict["messages"],
                    temperature=request_dict.get("temperature"),
                    top_p=request_dict.get("top_p"),
                    n=request_dict.get("n"),
                    stream=request_dict.get("stream", False),
                    stop=request_dict.get("stop"),
                    max_tokens=request_dict.get("max_tokens"),
                    presence_penalty=request_dict.get("presence_penalty"),
                    frequency_penalty=request_dict.get("frequency_penalty"),
                    logit_bias=request_dict.get("logit_bias"),
                    user=request_dict.get("user"),
                    tools=request_dict.get("tools"),
                    tool_choice=request_dict.get("tool_choice"),
                )
                # Update model name to requested model
                response_data["model"] = request.model

            elif provider_name == "nahcrof":
                # Nahcrof provider
                response_data = await nahcrof_provider.call(
                    model=provider_model,
                    messages=request_dict["messages"],
                    temperature=request_dict.get("temperature"),
                    top_p=request_dict.get("top_p"),
                    n=request_dict.get("n"),
                    stream=request_dict.get("stream", False),
                    stop=request_dict.get("stop"),
                    max_tokens=request_dict.get("max_tokens"),
                    presence_penalty=request_dict.get("presence_penalty"),
                    frequency_penalty=request_dict.get("frequency_penalty"),
                    logit_bias=request_dict.get("logit_bias"),
                    user=request_dict.get("user"),
                    tools=request_dict.get("tools"),
                    tool_choice=request_dict.get("tool_choice"),
                )
                # Update model name to requested model
                response_data["model"] = request.model

            elif provider_name == "groq":
                # Groq provider
                response_data = await groq_provider.call(
                    model=provider_model,
                    messages=request_dict["messages"],
                    temperature=request_dict.get("temperature"),
                    top_p=request_dict.get("top_p"),
                    n=request_dict.get("n"),
                    stream=request_dict.get("stream", False),
                    stop=request_dict.get("stop"),
                    max_tokens=request_dict.get("max_tokens"),
                    presence_penalty=request_dict.get("presence_penalty"),
                    frequency_penalty=request_dict.get("frequency_penalty"),
                    logit_bias=request_dict.get("logit_bias"),
                    user=request_dict.get("user"),
                    tools=request_dict.get("tools"),
                    tool_choice=request_dict.get("tool_choice"),
                )
                # Update model name to requested model
                response_data["model"] = request.model

            elif provider_name == "cerebras":
                # Cerebras provider
                response_data = await cerebras_provider.call(
                    model=provider_model,
                    messages=request_dict["messages"],
                    temperature=request_dict.get("temperature"),
                    top_p=request_dict.get("top_p"),
                    n=request_dict.get("n"),
                    stream=request_dict.get("stream", False),
                    stop=request_dict.get("stop"),
                    max_tokens=request_dict.get("max_tokens"),
                    presence_penalty=request_dict.get("presence_penalty"),
                    frequency_penalty=request_dict.get("frequency_penalty"),
                    logit_bias=request_dict.get("logit_bias"),
                    user=request_dict.get("user"),
                    tools=request_dict.get("tools"),
                    tool_choice=request_dict.get("tool_choice"),
                )
                # Update model name to requested model
                response_data["model"] = request.model

            elif provider_name in ["llama", "mistral", "cloudflare", "gemini", "chutes", "longcat"]:
                # Other OpenAI-compatible providers
                provider_map = {
                    "llama": llama_provider,
                    "mistral": mistral_provider,
                    "cloudflare": cloudflare_provider,
                    "gemini": gemini_provider,
                    "chutes": chutes_provider,
                    "longcat": longcat_provider
                }
                provider_instance = provider_map[provider_name]
                response_data = await provider_instance.call(
                    model=provider_model,
                    messages=request_dict["messages"],
                    temperature=request_dict.get("temperature"),
                    top_p=request_dict.get("top_p"),
                    n=request_dict.get("n"),
                    stream=request_dict.get("stream", False),
                    stop=request_dict.get("stop"),
                    max_tokens=request_dict.get("max_tokens"),
                    presence_penalty=request_dict.get("presence_penalty"),
                    frequency_penalty=request_dict.get("frequency_penalty"),
                    logit_bias=request_dict.get("logit_bias"),
                    user=request_dict.get("user"),
                    tools=request_dict.get("tools"),
                    tool_choice=request_dict.get("tool_choice"),
                )
                # Update model name to requested model
                response_data["model"] = request.model
            else:
                raise create_provider_error_response(
                    "openai",
                    400,
                    f"Unsupported provider: {provider_name}",
                    "invalid_request_error"
                )
        
        # Calculate response time
        response_time_ms = int((time.time() - start_time) * 1000)
        
        # Extract response content and usage
        response_content = extract_response_content(response_data, is_openai_format=True)
        usage = extract_usage_from_response(response_data)
        
        # Update log with response
        if request_id:
            try:
                logging_crud.update_request_log(
                    db=db,
                    request_id=request_id,
                    response_status=200,
                    response_time_ms=response_time_ms,
                    response_content=response_content,
                    response_usage=usage
                )
            except Exception as log_error:
                # Don't fail the request if logging fails
                print(f"Failed to update request log: {log_error}")
        
        # Legacy logging (keep for backward compatibility)
        try:
            crud.create_log(db, service=provider_name, request=request_dict, response=response_data)
        except Exception:
            pass  # Ignore legacy logging errors
        
        return response_data
        
    except HTTPException as e:
        # Log error
        if request_id:
            try:
                response_time_ms = int((time.time() - start_time) * 1000)
                error_detail = e.detail
                if isinstance(error_detail, dict) and "error" in error_detail:
                    error_message = error_detail["error"].get("message", str(error_detail))
                else:
                    error_message = str(error_detail)
                
                logging_crud.update_request_log(
                    db=db,
                    request_id=request_id,
                    response_status=e.status_code,
                    response_time_ms=response_time_ms,
                    error_message=error_message,
                    error_type=type(e).__name__
                )
            except Exception:
                pass  # Don't fail if logging fails
        raise
    except Exception as e:
        # Log error
        if request_id:
            try:
                response_time_ms = int((time.time() - start_time) * 1000)
                logging_crud.update_request_log(
                    db=db,
                    request_id=request_id,
                    response_status=500,
                    response_time_ms=response_time_ms,
                    error_message=str(e),
                    error_type=type(e).__name__
                )
            except Exception:
                pass  # Don't fail if logging fails
        
        raise create_provider_error_response(
            "openai",
            500,
            f"Error processing request: {str(e)}",
            "internal_server_error"
        )


@router.post("/v1/chat/completions-stream")
async def chat_completions_stream(
    request: ChatCompletionRequest,
    http_request: Request,
    db: Session = Depends(get_db),
    _: bool = Depends(verify_client_api_key)
):
    """
    OpenAI-compatible streaming chat completions endpoint.
    Routes requests to appropriate provider based on model configuration.
    """
    request_id = getattr(http_request.state, "request_id", None)
    start_time = getattr(http_request.state, "start_time", time.time())
    client_api_key_hash = hash_api_key(CLIENT_API_KEY) if CLIENT_API_KEY else None
    
    try:
        # Resolve model to provider
        model_config = resolve_model(request.model)
        provider_name = model_config["provider"]
        provider_model = model_config["provider_model"]
        
        # Prepare request dict
        request_dict = request.model_dump(exclude_none=True)
        request_dict["model"] = provider_model
        request_dict["stream"] = True
        
        # Extract parameters for logging
        parameters = extract_parameters_from_request(request_dict)
        
        # Create initial log entry for streaming
        if request_id:
            logging_crud.create_request_log(
                db=db,
                request_id=request_id,
                endpoint="/v1/chat/completions-stream",
                method="POST",
                requested_model=request.model,
                resolved_provider=provider_name,
                resolved_model=provider_model,
                parameters=parameters,
                messages=request_dict["messages"],
                client_api_key_hash=client_api_key_hash,
                is_streaming=True
            )
        
        async def generate_stream():
            stream_error = None
            try:
                if provider_name == "openai":
                    # Direct OpenAI streaming
                    async for chunk in openai_provider.call_stream(
                        model=provider_model,
                        messages=request_dict["messages"],
                        temperature=request_dict.get("temperature"),
                        top_p=request_dict.get("top_p"),
                        n=request_dict.get("n"),
                        stop=request_dict.get("stop"),
                        max_tokens=request_dict.get("max_tokens"),
                        presence_penalty=request_dict.get("presence_penalty"),
                        frequency_penalty=request_dict.get("frequency_penalty"),
                        logit_bias=request_dict.get("logit_bias"),
                        user=request_dict.get("user"),
                        tools=request_dict.get("tools"),
                        tool_choice=request_dict.get("tool_choice"),
                    ):
                        yield chunk

                elif provider_name == "anthropic":
                    # Convert to Anthropic format, stream, pass through (SSE format compatible)
                    anthropic_request = openai_to_anthropic_request(request_dict)
                    async for chunk in anthropic_provider.call_stream(
                        model=provider_model,
                        messages=anthropic_request["messages"],
                        max_tokens=anthropic_request["max_tokens"],
                        temperature=anthropic_request.get("temperature"),
                        top_p=anthropic_request.get("top_p"),
                        tools=anthropic_request.get("tools"),
                    ):
                        yield chunk

                elif provider_name == "github":
                    # GitHub Models uses Azure provider
                    async for chunk in azure_provider_github.call_stream(
                        model=provider_model,
                        messages=request_dict["messages"],
                        temperature=request_dict.get("temperature"),
                        top_p=request_dict.get("top_p"),
                        n=request_dict.get("n"),
                        stop=request_dict.get("stop"),
                        max_tokens=request_dict.get("max_tokens"),
                        presence_penalty=request_dict.get("presence_penalty"),
                        frequency_penalty=request_dict.get("frequency_penalty"),
                        logit_bias=request_dict.get("logit_bias"),
                        user=request_dict.get("user"),
                        tools=request_dict.get("tools"),
                        tool_choice=request_dict.get("tool_choice"),
                    ):
                        yield chunk

                elif provider_name == "azure":
                    # Azure AI uses Azure provider
                    async for chunk in azure_provider_azure.call_stream(
                        model=provider_model,
                        messages=request_dict["messages"],
                        temperature=request_dict.get("temperature"),
                        top_p=request_dict.get("top_p"),
                        n=request_dict.get("n"),
                        stop=request_dict.get("stop"),
                        max_tokens=request_dict.get("max_tokens"),
                        presence_penalty=request_dict.get("presence_penalty"),
                        frequency_penalty=request_dict.get("frequency_penalty"),
                        logit_bias=request_dict.get("logit_bias"),
                        user=request_dict.get("user"),
                        tools=request_dict.get("tools"),
                        tool_choice=request_dict.get("tool_choice"),
                    ):
                        yield chunk

                elif provider_name == "nahcrof":
                    # Nahcrof provider streaming
                    async for chunk in nahcrof_provider.call_stream(
                        model=provider_model,
                        messages=request_dict["messages"],
                        temperature=request_dict.get("temperature"),
                        top_p=request_dict.get("top_p"),
                        n=request_dict.get("n"),
                        stop=request_dict.get("stop"),
                        max_tokens=request_dict.get("max_tokens"),
                        presence_penalty=request_dict.get("presence_penalty"),
                        frequency_penalty=request_dict.get("frequency_penalty"),
                        logit_bias=request_dict.get("logit_bias"),
                        user=request_dict.get("user"),
                        tools=request_dict.get("tools"),
                        tool_choice=request_dict.get("tool_choice"),
                    ):
                        yield chunk

                elif provider_name == "groq":
                    # Groq provider streaming
                    async for chunk in groq_provider.call_stream(
                        model=provider_model,
                        messages=request_dict["messages"],
                        temperature=request_dict.get("temperature"),
                        top_p=request_dict.get("top_p"),
                        n=request_dict.get("n"),
                        stop=request_dict.get("stop"),
                        max_tokens=request_dict.get("max_tokens"),
                        presence_penalty=request_dict.get("presence_penalty"),
                        frequency_penalty=request_dict.get("frequency_penalty"),
                        logit_bias=request_dict.get("logit_bias"),
                        user=request_dict.get("user"),
                        tools=request_dict.get("tools"),
                        tool_choice=request_dict.get("tool_choice"),
                    ):
                        yield chunk

                elif provider_name == "cerebras":
                    # Cerebras provider streaming
                    async for chunk in cerebras_provider.call_stream(
                        model=provider_model,
                        messages=request_dict["messages"],
                        temperature=request_dict.get("temperature"),
                        top_p=request_dict.get("top_p"),
                        n=request_dict.get("n"),
                        stop=request_dict.get("stop"),
                        max_tokens=request_dict.get("max_tokens"),
                        presence_penalty=request_dict.get("presence_penalty"),
                        frequency_penalty=request_dict.get("frequency_penalty"),
                        logit_bias=request_dict.get("logit_bias"),
                        user=request_dict.get("user"),
                        tools=request_dict.get("tools"),
                        tool_choice=request_dict.get("tool_choice"),
                    ):
                        yield chunk

                elif provider_name in ["llama", "mistral", "cloudflare", "gemini", "chutes", "longcat"]:
                    # Other OpenAI-compatible providers streaming
                    provider_map = {
                        "llama": llama_provider,
                        "mistral": mistral_provider,
                        "cloudflare": cloudflare_provider,
                        "gemini": gemini_provider,
                        "chutes": chutes_provider,
                        "longcat": longcat_provider
                    }
                    provider_instance = provider_map[provider_name]
                    async for chunk in provider_instance.call_stream(
                        model=provider_model,
                        messages=request_dict["messages"],
                        temperature=request_dict.get("temperature"),
                        top_p=request_dict.get("top_p"),
                        n=request_dict.get("n"),
                        stop=request_dict.get("stop"),
                        max_tokens=request_dict.get("max_tokens"),
                        presence_penalty=request_dict.get("presence_penalty"),
                        frequency_penalty=request_dict.get("frequency_penalty"),
                        logit_bias=request_dict.get("logit_bias"),
                        user=request_dict.get("user"),
                        tools=request_dict.get("tools"),
                        tool_choice=request_dict.get("tool_choice"),
                    ):
                        yield chunk
                else:
                    stream_error = create_provider_error_response(
                        "openai",
                        400,
                        f"Unsupported provider: {provider_name}",
                        "invalid_request_error"
                    )
                    raise stream_error
                
                # Update log with success
                if request_id:
                    try:
                        response_time_ms = int((time.time() - start_time) * 1000)
                        logging_crud.update_request_log(
                            db=db,
                            request_id=request_id,
                            response_status=200,
                            response_time_ms=response_time_ms
                        )
                    except Exception:
                        pass  # Don't fail if logging fails
            except HTTPException:
                # Re-raise HTTPExceptions (they're already formatted)
                raise
            except Exception as e:
                # Log streaming error
                stream_error = e
                if request_id:
                    try:
                        response_time_ms = int((time.time() - start_time) * 1000)
                        logging_crud.update_request_log(
                            db=db,
                            request_id=request_id,
                            response_status=500,
                            response_time_ms=response_time_ms,
                            error_message=str(e),
                            error_type=type(e).__name__
                        )
                    except Exception:
                        pass  # Don't fail if logging fails
                
                # Format error as SSE event
                error_response = format_openai_error(500, f"Streaming error: {str(e)}", "internal_server_error")
                yield f"data: {json.dumps(error_response)}\n\n"
                yield "data: [DONE]\n\n"
                return
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )
        
    except HTTPException as e:
        # Log error
        if request_id:
            response_time_ms = int((time.time() - start_time) * 1000)
            logging_crud.update_request_log(
                db=db,
                request_id=request_id,
                response_status=e.status_code,
                response_time_ms=response_time_ms,
                error_message=str(e.detail),
                error_type=type(e).__name__
            )
        raise
    except Exception as e:
        # Log error
        if request_id:
            response_time_ms = int((time.time() - start_time) * 1000)
            logging_crud.update_request_log(
                db=db,
                request_id=request_id,
                response_status=500,
                response_time_ms=response_time_ms,
                error_message=str(e),
                error_type=type(e).__name__
            )
        raise HTTPException(
            status_code=500,
            detail=f"Error processing streaming request: {str(e)}"
        )


@router.post("/v1/chat/completions-stream", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest,
    db: Session = Depends(get_db),
    _: bool = Depends(verify_client_api_key)
):
    """
    OpenAI-compatible chat completions endpoint.
    Routes requests to appropriate provider based on model configuration.
    """
    try:
        # Resolve model to provider
        model_config = resolve_model(request.model)
        provider_name = model_config["provider"]
        provider_model = model_config["provider_model"]
        
        # Prepare request dict
        request_dict = request.model_dump(exclude_none=True)
        request_dict["model"] = provider_model
        
        # Route to appropriate provider
        if provider_name == "openai":
            # Direct OpenAI call
            response_data = await openai_provider.call(
                model=provider_model,
                messages=request_dict["messages"],
                temperature=request_dict.get("temperature"),
                top_p=request_dict.get("top_p"),
                n=request_dict.get("n"),
                stream=request_dict.get("stream", False),
                stop=request_dict.get("stop"),
                max_tokens=request_dict.get("max_tokens"),
                presence_penalty=request_dict.get("presence_penalty"),
                frequency_penalty=request_dict.get("frequency_penalty"),
                logit_bias=request_dict.get("logit_bias"),
                user=request_dict.get("user"),
                tools=request_dict.get("tools"),
                tool_choice=request_dict.get("tool_choice"),
            )
            # Update model name to requested model
            response_data["model"] = request.model
            
        elif provider_name == "anthropic":
            # Convert to Anthropic format, call, convert back
            anthropic_request = openai_to_anthropic_request(request_dict)
            anthropic_response = await anthropic_provider.call(
                model=provider_model,
                messages=anthropic_request["messages"],
                max_tokens=anthropic_request["max_tokens"],
                stream=anthropic_request.get("stream", False),
                temperature=anthropic_request.get("temperature"),
                top_p=anthropic_request.get("top_p"),
                tools=anthropic_request.get("tools"),
            )
            # Convert back to OpenAI format
            response_data = anthropic_to_openai_response(anthropic_response, request.model)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported provider: {provider_name}"
            )
        
        # Log request/response
        crud.create_log(db, service=provider_name, request=request_dict, response=response_data)
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )


@router.post("/v1/chat/completions-stream")
async def chat_completions_stream(
    request: ChatCompletionRequest,
    db: Session = Depends(get_db),
    _: bool = Depends(verify_client_api_key)
):
    """
    OpenAI-compatible streaming chat completions endpoint.
    Routes requests to appropriate provider based on model configuration.
    """
    try:
        # Resolve model to provider
        model_config = resolve_model(request.model)
        provider_name = model_config["provider"]
        provider_model = model_config["provider_model"]
        
        # Prepare request dict
        request_dict = request.model_dump(exclude_none=True)
        request_dict["model"] = provider_model
        request_dict["stream"] = True
        
        async def generate_stream():
            if provider_name == "openai":
                # Direct OpenAI streaming
                async for chunk in openai_provider.call_stream(
                    model=provider_model,
                    messages=request_dict["messages"],
                    temperature=request_dict.get("temperature"),
                    top_p=request_dict.get("top_p"),
                    n=request_dict.get("n"),
                    stop=request_dict.get("stop"),
                    max_tokens=request_dict.get("max_tokens"),
                    presence_penalty=request_dict.get("presence_penalty"),
                    frequency_penalty=request_dict.get("frequency_penalty"),
                    logit_bias=request_dict.get("logit_bias"),
                    user=request_dict.get("user"),
                    tools=request_dict.get("tools"),
                    tool_choice=request_dict.get("tool_choice"),
                ):
                    yield chunk
                    
            elif provider_name == "anthropic":
                # Convert to Anthropic format, stream, pass through (SSE format compatible)
                anthropic_request = openai_to_anthropic_request(request_dict)
                async for chunk in anthropic_provider.call_stream(
                    model=provider_model,
                    messages=anthropic_request["messages"],
                    max_tokens=anthropic_request["max_tokens"],
                    temperature=anthropic_request.get("temperature"),
                    top_p=anthropic_request.get("top_p"),
                    tools=anthropic_request.get("tools"),
                ):
                    yield chunk
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported provider: {provider_name}"
                )
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing streaming request: {str(e)}"
        )
