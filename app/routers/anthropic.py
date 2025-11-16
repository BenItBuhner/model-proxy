from fastapi import APIRouter, Depends, HTTPException, Request, Query
from fastapi.responses import StreamingResponse
from app.models.anthropic import AnthropicMessagesRequest, AnthropicMessagesResponse
from app.auth import verify_client_api_key, CLIENT_API_KEY
from app.core.model_resolver import resolve_model
from app.routing.router import call_with_fallback
from app.routing.models import ResolvedRoute, RoutingError
from app.providers.openai_provider import OpenAIProvider
from app.providers.anthropic_provider import AnthropicProvider
from app.providers.azure_provider import AzureProvider
from app.core.format_converters import (
    openai_to_anthropic_response,
    anthropic_to_openai_request
)
from app.core.logging import (
    hash_api_key,
    extract_parameters_from_request,
    extract_usage_from_response,
    extract_response_content
)
from app.core.error_formatters import create_provider_error_response
from app.database import logging_crud
from sqlalchemy.orm import Session
from app.database import crud
from app.database.database import SessionLocal
from typing import Any, Dict, List, Optional
import json
import logging
import time
import uuid
import math

logger = logging.getLogger("anthropic_router")


def _log_bad_request(endpoint: str, detail: Any, payload: Any) -> None:
    """Log 400-level failures with request payload for debugging."""
    try:
        serialized = json.dumps(payload, ensure_ascii=False)
    except Exception:
        serialized = str(payload)
    logger.warning(
        "Anthropic router 400 on %s: detail=%s payload=%s",
        endpoint,
        detail,
        serialized,
    )

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

router = APIRouter()

# Route execution functions for fallback routing
async def _execute_anthropic_route(
    resolved_route: ResolvedRoute,
    messages: list,
    max_tokens: int,
    stream: bool = False,
    temperature: float = None,
    top_p: float = None,
    top_k: int = None,
    tools: list = None,
) -> dict:
    """Execute an Anthropic-compatible route."""
    provider_name = resolved_route.provider

    if provider_name == "anthropic":
        provider = anthropic_provider
    else:
        # For non-Anthropic providers, we need to convert to OpenAI format and back
        provider = None
        openai_provider_instance = None

        # Get the appropriate OpenAI-compatible provider
        if provider_name == "openai":
            openai_provider_instance = openai_provider
        elif provider_name == "nahcrof":
            openai_provider_instance = nahcrof_provider
        elif provider_name == "groq":
            openai_provider_instance = groq_provider
        elif provider_name == "cerebras":
            openai_provider_instance = cerebras_provider
        elif provider_name == "llama":
            openai_provider_instance = llama_provider
        elif provider_name == "mistral":
            openai_provider_instance = mistral_provider
        elif provider_name == "cloudflare":
            openai_provider_instance = cloudflare_provider
        elif provider_name == "gemini":
            openai_provider_instance = gemini_provider
        elif provider_name == "chutes":
            openai_provider_instance = chutes_provider
        elif provider_name == "longcat":
            openai_provider_instance = longcat_provider
        elif provider_name == "github":
            openai_provider_instance = azure_provider_github
        elif provider_name == "azure":
            openai_provider_instance = azure_provider_azure
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported provider in Anthropic routing: {provider_name}"
            )

    # Set provider's base URL and API key if specified in route
    original_base_url = None
    original_api_key = None

    if provider:
        original_base_url = getattr(provider, '_base_url', None)
        original_api_key = getattr(provider, '_api_key', None)
    elif openai_provider_instance:
        original_base_url = getattr(openai_provider_instance, '_base_url', None)
        original_api_key = getattr(openai_provider_instance, '_api_key', None)

    try:
        if resolved_route.base_url:
            if provider:
                provider._base_url = resolved_route.base_url
            elif openai_provider_instance:
                openai_provider_instance._base_url = resolved_route.base_url
        if resolved_route.api_key:
            if provider:
                provider._api_key = resolved_route.api_key
            elif openai_provider_instance:
                openai_provider_instance._api_key = resolved_route.api_key

        if provider:
            # Direct Anthropic call
            return await provider.call(
                model=resolved_route.model,
                messages=messages,
                max_tokens=max_tokens,
                stream=stream,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                tools=tools,
            )
        else:
            # Convert to OpenAI format, call, convert back to Anthropic
            from app.core.format_converters import anthropic_to_openai_request, openai_to_anthropic_response
            anthropic_request = {
                "model": resolved_route.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "stream": stream,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "tools": tools,
            }
            openai_request = anthropic_to_openai_request(anthropic_request)
            openai_response = await openai_provider_instance.call(
                model=resolved_route.model,
                messages=openai_request["messages"],
                temperature=openai_request.get("temperature"),
                top_p=openai_request.get("top_p"),
                n=openai_request.get("n"),
                stream=openai_request.get("stream", False),
                stop=openai_request.get("stop"),
                max_tokens=openai_request.get("max_tokens"),
                presence_penalty=openai_request.get("presence_penalty"),
                frequency_penalty=openai_request.get("frequency_penalty"),
                logit_bias=openai_request.get("logit_bias"),
                user=openai_request.get("user"),
                tools=openai_request.get("tools"),
                tool_choice=openai_request.get("tool_choice"),
            )
            # Convert back to Anthropic format
            return openai_to_anthropic_response(openai_response, resolved_route.model)
    finally:
        # Restore original settings
        if original_base_url is not None:
            if provider:
                provider._base_url = original_base_url
            elif openai_provider_instance:
                openai_provider_instance._base_url = original_base_url
        if original_api_key is not None:
            if provider:
                provider._api_key = original_api_key
            elif openai_provider_instance:
                openai_provider_instance._api_key = original_api_key


async def _execute_anthropic_route_stream(
    resolved_route: ResolvedRoute,
    messages: list,
    max_tokens: int,
    temperature: float = None,
    top_p: float = None,
    top_k: int = None,
    tools: list = None,
) -> dict:
    """Execute an Anthropic-compatible streaming route."""
    provider_name = resolved_route.provider

    if provider_name == "anthropic":
        provider = anthropic_provider
    else:
        # For non-Anthropic providers, we need to convert to OpenAI format and back
        provider = None
        openai_provider_instance = None

        # Get the appropriate OpenAI-compatible provider
        if provider_name == "openai":
            openai_provider_instance = openai_provider
        elif provider_name == "nahcrof":
            openai_provider_instance = nahcrof_provider
        elif provider_name == "groq":
            openai_provider_instance = groq_provider
        elif provider_name == "cerebras":
            openai_provider_instance = cerebras_provider
        elif provider_name == "llama":
            openai_provider_instance = llama_provider
        elif provider_name == "mistral":
            openai_provider_instance = mistral_provider
        elif provider_name == "cloudflare":
            openai_provider_instance = cloudflare_provider
        elif provider_name == "gemini":
            openai_provider_instance = gemini_provider
        elif provider_name == "chutes":
            openai_provider_instance = chutes_provider
        elif provider_name == "longcat":
            openai_provider_instance = longcat_provider
        elif provider_name == "github":
            openai_provider_instance = azure_provider_github
        elif provider_name == "azure":
            openai_provider_instance = azure_provider_azure
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported provider in Anthropic routing: {provider_name}"
            )

    # Set provider's base URL and API key if specified in route
    original_base_url = None
    original_api_key = None

    if provider:
        original_base_url = getattr(provider, '_base_url', None)
        original_api_key = getattr(provider, '_api_key', None)
    elif openai_provider_instance:
        original_base_url = getattr(openai_provider_instance, '_base_url', None)
        original_api_key = getattr(openai_provider_instance, '_api_key', None)

    try:
        if resolved_route.base_url:
            if provider:
                provider._base_url = resolved_route.base_url
            elif openai_provider_instance:
                openai_provider_instance._base_url = resolved_route.base_url
        if resolved_route.api_key:
            if provider:
                provider._api_key = resolved_route.api_key
            elif openai_provider_instance:
                openai_provider_instance._api_key = resolved_route.api_key

        if provider:
            # Direct Anthropic streaming call
            async for chunk in provider.call_stream(
                model=resolved_route.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                tools=tools,
            ):
                yield chunk
        else:
            # For OpenAI-compatible providers, we need to convert formats in streaming
            # This is complex, so we'll use the existing streaming adapter logic
            from app.core.format_converters import anthropic_to_openai_request

            anthropic_request = {
                "model": resolved_route.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "stream": True,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "tools": tools,
            }
            openai_request = anthropic_to_openai_request(anthropic_request)

            # Use the streaming adapter from the _create_streaming_response function
            # This is a simplified version
            async for chunk in openai_provider_instance.call_stream(
                model=resolved_route.model,
                messages=openai_request["messages"],
                temperature=openai_request.get("temperature"),
                top_p=openai_request.get("top_p"),
                n=openai_request.get("n"),
                stop=openai_request.get("stop"),
                max_tokens=openai_request.get("max_tokens"),
                presence_penalty=openai_request.get("presence_penalty"),
                frequency_penalty=openai_request.get("frequency_penalty"),
                logit_bias=openai_request.get("logit_bias"),
                user=openai_request.get("user"),
                tools=openai_request.get("tools"),
                tool_choice=openai_request.get("tool_choice"),
            ):
                # For Anthropic streaming, we need to convert OpenAI SSE format to Anthropic SSE format
                # This is a simplified conversion - in practice, you'd need the full adapter
                yield chunk  # For now, pass through (this needs proper conversion)
    finally:
        # Restore original settings
        if original_base_url is not None:
            if provider:
                provider._base_url = original_base_url
            elif openai_provider_instance:
                openai_provider_instance._base_url = original_base_url
        if original_api_key is not None:
            if provider:
                provider._api_key = original_api_key
            elif openai_provider_instance:
                openai_provider_instance._api_key = original_api_key

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


@router.post("/v1/messages", response_model=AnthropicMessagesResponse)
async def messages(
    request: AnthropicMessagesRequest,
    http_request: Request,
    db: Session = Depends(get_db),
    _: bool = Depends(verify_client_api_key),
    beta: bool = Query(False, description="Enable beta (streaming) mode")
):
    """
    Anthropic-compatible messages endpoint.
    Routes requests to appropriate provider based on model configuration.
    """
    request_id = getattr(http_request.state, "request_id", None)
    start_time = getattr(http_request.state, "start_time", time.time())
    client_api_key_hash = hash_api_key(CLIENT_API_KEY) if CLIENT_API_KEY else None
    request_payload: Optional[Dict[str, Any]] = None
    request_payload: Optional[Dict[str, Any]] = None
    
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
        request_dict["stream"] = request_dict.get("stream", False) or beta
        request_payload = request_dict.copy()

        # Extract parameters for logging
        parameters = extract_parameters_from_request(request_dict)

        # Create initial log entry
        if request_id:
            if use_fallback_routing:
                logging_crud.create_request_log(
                    db=db,
                    request_id=request_id,
                    endpoint="/v1/messages",
                    method="POST",
                    requested_model=request.model,
                    resolved_provider="fallback_router",
                    resolved_model=logical_model,
                    parameters=parameters,
                    messages=request_dict["messages"],
                    client_api_key_hash=client_api_key_hash,
                    is_streaming=request_dict["stream"]
                )
            else:
                logging_crud.create_request_log(
                    db=db,
                    request_id=request_id,
                    endpoint="/v1/messages",
                    method="POST",
                    requested_model=request.model,
                    resolved_provider=provider_name,
                    resolved_model=provider_model,
                    parameters=parameters,
                    messages=request_dict["messages"],
                    client_api_key_hash=client_api_key_hash,
                    is_streaming=request_dict["stream"]
                )

        # If streaming requested, reuse streaming pipeline
        if request_dict["stream"]:
            return await _create_streaming_response(
                request_dict=request_dict,
                provider_name=provider_name,
                provider_model=provider_model,
                requested_model=request.model,
                request_id=request_id,
                start_time=start_time,
                db=db,
                client_api_key_hash=client_api_key_hash,
            )
        
        # Route using fallback routing or legacy routing
        if use_fallback_routing:
            # Use new fallback routing system
            def exec_route(resolved_route: ResolvedRoute):
                return _execute_anthropic_route(
                    resolved_route=resolved_route,
                    messages=request_dict["messages"],
                    max_tokens=request_dict["max_tokens"],
                    stream=request_dict.get("stream", False),
                    temperature=request_dict.get("temperature"),
                    top_p=request_dict.get("top_p"),
                    top_k=request_dict.get("top_k"),
                    tools=request_dict.get("tools"),
                )

            response_data = await call_with_fallback(logical_model, exec_route)
            # Update model name to requested model
            response_data["model"] = request.model
        else:
            # Legacy routing logic
            if provider_name == "anthropic":
                # Direct Anthropic call
                response_data = await anthropic_provider.call(
                    model=provider_model,
                    messages=request_dict["messages"],
                    max_tokens=request_dict["max_tokens"],
                    stream=request_dict.get("stream", False),
                    temperature=request_dict.get("temperature"),
                    top_p=request_dict.get("top_p"),
                    tools=request_dict.get("tools"),
                )
                # Update model name to requested model
                response_data["model"] = request.model

            elif provider_name == "openai":
                # Convert to OpenAI format, call, convert back
                openai_request = anthropic_to_openai_request(request_dict)
                openai_response = await openai_provider.call(
                    model=provider_model,
                    messages=openai_request["messages"],
                    temperature=openai_request.get("temperature"),
                    top_p=openai_request.get("top_p"),
                    n=openai_request.get("n"),
                    stream=openai_request.get("stream", False),
                    stop=openai_request.get("stop"),
                    max_tokens=openai_request.get("max_tokens"),
                    presence_penalty=openai_request.get("presence_penalty"),
                    frequency_penalty=openai_request.get("frequency_penalty"),
                    logit_bias=openai_request.get("logit_bias"),
                    user=openai_request.get("user"),
                    tools=openai_request.get("tools"),
                    tool_choice=openai_request.get("tool_choice"),
                )
                # Convert back to Anthropic format
                response_data = openai_to_anthropic_response(openai_response, request.model)

            elif provider_name == "nahcrof":
                # Nahcrof provider - convert to OpenAI format, call, convert back to Anthropic
                openai_request = anthropic_to_openai_request(request_dict)
                openai_response = await nahcrof_provider.call(
                    model=provider_model,
                    messages=openai_request["messages"],
                    temperature=openai_request.get("temperature"),
                    top_p=openai_request.get("top_p"),
                    n=openai_request.get("n"),
                    stream=openai_request.get("stream", False),
                    stop=openai_request.get("stop"),
                    max_tokens=openai_request.get("max_tokens"),
                    presence_penalty=openai_request.get("presence_penalty"),
                    frequency_penalty=openai_request.get("frequency_penalty"),
                    logit_bias=openai_request.get("logit_bias"),
                    user=openai_request.get("user"),
                    tools=openai_request.get("tools"),
                    tool_choice=openai_request.get("tool_choice"),
                )
                # Convert back to Anthropic format
                response_data = openai_to_anthropic_response(openai_response, request.model)

            elif provider_name == "groq":
                # Groq provider - convert to OpenAI format, call, convert back to Anthropic
                openai_request = anthropic_to_openai_request(request_dict)
                openai_response = await groq_provider.call(
                    model=provider_model,
                    messages=openai_request["messages"],
                    temperature=openai_request.get("temperature"),
                    top_p=openai_request.get("top_p"),
                    n=openai_request.get("n"),
                    stream=openai_request.get("stream", False),
                    stop=openai_request.get("stop"),
                    max_tokens=openai_request.get("max_tokens"),
                    presence_penalty=openai_request.get("presence_penalty"),
                    frequency_penalty=openai_request.get("frequency_penalty"),
                    logit_bias=openai_request.get("logit_bias"),
                    user=openai_request.get("user"),
                    tools=openai_request.get("tools"),
                    tool_choice=openai_request.get("tool_choice"),
                )
                # Convert back to Anthropic format
                response_data = openai_to_anthropic_response(openai_response, request.model)

            elif provider_name == "cerebras":
                # Cerebras provider - convert to OpenAI format, call, convert back to Anthropic
                openai_request = anthropic_to_openai_request(request_dict)
                openai_response = await cerebras_provider.call(
                    model=provider_model,
                    messages=openai_request["messages"],
                    temperature=openai_request.get("temperature"),
                    top_p=openai_request.get("top_p"),
                    n=openai_request.get("n"),
                    stream=openai_request.get("stream", False),
                    stop=openai_request.get("stop"),
                    max_tokens=openai_request.get("max_tokens"),
                    presence_penalty=openai_request.get("presence_penalty"),
                    frequency_penalty=openai_request.get("frequency_penalty"),
                    logit_bias=openai_request.get("logit_bias"),
                    user=openai_request.get("user"),
                    tools=openai_request.get("tools"),
                    tool_choice=openai_request.get("tool_choice"),
                )
                # Convert back to Anthropic format
                response_data = openai_to_anthropic_response(openai_response, request.model)

            elif provider_name in ["llama", "mistral", "cloudflare", "gemini", "chutes", "longcat"]:
                # Other OpenAI-compatible providers - convert to OpenAI format, call, convert back to Anthropic
                provider_map = {
                    "llama": llama_provider,
                    "mistral": mistral_provider,
                    "cloudflare": cloudflare_provider,
                    "gemini": gemini_provider,
                    "chutes": chutes_provider,
                    "longcat": longcat_provider
                }
                provider_instance = provider_map[provider_name]
                openai_request = anthropic_to_openai_request(request_dict)
                openai_response = await provider_instance.call(
                    model=provider_model,
                    messages=openai_request["messages"],
                    temperature=openai_request.get("temperature"),
                    top_p=openai_request.get("top_p"),
                    n=openai_request.get("n"),
                    stream=openai_request.get("stream", False),
                    stop=openai_request.get("stop"),
                    max_tokens=openai_request.get("max_tokens"),
                    presence_penalty=openai_request.get("presence_penalty"),
                    frequency_penalty=openai_request.get("frequency_penalty"),
                    logit_bias=openai_request.get("logit_bias"),
                    user=openai_request.get("user"),
                    tools=openai_request.get("tools"),
                    tool_choice=openai_request.get("tool_choice"),
                )
                # Convert back to Anthropic format
                response_data = openai_to_anthropic_response(openai_response, request.model)
            else:
                raise create_provider_error_response(
                    "anthropic",
                    400,
                    f"Unsupported provider: {provider_name}",
                    "invalid_request_error"
                )
        
        # Calculate response time
        response_time_ms = int((time.time() - start_time) * 1000)
        
        # Extract response content and usage
        response_content = extract_response_content(response_data, is_openai_format=False)
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
        if e.status_code == 400:
            try:
                payload = request_payload or request.model_dump(exclude_none=True)
            except Exception:
                payload = request_payload or {}
            _log_bad_request("/v1/messages", e.detail, payload)
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
            "anthropic",
            500,
            f"Error processing request: {str(e)}",
            "internal_server_error"
        )


@router.post("/v1/messages-stream")
async def messages_stream(
    request: AnthropicMessagesRequest,
    http_request: Request,
    db: Session = Depends(get_db),
    _: bool = Depends(verify_client_api_key)
):
    """
    Anthropic-compatible streaming messages endpoint.
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
        request_payload = request_dict.copy()
        
        # Extract parameters for logging
        parameters = extract_parameters_from_request(request_dict)
        
        # Create initial log entry for streaming
        if request_id:
            logging_crud.create_request_log(
                db=db,
                request_id=request_id,
                endpoint="/v1/messages-stream",
                method="POST",
                requested_model=request.model,
                resolved_provider=provider_name,
                resolved_model=provider_model,
                parameters=parameters,
                messages=request_dict["messages"],
                client_api_key_hash=client_api_key_hash,
                is_streaming=True
            )

        return await _create_streaming_response(
            request_dict=request_dict,
            provider_name=provider_name,
            provider_model=provider_model,
            requested_model=request.model,
            request_id=request_id,
            start_time=start_time,
            db=db,
            client_api_key_hash=client_api_key_hash,
        )
    except HTTPException as e:
        if request_id:
            try:
                response_time_ms = int((time.time() - start_time) * 1000)
                logging_crud.update_request_log(
                    db=db,
                    request_id=request_id,
                    response_status=e.status_code,
                    response_time_ms=response_time_ms,
                    error_message=str(e.detail),
                    error_type=type(e).__name__
                )
            except Exception:
                pass
        if e.status_code == 400:
            _log_bad_request(
                endpoint="/v1/messages-stream",
                detail=e.detail,
                payload=request_payload or request.model_dump(exclude_none=True),
            )
        raise
    except Exception as e:
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
                pass
        raise create_provider_error_response(
            "anthropic",
            500,
            f"Error processing streaming request: {str(e)}",
            "internal_server_error"
        )


async def _create_streaming_response(
    request_dict: Dict[str, Any],
    provider_name: str,
    provider_model: str,
    requested_model: str,
    request_id: Optional[str],
    start_time: float,
    db: Session,
    client_api_key_hash: Optional[str],
):
    class OpenAIStreamAdapter:
        """Translates OpenAI-style SSE chunks into Anthropic-compatible SSE."""

        STOP_REASON_MAP = {
            "stop": "end_turn",
            "length": "max_tokens",
            "content_filter": "content_filter",
            "tool_calls": "tool_use",
            "function_call": "tool_use",
        }

        def __init__(self, requested_model: str):
            self.requested_model = requested_model
            self.message_id = f"msg_{uuid.uuid4().hex}"
            self.response_content: List[Dict[str, Any]] = []
            self.active_blocks: Dict[int, Dict[str, Any]] = {}
            self.current_text_index: Optional[int] = None
            self.next_block_index = 0
            self.tool_state: Dict[str, Dict[str, Any]] = {}
            self.finish_reason: Optional[str] = None
            self.usage: Optional[Dict[str, Any]] = None

        def _sse(self, event: str, payload: Dict[str, Any]):
            data = json.dumps(payload, separators=(",", ":"))
            yield f"event: {event}\n"
            yield f"data: {data}\n\n"

        def start(self):
            message_payload = {
                "type": "message_start",
                "message": {
                    "id": self.message_id,
                    "type": "message",
                    "role": "assistant",
                    "model": self.requested_model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "cache_creation_input_tokens": 0,
                        "cache_read_input_tokens": 0,
                    },
                },
            }
            yield from self._sse("message_start", message_payload)

        def process_chunk(self, raw_chunk: str):
            try:
                lines = raw_chunk.splitlines()
            except Exception:
                lines = [raw_chunk]
            for line in lines:
                stripped = line.strip()
                if not stripped or not stripped.startswith("data:"):
                    continue
                payload = stripped[5:].strip()
                if not payload:
                    continue
                if payload == "[DONE]":
                    continue
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                yield from self._handle_openai_object(obj)

        def _handle_openai_object(self, obj: Dict[str, Any]):
            usage = obj.get("usage")
            if usage:
                self.usage = usage
            choices = obj.get("choices")
            if not isinstance(choices, list) or not choices:
                return
            choice = choices[0]
            delta = choice.get("delta") or {}
            finish_reason = choice.get("finish_reason")
            if finish_reason:
                self.finish_reason = finish_reason
            if delta:
                yield from self._handle_delta(delta)

        def _handle_delta(self, delta: Dict[str, Any]):
            content = delta.get("content")
            if isinstance(content, list):
                for part in content:
                    part_type = part.get("type")
                    if part_type == "text":
                        yield from self._emit_text(part.get("text", ""))
                    elif part_type == "tool_call":
                        fn = part.get("function", {}) or {}
                        tool_id = part.get("id") or fn.get("name") or f"tool_{len(self.tool_state)}"
                        name = fn.get("name") or tool_id
                        args_value = fn.get("arguments") or ""
                        idx = yield from self._ensure_tool_block(tool_id, name)
                        yield from self._emit_tool_args(tool_id, idx, args_value)
            elif isinstance(content, str):
                yield from self._emit_text(content)
            elif content:
                yield from self._emit_text(str(content))

            tool_calls = delta.get("tool_calls")
            if isinstance(tool_calls, list):
                for call in tool_calls:
                    fn = call.get("function", {}) or {}
                    tool_id = call.get("id") or fn.get("name") or f"tool_{len(self.tool_state)}"
                    name = fn.get("name") or tool_id
                    args_value = fn.get("arguments") or ""
                    idx = yield from self._ensure_tool_block(tool_id, name)
                    yield from self._emit_tool_args(tool_id, idx, args_value)

        def _emit_text(self, text_value: str):
            if not text_value:
                return
            idx = yield from self._ensure_text_block()
            block = self.active_blocks.get(idx)
            if block is not None:
                block["text"] = block.get("text", "") + text_value
            payload = {
                "type": "content_block_delta",
                "index": idx,
                "delta": {"type": "text_delta", "text": text_value},
            }
            yield from self._sse("content_block_delta", payload)

        def _ensure_text_block(self):
            if self.current_text_index is not None:
                return self.current_text_index
            idx = self._next_block_index()
            block = {"type": "text", "text": ""}
            self.response_content.append(block)
            self.active_blocks[idx] = block
            self.current_text_index = idx
            payload = {
                "type": "content_block_start",
                "index": idx,
                "content_block": {"type": "text", "text": ""},
            }
            yield from self._sse("content_block_start", payload)
            return idx

        def _ensure_tool_block(self, tool_id: str, tool_name: str):
            if tool_id in self.tool_state:
                return self.tool_state[tool_id]["block_index"]
            if self.current_text_index is not None:
                yield from self._finalize_block(self.current_text_index)
            idx = self._next_block_index()
            block = {"type": "tool_use", "id": tool_id, "name": tool_name, "input": {}}
            self.response_content.append(block)
            self.active_blocks[idx] = block
            self.tool_state[tool_id] = {"block_index": idx, "buffer": "", "name": tool_name}
            payload = {
                "type": "content_block_start",
                "index": idx,
                "content_block": {"type": "tool_use", "id": tool_id, "name": tool_name, "input": {}},
            }
            yield from self._sse("content_block_start", payload)
            return idx

        def _emit_tool_args(self, tool_id: str, block_index: int, arguments: str):
            state = self.tool_state.get(tool_id)
            if state is None:
                return
            prev = state.get("buffer", "")
            full_args = arguments or ""
            delta_text = full_args[len(prev):] if full_args.startswith(prev) else full_args
            state["buffer"] = full_args
            if not delta_text:
                return
            payload = {
                "type": "content_block_delta",
                "index": block_index,
                "delta": {"type": "input_json_delta", "partial_json": delta_text},
            }
            yield from self._sse("content_block_delta", payload)

        def _next_block_index(self) -> int:
            idx = self.next_block_index
            self.next_block_index += 1
            return idx

        def _finalize_block(self, idx: int):
            block = self.active_blocks.pop(idx, None)
            if not block:
                return
            if block.get("type") == "text":
                if self.current_text_index == idx:
                    self.current_text_index = None
            elif block.get("type") == "tool_use":
                tool_id = block.get("id")
                state = self.tool_state.pop(tool_id, None) if tool_id else None
                buffer = state.get("buffer", "") if state else ""
                if buffer:
                    try:
                        block["input"] = json.loads(buffer)
                    except json.JSONDecodeError:
                        block["input"] = {"_raw": buffer}
            payload = {"type": "content_block_stop", "index": idx}
            yield from self._sse("content_block_stop", payload)

        def finalize(self):
            # Close tool blocks first so their inputs are captured.
            for tool_id, state in list(self.tool_state.items()):
                yield from self._finalize_block(state["block_index"])
            if self.current_text_index is not None:
                yield from self._finalize_block(self.current_text_index)
            # Close any remaining blocks.
            for idx in list(self.active_blocks.keys()):
                yield from self._finalize_block(idx)

            message_payload = {
                "type": "message_stop",
                "message": {
                    "id": self.message_id,
                    "type": "message",
                    "role": "assistant",
                    "model": self.requested_model,
                    "content": self.response_content,
                    "stop_reason": self._stop_reason(),
                    "stop_sequence": None,
                    "usage": self._usage_payload(),
                },
            }
            yield from self._sse("message_stop", message_payload)
            yield "data: [DONE]\n\n"

        def _usage_payload(self) -> Dict[str, Any]:
            usage = self.usage or {}
            return {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            }

        def _stop_reason(self) -> str:
            if self.finish_reason:
                return self.STOP_REASON_MAP.get(self.finish_reason, "end_turn")
            return "end_turn"

    async def generate_stream():
        if provider_name == "anthropic":
            async for chunk in anthropic_provider.call_stream(
                model=provider_model,
                messages=request_dict["messages"],
                max_tokens=request_dict.get("max_tokens"),
                temperature=request_dict.get("temperature"),
                top_p=request_dict.get("top_p"),
                top_k=request_dict.get("top_k"),
                tools=request_dict.get("tools"),
            ):
                yield chunk
            yield "data: [DONE]\n\n"
            return

        openai_request = anthropic_to_openai_request(request_dict)
        if provider_name == "openai":
            stream_fn = openai_provider.call_stream
        elif provider_name == "nahcrof":
            stream_fn = nahcrof_provider.call_stream
        elif provider_name == "groq":
            stream_fn = groq_provider.call_stream
        elif provider_name == "cerebras":
            stream_fn = cerebras_provider.call_stream
        elif provider_name in ["llama", "mistral", "cloudflare", "gemini", "chutes", "longcat"]:
            provider_map = {
                "llama": llama_provider,
                "mistral": mistral_provider,
                "cloudflare": cloudflare_provider,
                "gemini": gemini_provider,
                "chutes": chutes_provider,
                "longcat": longcat_provider,
            }
            stream_fn = provider_map[provider_name].call_stream
        else:
            raise create_provider_error_response(
                "anthropic",
                400,
                f"Unsupported provider: {provider_name}",
                "invalid_request_error",
            )

        adapter = OpenAIStreamAdapter(requested_model)
        for chunk in adapter.start():
            yield chunk

        async for chunk in stream_fn(
            model=provider_model,
            messages=openai_request["messages"],
            temperature=openai_request.get("temperature"),
            top_p=openai_request.get("top_p"),
            n=openai_request.get("n"),
            stop=openai_request.get("stop"),
            max_tokens=openai_request.get("max_tokens"),
            presence_penalty=openai_request.get("presence_penalty"),
            frequency_penalty=openai_request.get("frequency_penalty"),
            logit_bias=openai_request.get("logit_bias"),
            user=openai_request.get("user"),
            tools=openai_request.get("tools"),
            tool_choice=openai_request.get("tool_choice"),
        ):
            for converted in adapter.process_chunk(chunk):
                yield converted

        for chunk in adapter.finalize():
            yield chunk

        if request_id:
            response_time_ms = int((time.time() - start_time) * 1000)
            logging_crud.update_request_log(
                db=db,
                request_id=request_id,
                response_status=200,
                response_time_ms=response_time_ms,
            )

    try:
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
        if request_id:
            response_time_ms = int((time.time() - start_time) * 1000)
            logging_crud.update_request_log(
                db=db,
                request_id=request_id,
                response_status=e.status_code,
                response_time_ms=response_time_ms,
                error_message=str(e.detail),
                error_type=type(e).__name__,
            )
        if e.status_code == 400:
            _log_bad_request(
                endpoint="streaming_response",
                detail=e.detail,
                payload=request_dict,
            )
        raise
    except Exception as e:
        if request_id:
            response_time_ms = int((time.time() - start_time) * 1000)
            logging_crud.update_request_log(
                db=db,
                request_id=request_id,
                response_status=500,
                response_time_ms=response_time_ms,
                error_message=str(e),
                error_type=type(e).__name__,
            )
        raise HTTPException(
            status_code=500,
            detail=f"Error processing streaming request: {str(e)}",
        )

@router.post("/v1/messages/count_tokens")
async def messages_count_tokens(
    request: AnthropicMessagesRequest,
    http_request: Request,
    _: bool = Depends(verify_client_api_key),
):
    """
    Anthropic-compatible token counting endpoint.
    Returns an approximate input token count for the provided messages.
    """
    try:
        # Convert Anthropic request to OpenAI format to flatten content into text
        request_dict = request.model_dump(exclude_none=True)
        openai_request = anthropic_to_openai_request(request_dict)

        def approximate_tokens_from_text(text_value: str) -> int:
            if not text_value:
                return 0
            # Approximate 1 token ~= 4 characters
            return (len(text_value) + 3) // 4

        total_tokens = 0
        for msg in openai_request.get("messages", []):
            content = msg.get("content")
            if isinstance(content, str):
                total_tokens += approximate_tokens_from_text(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        text_part = part.get("text") or json.dumps(part, separators=(",", ":"))
                        total_tokens += approximate_tokens_from_text(text_part)
                    elif isinstance(part, str):
                        total_tokens += approximate_tokens_from_text(part)

        return {"input_tokens": int(total_tokens)}
    except Exception as e:
        raise create_provider_error_response(
            "anthropic",
            500,
            f"Error processing token count: {str(e)}",
            "internal_server_error",
        )
