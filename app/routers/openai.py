"""
OpenAI-compatible API router.
Provides /v1/chat/completions and /v1/models endpoints with full fallback routing support.

This router implements the OpenAI Chat Completions API format and handles:
- Request validation and normalization
- Fallback routing through multiple providers/models
- Protocol conversion for non-OpenAI providers (e.g., Anthropic)
- Streaming with proper SSE format
- Request/response logging
"""

import json
import logging
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.auth import CLIENT_API_KEY, verify_client_api_key
from app.core.error_formatters import (
    format_openai_error,
)
from app.core.logging import (
    extract_parameters_from_request,
    extract_response_content,
    extract_usage_from_response,
    hash_api_key,
)
from app.database import crud, logging_crud
from app.database.database import SessionLocal
from app.models.openai import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ListModelsResponse,
    Model,
)
from app.routing.config_loader import config_loader
from app.routing.models import RoutingError
from app.routing.router import FallbackRouter

logger = logging.getLogger("openai_router")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


router = APIRouter()


async def _create_streaming_response(
    request_dict: Dict[str, Any],
    requested_model: str,
    request_id: Optional[str],
    start_time: float,
    db: Session,
    client_api_key_hash: Optional[str],
    fallback_router: FallbackRouter,
):
    """Create a streaming response with full fallback routing support."""

    async def generate_stream():
        try:
            # Get the stream generator from fallback router
            stream_generator = await fallback_router.call_with_fallback(
                logical_model=requested_model,
                request_data=request_dict,
                target_protocol="openai",
                stream=True,
            )

            async for chunk in stream_generator:
                yield chunk

            # Update log on success
            if request_id:
                response_time_ms = int((time.time() - start_time) * 1000)
                try:
                    logging_crud.update_request_log(
                        db=db,
                        request_id=request_id,
                        response_status=200,
                        response_time_ms=response_time_ms,
                    )
                except Exception:
                    pass

        except RoutingError as e:
            # All routes failed - emit error in SSE format
            error_response = format_openai_error(
                503,
                f"All routes failed: {e.get_error_summary()}",
                "service_unavailable",
            )
            yield f"data: {json.dumps(error_response)}\n\n"
            yield "data: [DONE]\n\n"

            if request_id:
                response_time_ms = int((time.time() - start_time) * 1000)
                try:
                    logging_crud.update_request_log(
                        db=db,
                        request_id=request_id,
                        response_status=503,
                        response_time_ms=response_time_ms,
                        error_message=str(e),
                        error_type="RoutingError",
                    )
                except Exception:
                    pass

        except Exception as e:
            logger.error(f"Streaming error: {str(e)}")
            error_response = format_openai_error(
                500, f"Streaming error: {str(e)}", "internal_server_error"
            )
            yield f"data: {json.dumps(error_response)}\n\n"
            yield "data: [DONE]\n\n"

            if request_id:
                response_time_ms = int((time.time() - start_time) * 1000)
                try:
                    logging_crud.update_request_log(
                        db=db,
                        request_id=request_id,
                        response_status=500,
                        response_time_ms=response_time_ms,
                        error_message=str(e),
                        error_type=type(e).__name__,
                    )
                except Exception:
                    pass

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.get("/v1/models", response_model=ListModelsResponse)
async def list_models(_: bool = Depends(verify_client_api_key)):
    """
    List all available models.
    Returns models from the routing configuration in OpenAI-compatible format.
    """
    try:
        # Get available models from routing config
        available_models = config_loader.get_available_models()

        models = []
        for model_name in available_models:
            try:
                config = config_loader.load_config(model_name)
                # Use first provider as the "owner"
                owner = "unknown"
                if config.model_routings:
                    owner = config.model_routings[0].provider

                model = Model(
                    id=model_name,
                    created=int(time.time()),
                    owned_by=owner,
                )
                models.append(model)
            except Exception as e:
                logger.warning(f"Failed to load config for model '{model_name}': {e}")
                continue

        return ListModelsResponse(data=models)

    except Exception as e:
        logger.exception(f"Error listing models: {e}")
        raise HTTPException(
            status_code=500,
            detail=format_openai_error(500, f"Error listing models: {str(e)}"),
        )


@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest,
    http_request: Request,
    db: Session = Depends(get_db),
    _: bool = Depends(verify_client_api_key),
):
    """
    OpenAI-compatible chat completions endpoint.
    Routes requests to appropriate provider based on model configuration.
    Supports full fallback routing (API key, provider, and model level).
    """
    request_id = getattr(http_request.state, "request_id", None)
    start_time = getattr(http_request.state, "start_time", time.time())
    client_api_key_hash = hash_api_key(CLIENT_API_KEY) if CLIENT_API_KEY else None

    try:
        # Check if this model uses the fallback routing system
        try:
            config_loader.load_config(request.model)
            logical_model = request.model
        except (FileNotFoundError, ValueError) as e:
            logger.warning(f"No routing config found for model '{request.model}': {e}")
            raise HTTPException(
                status_code=400,
                detail=format_openai_error(
                    400,
                    f"Model '{request.model}' not found in routing configuration. "
                    f"Available models: {', '.join(config_loader.get_available_models())}",
                    "invalid_request_error",
                ),
            )

        # Prepare request dict
        request_dict = request.model_dump(exclude_none=True)
        request_dict.copy()
        is_stream = bool(request_dict.get("stream", False))

        # Extract parameters for logging
        parameters = extract_parameters_from_request(request_dict)

        # Create initial log entry
        if request_id:
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
                is_streaming=is_stream,
            )

        # Create fallback router
        fallback_router = FallbackRouter()

        # Handle streaming
        if is_stream:
            return await _create_streaming_response(
                request_dict=request_dict,
                requested_model=request.model,
                request_id=request_id,
                start_time=start_time,
                db=db,
                client_api_key_hash=client_api_key_hash,
                fallback_router=fallback_router,
            )

        # Non-streaming request
        response_data = await fallback_router.call_with_fallback(
            logical_model=logical_model,
            request_data=request_dict,
            target_protocol="openai",
            stream=False,
        )

        # Update model name to requested model (preserve client's model name)
        response_data["model"] = request.model

        # Calculate response time
        response_time_ms = int((time.time() - start_time) * 1000)

        # Extract response content and usage for logging
        response_content = extract_response_content(
            response_data, is_openai_format=True
        )
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
                    response_usage=usage,
                )
            except Exception as log_error:
                logger.warning(f"Failed to update request log: {log_error}")

        # Legacy logging (keep for backward compatibility)
        try:
            crud.create_log(
                db,
                service="fallback_router",
                request=request_dict,
                response=response_data,
            )
        except Exception:
            pass

        return response_data

    except RoutingError as e:
        # All routes failed
        if request_id:
            try:
                response_time_ms = int((time.time() - start_time) * 1000)
                logging_crud.update_request_log(
                    db=db,
                    request_id=request_id,
                    response_status=503,
                    response_time_ms=response_time_ms,
                    error_message=str(e),
                    error_type="RoutingError",
                )
            except Exception:
                pass

        raise HTTPException(
            status_code=503,
            detail=format_openai_error(
                503,
                f"All routes failed for model '{request.model}': {e.get_error_summary()}",
                "service_unavailable",
            ),
        )

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise

    except Exception as e:
        # Log unexpected errors
        if request_id:
            try:
                response_time_ms = int((time.time() - start_time) * 1000)
                logging_crud.update_request_log(
                    db=db,
                    request_id=request_id,
                    response_status=500,
                    response_time_ms=response_time_ms,
                    error_message=str(e),
                    error_type=type(e).__name__,
                )
            except Exception:
                pass

        logger.exception(f"Unexpected error in /v1/chat/completions: {e}")
        raise HTTPException(
            status_code=500,
            detail=format_openai_error(
                500,
                f"Error processing request: {str(e)}",
                "internal_server_error",
            ),
        )


@router.post("/v1/chat/completions/stream")
async def chat_completions_stream(
    request: ChatCompletionRequest,
    http_request: Request,
    db: Session = Depends(get_db),
    _: bool = Depends(verify_client_api_key),
):
    """
    OpenAI-compatible streaming chat completions endpoint.
    Always returns a streaming response.
    """
    # Force streaming and delegate to main endpoint
    request_dict = request.model_dump(exclude_none=True)
    request_dict["stream"] = True

    request_id = getattr(http_request.state, "request_id", None)
    start_time = getattr(http_request.state, "start_time", time.time())
    client_api_key_hash = hash_api_key(CLIENT_API_KEY) if CLIENT_API_KEY else None

    try:
        # Check if this model uses the fallback routing system
        try:
            config_loader.load_config(request.model)
            logical_model = request.model
        except (FileNotFoundError, ValueError) as e:
            logger.warning(f"No routing config found for model '{request.model}': {e}")
            raise HTTPException(
                status_code=400,
                detail=format_openai_error(
                    400,
                    f"Model '{request.model}' not found in routing configuration.",
                    "invalid_request_error",
                ),
            )

        # Extract parameters for logging
        parameters = extract_parameters_from_request(request_dict)

        # Create initial log entry
        if request_id:
            logging_crud.create_request_log(
                db=db,
                request_id=request_id,
                endpoint="/v1/chat/completions/stream",
                method="POST",
                requested_model=request.model,
                resolved_provider="fallback_router",
                resolved_model=logical_model,
                parameters=parameters,
                messages=request_dict["messages"],
                client_api_key_hash=client_api_key_hash,
                is_streaming=True,
            )

        # Create fallback router and return streaming response
        fallback_router = FallbackRouter()

        return await _create_streaming_response(
            request_dict=request_dict,
            requested_model=request.model,
            request_id=request_id,
            start_time=start_time,
            db=db,
            client_api_key_hash=client_api_key_hash,
            fallback_router=fallback_router,
        )

    except HTTPException:
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
                    error_type=type(e).__name__,
                )
            except Exception:
                pass

        logger.exception(f"Unexpected error in /v1/chat/completions/stream: {e}")
        raise HTTPException(
            status_code=500,
            detail=format_openai_error(
                500,
                f"Error processing streaming request: {str(e)}",
                "internal_server_error",
            ),
        )
