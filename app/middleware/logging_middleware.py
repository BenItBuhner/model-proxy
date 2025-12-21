"""
Middleware for capturing request metadata and timing.
"""

import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from app.core.logging import generate_request_id, hash_api_key
from app.auth import CLIENT_API_KEY


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to capture request start time and generate request IDs.
    Stores request metadata in request.state for later logging.
    """

    async def dispatch(self, request: Request, call_next):
        # Generate request ID
        request_id = generate_request_id()
        request.state.request_id = request_id
        request.state.start_time = time.time()

        # Store client API key hash for rate limiting
        if CLIENT_API_KEY:
            request.state.client_api_key_hash = hash_api_key(CLIENT_API_KEY)
        else:
            request.state.client_api_key_hash = "anonymous"

        # Process request
        response = await call_next(request)

        # Calculate response time
        request.state.response_time_ms = int(
            (time.time() - request.state.start_time) * 1000
        )

        # Add request ID to response headers for tracing
        response.headers["X-Request-ID"] = request_id

        return response
