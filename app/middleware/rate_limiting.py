"""
Rate limiting middleware for request throttling.
"""
import time
from collections import defaultdict
from typing import Dict, Tuple
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from app.core.logging import hash_api_key
import os


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware using token bucket algorithm.
    Tracks requests per client API key hash.
    """
    
    def __init__(self, app, requests_per_minute: int = None, tokens_per_minute: int = None):
        super().__init__(app)
        # Get rate limits from environment or use defaults
        self.requests_per_minute = requests_per_minute or int(
            os.getenv("RATE_LIMIT_REQUESTS_PER_MINUTE", "60")
        )
        self.tokens_per_minute = tokens_per_minute or int(
            os.getenv("RATE_LIMIT_TOKENS_PER_MINUTE", "100000")
        )
        
        # Track requests: {client_key_hash: [(timestamp, tokens), ...]}
        self._request_history: Dict[str, list] = defaultdict(list)
        
        # Cleanup old entries every N requests
        self._cleanup_counter = 0
        self._cleanup_interval = 100
    
    def _get_client_key_from_request(self, request: Request) -> str:
        """
        Extract client API key hash from request.
        Tries to get from request.state first, otherwise extracts from Authorization header.
        """
        # Try to get from request state (set by logging middleware if available)
        if hasattr(request.state, "client_api_key_hash"):
            return request.state.client_api_key_hash
        
        # Extract from Authorization header
        auth_header = request.headers.get("Authorization", "")
        if auth_header:
            # Strip Bearer prefix
            api_key = auth_header.strip()
            while api_key.lower().startswith("bearer "):
                api_key = api_key[7:].strip()
            if api_key:
                return hash_api_key(api_key)
        
        return "anonymous"
    
    def _cleanup_old_entries(self, client_key: str):
        """Remove entries older than 1 minute."""
        current_time = time.time()
        one_minute_ago = current_time - 60
        
        self._request_history[client_key] = [
            (ts, tokens) for ts, tokens in self._request_history[client_key]
            if ts > one_minute_ago
        ]
    
    def _check_rate_limit(self, client_key: str, estimated_tokens: int = 0) -> Tuple[bool, str]:
        """
        Check if request should be rate limited.
        
        Args:
            client_key: Client API key hash
            estimated_tokens: Estimated tokens for this request
            
        Returns:
            Tuple of (is_allowed, error_message)
        """
        current_time = time.time()
        
        # Cleanup old entries periodically
        self._cleanup_counter += 1
        if self._cleanup_counter >= self._cleanup_interval:
            for key in list(self._request_history.keys()):
                self._cleanup_old_entries(key)
            self._cleanup_counter = 0
        else:
            self._cleanup_old_entries(client_key)
        
        # Count requests in last minute
        recent_requests = [
            (ts, tokens) for ts, tokens in self._request_history[client_key]
            if ts > current_time - 60
        ]
        
        request_count = len(recent_requests)
        total_tokens = sum(tokens for _, tokens in recent_requests) + estimated_tokens
        
        # Check request rate limit
        if request_count >= self.requests_per_minute:
            return False, f"Rate limit exceeded: {request_count}/{self.requests_per_minute} requests per minute"
        
        # Check token rate limit
        if total_tokens >= self.tokens_per_minute:
            return False, f"Rate limit exceeded: {total_tokens}/{self.tokens_per_minute} tokens per minute"
        
        # Record this request
        self._request_history[client_key].append((current_time, estimated_tokens))
        
        return True, ""
    
    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health checks
        if request.url.path.startswith("/health"):
            return await call_next(request)
        
        # Get client API key hash from request
        client_key = self._get_client_key_from_request(request)
        
        # Estimate tokens (rough estimate: ~4 chars per token)
        # For now, we'll use a conservative estimate
        # TODO: Improve token estimation based on actual request size
        estimated_tokens = 100  # Default estimate
        
        # Check rate limit
        allowed, error_msg = self._check_rate_limit(client_key, estimated_tokens)
        
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "message": error_msg,
                        "type": "rate_limit_error",
                        "code": "rate_limit_exceeded"
                    }
                }
            )
        
        response = await call_next(request)
        
        # Add rate limit headers
        recent_requests = [
            (ts, tokens) for ts, tokens in self._request_history[client_key]
            if ts > time.time() - 60
        ]
        request_count = len(recent_requests)
        total_tokens = sum(tokens for _, tokens in recent_requests)
        
        response.headers["X-RateLimit-Limit-Requests"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining-Requests"] = str(max(0, self.requests_per_minute - request_count))
        response.headers["X-RateLimit-Limit-Tokens"] = str(self.tokens_per_minute)
        response.headers["X-RateLimit-Remaining-Tokens"] = str(max(0, self.tokens_per_minute - total_tokens))
        
        return response

