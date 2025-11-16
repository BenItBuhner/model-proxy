"""
Core fallback router implementation.
Handles multi-level routing with API key, provider, and logical model fallbacks.
"""
import asyncio
import os
from typing import List, Optional, Callable, Any, Dict, Union
from contextlib import asynccontextmanager
import aiohttp
from aiohttp import ClientTimeout
from aiohttp.client_exceptions import ClientError

from app.routing.models import (
    ModelRoutingConfig, RouteConfig, ResolvedRoute, Attempt, RoutingError
)
from app.routing.config_loader import config_loader


class FallbackRouter:
    """Router that implements multi-level fallback logic."""

    def __init__(self):
        self._visited_models: set[str] = set()  # Track visited models to prevent cycles

    async def call_with_fallback(
        self,
        logical_model: str,
        exec_route_fn: Callable[[ResolvedRoute], Any]
    ) -> Any:
        """
        Execute a logical model request with full fallback routing.

        Args:
            logical_model: The logical model name to resolve
            exec_route_fn: Function to execute a resolved route, should return response or raise

        Returns:
            The successful response from the first working route

        Raises:
            RoutingError: If all routes fail
        """
        attempts = self.resolve_attempts(logical_model)

        errors = []
        for attempt in attempts:
            try:
                result = await exec_route_fn(attempt.route)
                return result
            except Exception as e:
                # Check if this is a fallback-worthy error
                if self._is_fallback_worthy_error(e):
                    errors.append({
                        "attempt": attempt,
                        "error": str(e),
                        "error_type": type(e).__name__
                    })
                    continue
                else:
                    # Non-fallback error, re-raise immediately
                    raise e

        # All attempts failed
        raise RoutingError(
            logical_model=logical_model,
            attempted_routes=attempts,
            errors=errors,
            message=self._format_routing_error_message(logical_model, attempts, errors)
        )

    def resolve_attempts(self, logical_model: str) -> List[Attempt]:
        """
        Resolve all possible attempts for a logical model, including fallbacks.

        Returns ordered list of attempts to try in sequence.
        """
        self._visited_models.clear()
        return self._resolve_attempts_recursive(logical_model, is_fallback=False)

    def _resolve_attempts_recursive(
        self,
        logical_model: str,
        is_fallback: bool = False
    ) -> List[Attempt]:
        """Recursively resolve attempts, including fallback logical models."""
        # Cycle detection
        if logical_model in self._visited_models:
            return []  # Skip cycles

        self._visited_models.add(logical_model)

        try:
            config = config_loader.load_config(logical_model)
        except Exception:
            # If we can't load the config, skip it
            self._visited_models.remove(logical_model)
            return []

        attempts = []

        # First, add attempts from this model's routes
        attempt_number = 1

        # Expand each route into attempts (one per API key)
        for route_config in config.model_routings:
            route_attempts = self._expand_route_to_attempts(
                route_config, logical_model, config.timeout_seconds, attempt_number, is_fallback
            )
            attempts.extend(route_attempts)
            attempt_number += len(route_attempts)

        # Then recursively add attempts from fallback models
        for fallback_model in config.fallback_model_routings:
            fallback_attempts = self._resolve_attempts_recursive(
                fallback_model, is_fallback=True
            )
            # Adjust attempt numbers for the fallback attempts
            for attempt in fallback_attempts:
                attempt.attempt_number = attempt_number
                attempt_number += 1
            attempts.extend(fallback_attempts)

        self._visited_models.remove(logical_model)
        return attempts

    def _expand_route_to_attempts(
        self,
        route_config: RouteConfig,
        source_logical_model: str,
        default_timeout: int,
        start_attempt_number: int,
        is_fallback_route: bool
    ) -> List[Attempt]:
        """Expand a single route config into multiple attempts (one per API key)."""
        attempts = []

        for i, api_key_env in enumerate(route_config.api_key_env):
            # Try to get the API key from environment
            api_key = os.getenv(api_key_env)
            if not api_key:
                continue  # Skip if API key not available

            resolved_route = ResolvedRoute(
                source_logical_model=source_logical_model,
                wire_protocol=route_config.wire_protocol,
                provider=route_config.provider,
                model=route_config.model,
                base_url=route_config.base_url,
                api_key=api_key,
                timeout_seconds=route_config.timeout_seconds or default_timeout,
                route_id=route_config.id
            )

            attempt = Attempt(
                route=resolved_route,
                attempt_number=start_attempt_number + i,
                is_fallback_route=is_fallback_route
            )

            attempts.append(attempt)

        return attempts

    def _is_fallback_worthy_error(self, error: BaseException) -> bool:
        """
        Determine if an error should trigger fallback to the next route.

        Fallback-worthy errors include:
        - HTTP 4xx/5xx status codes (but not auth-related 401/403 that might be permanent)
        - Network errors (connection, DNS, etc.)
        - Timeout errors
        """
        # Check for HTTP status errors
        if hasattr(error, 'status'):
            status = error.status
            # Treat 4xx and 5xx as fallback-worthy, but not 401/403 which are auth errors
            if 400 <= status < 600 and status not in (401, 403):
                return True

        # Check for aiohttp client errors (network issues)
        if isinstance(error, (aiohttp.ClientError, aiohttp.ServerTimeoutError, asyncio.TimeoutError)):
            return True

        # Check for specific error messages that indicate transient issues
        error_msg = str(error).lower()
        transient_indicators = [
            'timeout', 'connection', 'network', 'unreachable', 'reset',
            'server error', 'internal server error', 'bad gateway', 'service unavailable',
            'gateway timeout', 'too many requests'  # 429 rate limit
        ]

        return any(indicator in error_msg for indicator in transient_indicators)

    def _format_routing_error_message(
        self,
        logical_model: str,
        attempts: List[Attempt],
        errors: List[Dict[str, Any]]
    ) -> str:
        """Format a human-readable error message for routing failures."""
        parts = [f"All routes failed for logical model '{logical_model}'"]

        if attempts:
            parts.append(f"Attempted {len(attempts)} routes:")
            for i, attempt in enumerate(attempts, 1):
                route = attempt.route
                status = "FAILED"
                if i <= len(errors):
                    error_info = errors[i-1]
                    status = f"FAILED ({error_info['error_type']}: {error_info['error'][:50]}...)"

                parts.append(f"  {i}. {route.provider}/{route.model} ({route.wire_protocol}) - {status}")

        return "\n".join(parts)


# Convenience function for easy access
async def call_with_fallback(
    logical_model: str,
    exec_route_fn: Callable[[ResolvedRoute], Any]
) -> Any:
    """Convenience function to call with fallback routing."""
    router = FallbackRouter()
    return await router.call_with_fallback(logical_model, exec_route_fn)
