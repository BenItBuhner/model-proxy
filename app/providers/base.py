"""
Base provider class defining the interface for all LLM providers.
Enhanced with route configuration injection for fallback routing support.
"""

from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, Optional


class BaseProvider(ABC):
    """
    Abstract base class for LLM providers.
    All providers must implement call() and call_stream() methods.

    Supports route-specific configuration injection for fallback routing,
    allowing per-request API key and base URL overrides.
    """

    def __init__(self, provider_name: str):
        """
        Initialize provider.

        Args:
            provider_name: Name of the provider (e.g., "openai", "anthropic")
        """
        self.provider_name = provider_name
        # Route-specific configuration (injected per-route)
        self._route_api_key: Optional[str] = None
        self._route_base_url: Optional[str] = None

    def set_route_config(
        self, api_key: Optional[str] = None, base_url: Optional[str] = None
    ) -> "BaseProvider":
        """
        Set route-specific configuration for this provider instance.

        This allows the fallback router to inject per-route API keys
        and base URLs without modifying the provider's default configuration.

        Args:
            api_key: API key to use for this specific route
            base_url: Base URL override for this specific route

        Returns:
            Self for method chaining
        """
        self._route_api_key = api_key
        self._route_base_url = base_url
        return self

    def clear_route_config(self) -> "BaseProvider":
        """
        Clear route-specific configuration, reverting to defaults.

        Returns:
            Self for method chaining
        """
        self._route_api_key = None
        self._route_base_url = None
        return self

    def _get_effective_api_key(self) -> Optional[str]:
        """
        Get the effective API key to use for requests.

        Route-specific API key takes precedence over the default
        key retrieval mechanism.

        Returns:
            API key string, or None if no key available
        """
        if self._route_api_key:
            return self._route_api_key
        return self._get_api_key()

    def _get_effective_base_url(self, default_base_url: str) -> str:
        """
        Get the effective base URL to use for requests.

        Route-specific base URL takes precedence over the default.

        Args:
            default_base_url: The provider's default base URL

        Returns:
            Base URL to use
        """
        if self._route_base_url:
            return self._route_base_url
        return default_base_url

    @abstractmethod
    async def call(self, model: str, messages: list, **kwargs) -> Dict[str, Any]:
        """
        Make a synchronous API call to the provider.

        Args:
            model: Model name to use
            messages: List of messages
            **kwargs: Additional provider-specific parameters

        Returns:
            Response dictionary

        Raises:
            Exception: On API errors
        """
        pass

    @abstractmethod
    async def call_stream(
        self, model: str, messages: list, **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Make a streaming API call to the provider.

        Args:
            model: Model name to use
            messages: List of messages
            **kwargs: Additional provider-specific parameters

        Yields:
            Chunks of streaming response data

        Raises:
            Exception: On API errors
        """
        pass

    def _get_api_key(self) -> Optional[str]:
        """
        Get an available API key for this provider using the default mechanism.

        Returns:
            API key or None if none available
        """
        from app.core.api_key_manager import get_api_key

        return get_api_key(self.provider_name)

    def _mark_key_failed(self, key: str) -> None:
        """
        Mark an API key as failed.

        Args:
            key: The failed API key
        """
        from app.core.api_key_manager import mark_key_failed

        mark_key_failed(self.provider_name, key)
