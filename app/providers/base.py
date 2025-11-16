"""
Base provider class defining the interface for all LLM providers.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, AsyncGenerator, Optional


class BaseProvider(ABC):
    """
    Abstract base class for LLM providers.
    All providers must implement call() and call_stream() methods.
    """
    
    def __init__(self, provider_name: str):
        """
        Initialize provider.
        
        Args:
            provider_name: Name of the provider (e.g., "openai", "anthropic")
        """
        self.provider_name = provider_name
    
    @abstractmethod
    async def call(
        self,
        model: str,
        messages: list,
        **kwargs
    ) -> Dict[str, Any]:
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
        self,
        model: str,
        messages: list,
        **kwargs
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
        Get an available API key for this provider.
        
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

