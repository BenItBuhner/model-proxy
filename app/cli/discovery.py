import datetime
from typing import Dict, List

import httpx

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from app.cli.config_manager import ConfigManager
from app.cli.interactive import (
    display_error,
    display_info,
    display_success,
    display_warning,
)
from app.core.api_key_manager import get_api_key
from app.core.provider_config import get_provider_auth_headers


class ModelDiscovery:
    """
    Discovers available models from provider APIs.

    Supports:
    - OpenAI-compatible APIs
    - Anthropic API
    - Google Gemini API
    - Azure OpenAI API
    - Custom endpoints
    """

    def __init__(
        self, config_manager: ConfigManager, timeout: int = 30, quiet: bool = False
    ):
        """
        Initialize the model discovery service.

        Args:
            config_manager: Configuration manager instance
            timeout: Timeout for HTTP requests in seconds
            quiet: If True, suppress interactive CLI output (useful for background refresh)
        """
        self.config_manager = config_manager
        self.timeout = timeout
        self.quiet = quiet
        self.client = httpx.AsyncClient(timeout=timeout)

    def _info(self, message: str) -> None:
        if not self.quiet:
            display_info(message)

    def _success(self, message: str) -> None:
        if not self.quiet:
            display_success(message)

    def _warning(self, message: str) -> None:
        if not self.quiet:
            display_warning(message)

    def _error(self, message: str) -> None:
        if not self.quiet:
            display_error(message)

    async def discover_models_for_all_providers(self) -> Dict[str, List[str]]:
        """
        Discover models for all enabled providers.

        Returns:
            Dictionary of provider -> list of model IDs
        """
        providers = self.config_manager.get_providers()
        results = {}

        tasks = []
        for provider_name, provider_config in providers.items():
            if not provider_config.get("enabled"):
                continue

            task = self._discover_provider_models(provider_name, provider_config)
            tasks.append((provider_name, task))

        # Run discovery for all providers concurrently
        for provider_name, task in tasks:
            try:
                models = await task
                if models:
                    results[provider_name] = models
                    self._success(
                        f"Discovered {len(models)} models from {provider_name}"
                    )
            except Exception as e:
                self._error(f"Failed to discover models from {provider_name}: {e}")

        # Add custom models from cache
        cache = self.config_manager.get_models_cache()
        custom_models = cache.get("custom_models", {})
        for provider_name, models in custom_models.items():
            if provider_name in results:
                # Merge with discovered models, remove duplicates
                existing = set(results[provider_name])
                for model in models:
                    if model not in existing:
                        results[provider_name].append(model)
            else:
                results[provider_name] = models.copy()

        return results

    async def _discover_provider_models(
        self, provider_name: str, provider_config: Dict
    ) -> List[str]:
        """
        Discover models from a specific provider.

        Args:
            provider_name: Provider identifier
            provider_config: Provider configuration

        Returns:
            List of model IDs
        """
        format_type = provider_config.get("endpoints", {}).get("compatible_format")
        base_url = provider_config.get("endpoints", {}).get("base_url", "")

        # Get API key and auth headers for this provider
        api_key = get_api_key(provider_name)
        if not api_key:
            self._warning(
                f"No API key configured for {provider_name}, skipping discovery"
            )
            return []

        try:
            headers = get_provider_auth_headers(provider_name, api_key)
        except Exception as e:
            self._warning(f"Could not get auth headers for {provider_name}: {e}")
            headers = {}

        if format_type == "openai":
            return await self._discover_openai_models(base_url, headers)
        elif format_type == "anthropic":
            return await self._discover_anthropic_models(base_url, headers)
        elif format_type == "gemini":
            return await self._discover_gemini_models(base_url, headers)
        elif format_type == "azure":
            return await self._discover_azure_models(base_url, provider_config, headers)
        else:
            self._warning(f"Unsupported format for model discovery: {format_type}")
            return []

    async def _discover_openai_models(
        self, base_url: str, headers: Dict[str, str]
    ) -> List[str]:
        """Discover models from OpenAI-compatible endpoint."""
        # Check if base_url already ends with /v1 to avoid /v1/v1/models
        base = base_url.rstrip("/")
        if base.endswith("/v1") or base.endswith("/v1beta") or base.endswith("/openai"):
            models_url = f"{base}/models"
        else:
            models_url = f"{base}/v1/models"

        try:
            response = await self.client.get(models_url, headers=headers)
            response.raise_for_status()

            data = response.json()
            models = [m.get("id", "") for m in data.get("data", [])]
            return sorted(models)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                self._warning("OpenAI model discovery failed: API key required")
            else:
                self._error(f"OpenAI model discovery HTTP error: {e}")
            return []
        except Exception as e:
            self._error(f"OpenAI model discovery failed: {e}")
            return []

    async def _discover_anthropic_models(
        self, base_url: str, headers: Dict[str, str]
    ) -> List[str]:
        """Discover models from Anthropic API."""
        # Anthropic doesn't have a models endpoint, use known models
        known_models = [
            "claude-sonnet-4-20250514",
            "claude-opus-4-20250514",
            "claude-3-7-sonnet-20250219",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-sonnet-20240620",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
        ]

        return known_models

    async def _discover_gemini_models(
        self, base_url: str, headers: Dict[str, str]
    ) -> List[str]:
        """Discover models from Google Gemini API."""
        # Check if base_url already contains v1beta to avoid duplication
        base = base_url.rstrip("/")
        if "v1beta" in base or base.endswith("/openai"):
            # For OpenAI-compatible Gemini endpoint, use /models
            models_url = f"{base}/models"
        else:
            models_url = f"{base}/v1beta/models"

        try:
            response = await self.client.get(models_url, headers=headers)
            response.raise_for_status()

            data = response.json()
            models = [
                m.get("name", "").replace("models/", "")
                for m in data.get("models", [])
                if "generateContent" in m.get("supportedGenerationMethods", [])
            ]
            return sorted(models)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                self._warning("Gemini model discovery failed: API key required")
            else:
                self._error(f"Gemini model discovery HTTP error: {e}")
            return []
        except Exception as e:
            self._error(f"Gemini model discovery failed: {e}")
            return []

    async def _discover_azure_models(
        self, base_url: str, provider_config: Dict, headers: Dict[str, str]
    ) -> List[str]:
        """Discover models from Azure OpenAI API."""
        # Azure uses deployment names which need separate listing
        base = base_url.rstrip("/")
        # Remove /v1 suffix if present since Azure doesn't use it for deployments
        if base.endswith("/v1"):
            base = base[:-3]
        deployments_url = f"{base}/deployments?api-version=2023-05-15"

        try:
            response = await self.client.get(deployments_url, headers=headers)
            response.raise_for_status()

            data = response.json()
            deployments = [
                d.get("model", {}).get("name", "") for d in data.get("data", [])
            ]
            return sorted(deployments)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                self._warning("Azure model discovery failed: API key required")
            else:
                self._error(f"Azure model discovery HTTP error: {e}")
            return []
        except Exception as e:
            self._error(f"Azure model discovery failed: {e}")
            return []

    async def close(self) -> None:
        """Close HTTP client."""
        await self.client.aclose()


async def discover_and_cache_models(*, quiet: bool = False) -> None:
    """
    Discover models from all providers and update cache.

    This is called automatically after model configuration changes
    or when explicitly requested by user.
    """
    config_manager = ConfigManager()
    discovery = ModelDiscovery(config_manager, quiet=quiet)

    try:
        if not quiet:
            display_info("Discovering models from all providers...")
        models_by_provider = await discovery.discover_models_for_all_providers()

        # Update cache
        cache = config_manager.get_models_cache()
        cache["discovered_models"] = models_by_provider
        cache["last_updated"] = datetime.datetime.now().isoformat()
        config_manager.update_models_cache(cache)

        if not quiet:
            display_success(
                f"Model cache updated with {len(models_by_provider)} provider(s)"
            )
    finally:
        await discovery.close()
