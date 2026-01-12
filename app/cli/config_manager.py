"""
Configuration Manager for Model-Proxy CLI.

This module provides safe, validated operations on JSON configuration files
for providers, models, and cached data.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

from app.core.config_paths import get_config_search_paths, get_writable_config_dir


def _find_config_dir() -> Path:
    """
    Find the config directory by searching multiple locations.

    Search order:
    1. Package-relative (for source installs): <package_root>/config
    2. Current working directory: ./config
    3. User home directory: ~/.model-proxy/config

    Returns:
        Path to config directory (creates if needed)
    """
    # Backward-compatible wrapper for shared config path logic.
    return get_writable_config_dir()


class ConfigManager:
    """
    Manages JSON configuration files with validation.

    This class provides a unified interface for reading and writing
    configuration files with automatic validation and error handling.
    """

    # Required fields for provider configurations
    PROVIDER_REQUIRED_FIELDS = [
        "name",
        "display_name",
        "api_keys",
        "endpoints",
    ]

    # Required fields for model configurations
    MODEL_REQUIRED_FIELDS = [
        "logical_name",
        "timeout_seconds",
        "model_routings",
    ]

    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize the configuration manager.

        Args:
            config_dir: Path to config directory (default: auto-detect)
        """
        if config_dir is None:
            self.config_dir = get_writable_config_dir()
            self.search_paths = get_config_search_paths()
        else:
            self.config_dir = Path(config_dir)
            self.search_paths = [self.config_dir]
        self.providers_dir = self.config_dir / "providers"
        self.models_dir = self.config_dir / "models"
        self.cache_file = self.config_dir / "models.json"

        # Ensure directories exist
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Ensure required directories exist."""
        self.providers_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def _find_config_file(self, relative_path: Path) -> Optional[Path]:
        """Find a config file by searching this manager's paths."""
        for root in self.search_paths:
            candidate = root / relative_path
            if candidate.exists():
                return candidate
        return None

    def get_providers(self) -> Dict[str, Dict]:
        """
        Load all provider configurations.

        Returns:
            Dictionary mapping provider names to their configurations

        Raises:
            FileNotFoundError: If providers directory doesn't exist
            json.JSONDecodeError: If configuration files are invalid JSON
        """
        providers: Dict[str, Dict] = {}

        for root in self.search_paths:
            providers_dir = root / "providers"
            if not providers_dir.exists():
                continue

            for provider_file in providers_dir.glob("*.json"):
                try:
                    with open(provider_file, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    # Prefer the first occurrence by search path precedence.
                    if "name" in data and data["name"] not in providers:
                        providers[data["name"]] = data

                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON in {provider_file}: {e}")
                except Exception as e:
                    print(f"Warning: Could not load provider from {provider_file}: {e}")

        return providers

    def get_provider(self, provider_name: str) -> Optional[Dict]:
        """
        Load a specific provider configuration.

        Args:
            provider_name: Name of the provider to load

        Returns:
            Provider configuration or None if not found
        """
        provider_file = self._find_config_file(
            Path("providers") / f"{provider_name}.json"
        )
        if not provider_file:
            return None

        try:
            with open(provider_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load provider {provider_name}: {e}")

    def get_models(self) -> List[str]:
        """
        Load all available model configuration names.

        Returns:
            Sorted list of model configuration names
        """
        models = set()

        for root in self.search_paths:
            models_dir = root / "models"
            if not models_dir.exists():
                continue
            for model_file in models_dir.glob("*.json"):
                models.add(model_file.stem)

        return sorted(models)

    def get_model_config(self, model_name: str) -> Dict:
        """
        Load a specific model configuration.

        Args:
            model_name: Name of the model configuration to load

        Returns:
            Model configuration dictionary

        Raises:
            FileNotFoundError: If model configuration doesn't exist
            ValueError: If configuration is invalid
        """
        model_file = self._find_config_file(Path("models") / f"{model_name}.json")
        if not model_file:
            raise FileNotFoundError(f"Model config not found: {model_name}")

        try:
            with open(model_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            return data
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in model config {model_name}: {e}")

    def save_provider(self, provider_data: Dict, overwrite: bool = False) -> None:
        """
        Save provider configuration.

        Args:
            provider_data: Provider configuration dictionary
            overwrite: Whether to overwrite existing provider

        Raises:
            ValueError: If validation fails or provider exists without overwrite
            IOError: If file cannot be written
        """
        # Validate provider configuration
        self._validate_provider_config(provider_data)

        provider_name = provider_data["name"]
        provider_file = self.providers_dir / f"{provider_name}.json"
        existing_path = self._find_config_file(
            Path("providers") / f"{provider_name}.json"
        )

        # Check if provider exists anywhere in the search path
        if existing_path and not overwrite:
            raise ValueError(
                f"Provider '{provider_name}' already exists. "
                f"Use overwrite=True to replace it."
            )

        # Create directory if needed
        self.providers_dir.mkdir(parents=True, exist_ok=True)

        # Save configuration
        try:
            with open(provider_file, "w", encoding="utf-8") as f:
                json.dump(provider_data, f, indent=2)
        except Exception as e:
            raise IOError(f"Failed to save provider {provider_name}: {e}")

    def save_model_config(
        self, model_name: str, config: Dict, overwrite: bool = False
    ) -> None:
        """
        Save model configuration.

        Args:
            model_name: Name of the model configuration
            config: Model configuration dictionary
            overwrite: Whether to overwrite existing configuration

        Raises:
            ValueError: If validation fails or model exists without overwrite
            IOError: If file cannot be written
        """
        # Validate model configuration
        self._validate_model_config(config)

        model_file = self.models_dir / f"{model_name}.json"
        existing_path = self._find_config_file(Path("models") / f"{model_name}.json")

        # Check if model exists anywhere in the search path
        if existing_path and not overwrite:
            raise ValueError(
                f"Model '{model_name}' already exists. "
                f"Use overwrite=True to replace it."
            )

        # Create directory if needed
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Save configuration
        try:
            with open(model_file, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            raise IOError(f"Failed to save model {model_name}: {e}")

    def update_models_cache(self, cache_data: Dict) -> None:
        """
        Update models.json cache.

        Args:
            cache_data: Cache dictionary with discovered models

        Raises:
            IOError: If file cannot be written
        """
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, indent=2)
        except Exception as e:
            raise IOError(f"Failed to update models cache: {e}")

    def get_models_cache(self) -> Dict:
        """
        Load models.json cache.

        Returns:
            Cache dictionary (empty if file doesn't exist)
        """
        if not self.cache_file.exists():
            return {"discovered_models": {}, "custom_models": {}, "last_updated": None}

        try:
            with open(self.cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            # If cache is corrupted, return empty
            return {"discovered_models": {}, "custom_models": {}, "last_updated": None}

    def delete_provider(self, provider_name: str) -> bool:
        """
        Delete a provider configuration.

        Args:
            provider_name: Name of the provider to delete

        Returns:
            True if deleted, False if not found
        """
        provider_file = self.providers_dir / f"{provider_name}.json"

        if not provider_file.exists():
            return False

        try:
            provider_file.unlink()
            return True
        except Exception:
            return False

    def delete_model_config(self, model_name: str) -> bool:
        """
        Delete a model configuration.

        Args:
            model_name: Name of the model configuration to delete

        Returns:
            True if deleted, False if not found
        """
        model_file = self.models_dir / f"{model_name}.json"

        if not model_file.exists():
            return False

        try:
            model_file.unlink()
            return True
        except Exception:
            return False

    def _validate_provider_config(self, config: Dict) -> None:
        """
        Validate provider configuration structure.

        Args:
            config: Provider configuration dictionary

        Raises:
            ValueError: If configuration is invalid
        """
        # Check required fields
        for field in self.PROVIDER_REQUIRED_FIELDS:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")

        # Validate provider name
        if not config["name"]:
            raise ValueError("Provider name cannot be empty")

        # Validate display name
        if not config["display_name"]:
            raise ValueError("Display name cannot be empty")

        # Validate api_keys structure
        api_keys = config["api_keys"]
        if "env_var_patterns" not in api_keys:
            raise ValueError("api_keys must have 'env_var_patterns' field")

        # Validate endpoints structure
        endpoints = config["endpoints"]
        if "base_url" not in endpoints:
            raise ValueError("endpoints must have 'base_url' field")

        # Validate URL format
        base_url = endpoints["base_url"]
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")

    def _validate_model_config(self, config: Dict) -> None:
        """
        Validate model configuration structure.

        Args:
            config: Model configuration dictionary

        Raises:
            ValueError: If configuration is invalid
        """
        # Check required fields
        for field in self.MODEL_REQUIRED_FIELDS:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")

        # Validate logical name
        if not config["logical_name"]:
            raise ValueError("Logical name cannot be empty")

        # Validate timeout
        timeout = config.get("timeout_seconds", 0)
        if not isinstance(timeout, int) or timeout <= 0:
            raise ValueError("timeout_seconds must be a positive integer")

        # Validate model_routings
        routings = config["model_routings"]
        if not isinstance(routings, list) or not routings:
            raise ValueError("model_routings must be a non-empty list")

        for routing in routings:
            if not isinstance(routing, dict):
                raise ValueError("Each routing in model_routings must be a dictionary")
            if "provider" not in routing or "model" not in routing:
                raise ValueError("Each routing must have 'provider' and 'model' fields")

    def provider_exists(self, provider_name: str) -> bool:
        """
        Check if a provider configuration exists.

        Args:
            provider_name: Name of the provider

        Returns:
            True if provider exists
        """
        return (
            self._find_config_file(Path("providers") / f"{provider_name}.json")
            is not None
        )

    def model_config_exists(self, model_name: str) -> bool:
        """
        Check if a model configuration exists.

        Args:
            model_name: Name of the model configuration

        Returns:
            True if model configuration exists
        """
        return self._find_config_file(Path("models") / f"{model_name}.json") is not None

    def get_config_stats(self) -> Dict:
        """
        Get statistics about current configuration.

        Returns:
            Dictionary with configuration statistics
        """
        stats = {
            "total_providers": 0,
            "enabled_providers": 0,
            "total_models": 0,
            "providers_with_keys": 0,
            "custom_models_count": 0,
        }

        # Provider stats
        providers = self.get_providers()
        stats["total_providers"] = len(providers)

        for name, config in providers.items():
            if config.get("enabled", False):
                stats["enabled_providers"] += 1

            # Check for API keys in environment
            import os

            provider_upper = name.upper()
            patterns = config.get("api_keys", {}).get("env_var_patterns", [])
            has_key = False

            for pattern in patterns:
                try:
                    if "{INDEX}" in pattern:
                        # Check a few indices
                        for i in range(1, 4):
                            env_var = pattern.format(PROVIDER=provider_upper, INDEX=i)
                            if os.getenv(env_var):
                                has_key = True
                                break
                    else:
                        env_var = pattern.format(PROVIDER=provider_upper)
                        if os.getenv(env_var):
                            has_key = True
                except KeyError:
                    pass

            if has_key:
                stats["providers_with_keys"] += 1

        # Model stats
        stats["total_models"] = len(self.get_models())

        # Cache stats
        cache = self.get_models_cache()
        custom_models = cache.get("custom_models", {})
        stats["custom_models_count"] = sum(
            len(models) for models in custom_models.values()
        )

        return stats
