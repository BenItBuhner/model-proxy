"""
Configuration loader for model routing.
Loads and caches JSON configurations from config/models/ directory.
"""

import json
import time
from pathlib import Path
from typing import Dict, Optional, Any
from app.routing.models import ModelRoutingConfig


class ModelConfigLoader:
    """Loads and caches model routing configurations."""

    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize the config loader.

        Args:
            config_dir: Directory containing model JSON configs. Defaults to config/models/
        """
        if config_dir is None:
            # Default to config/models relative to project root
            self.config_dir = Path(__file__).parent.parent.parent / "config" / "models"
        else:
            self.config_dir = config_dir

        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_timestamps: Dict[str, float] = {}
        self._config_cache: Dict[str, ModelRoutingConfig] = {}

    def load_config(
        self, logical_model: str, force_reload: bool = False
    ) -> ModelRoutingConfig:
        """
        Load configuration for a logical model.

        Args:
            logical_model: The logical model name (e.g., 'glm-4.6')
            force_reload: Force reload from disk even if cached

        Returns:
            ModelRoutingConfig: The validated configuration

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is invalid
        """
        config_path = self.config_dir / f"{logical_model}.json"

        # Check if we have a valid cached version
        if not force_reload and logical_model in self._config_cache:
            if self._is_cache_valid(logical_model, config_path):
                return self._config_cache[logical_model]

        # Load from disk
        if not config_path.exists():
            available_models = self._get_available_models()
            raise FileNotFoundError(
                f"Configuration file not found for model '{logical_model}': {config_path}. "
                f"Available models: {available_models}"
            )

        try:
            with open(config_path, "r") as f:
                raw_config = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {config_path}: {e}")

        # Validate with Pydantic
        try:
            config = ModelRoutingConfig(**raw_config)
        except Exception as e:
            raise ValueError(f"Invalid configuration in {config_path}: {e}")

        # Validate that logical_name matches filename
        if config.logical_name != logical_model:
            raise ValueError(
                f"Logical name mismatch in {config_path}: "
                f"filename suggests '{logical_model}' but config has '{config.logical_name}'"
            )

        # Cache the result
        self._config_cache[logical_model] = config
        self._cache_timestamps[logical_model] = time.time()

        return config

    def _is_cache_valid(self, logical_model: str, config_path: Path) -> bool:
        """Check if cached config is still valid."""
        if logical_model not in self._cache_timestamps:
            return False

        # Check if file has been modified since caching
        try:
            file_mtime = config_path.stat().st_mtime
            return self._cache_timestamps[logical_model] >= file_mtime
        except OSError:
            # File no longer exists or inaccessible
            return False

    def _get_available_models(self) -> list[str]:
        """Get list of available model configurations."""
        if not self.config_dir.exists():
            return []

        models = []
        for file_path in self.config_dir.glob("*.json"):
            if file_path.is_file():
                model_name = file_path.stem
                models.append(model_name)

        return sorted(models)

    def clear_cache(self):
        """Clear all cached configurations."""
        self._cache.clear()
        self._cache_timestamps.clear()
        self._config_cache.clear()

    def reload_config(self, logical_model: str) -> ModelRoutingConfig:
        """Force reload a specific configuration."""
        return self.load_config(logical_model, force_reload=True)

    def get_available_models(self) -> list[str]:
        """Get list of all available logical models."""
        return self._get_available_models()


# Global instance for easy access
config_loader = ModelConfigLoader()
