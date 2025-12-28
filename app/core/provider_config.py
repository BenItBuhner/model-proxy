"""
Provider configuration loader and manager.
Loads provider configurations from JSON files and provides access to settings.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List

# Cache for loaded provider configs
_provider_configs: Dict[str, Dict[str, Any]] = {}


def _get_provider_config_path(provider_name: str) -> Path:
    """
    Get the path to a provider configuration file.

    Args:
        provider_name: Provider name (e.g., "openai", "anthropic")

    Returns:
        Path to provider config file
    """
    config_dir = Path(__file__).parent.parent.parent / "config" / "providers"
    return config_dir / f"{provider_name}.json"


def load_provider_config(provider_name: str) -> Dict[str, Any]:
    """
    Load and validate a provider configuration from JSON file.

    Args:
        provider_name: Provider name

    Returns:
        Provider configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    config_path = _get_provider_config_path(provider_name)

    if not config_path.exists():
        raise FileNotFoundError(f"Provider config not found: {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    # Validate required fields
    required_fields = ["name", "enabled", "endpoints", "authentication"]
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Provider config missing required field: {field}")

    # Validate endpoints
    if "base_url" not in config["endpoints"]:
        raise ValueError("Provider config missing base_url in endpoints")

    # Store in cache
    _provider_configs[provider_name] = config

    return config


def get_all_provider_configs() -> Dict[str, Dict[str, Any]]:
    """
    Load all provider configuration files.

    Returns:
        Dictionary mapping provider names to their configs
    """
    config_dir = Path(__file__).parent.parent.parent / "config" / "providers"

    if not config_dir.exists():
        return {}

    configs = {}
    for config_file in config_dir.glob("*.json"):
        provider_name = config_file.stem
        try:
            configs[provider_name] = load_provider_config(provider_name)
        except Exception as e:
            # Log error but continue loading other configs
            print(f"Warning: Failed to load config for {provider_name}: {e}")

    return configs


def get_provider_config(provider_name: str) -> Optional[Dict[str, Any]]:
    """
    Get a provider configuration (from cache if available).

    Args:
        provider_name: Provider name

    Returns:
        Provider configuration or None if not found
    """
    if provider_name in _provider_configs:
        return _provider_configs[provider_name]

    try:
        return load_provider_config(provider_name)
    except FileNotFoundError:
        return None


def reload_provider_config(provider_name: str) -> Dict[str, Any]:
    """
    Reload a provider configuration (useful for hot-reloading).

    Args:
        provider_name: Provider name

    Returns:
        Reloaded provider configuration
    """
    # Clear from cache
    if provider_name in _provider_configs:
        del _provider_configs[provider_name]

    return load_provider_config(provider_name)


def validate_provider_config(config: Dict[str, Any]) -> bool:
    """
    Validate a provider configuration schema.

    Args:
        config: Configuration dictionary

    Returns:
        True if valid

    Raises:
        ValueError: If config is invalid
    """
    required_fields = ["name", "enabled", "endpoints", "authentication"]
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field: {field}")

    if "base_url" not in config["endpoints"]:
        raise ValueError("Missing base_url in endpoints")

    return True


def _substitute_env_vars(text: str) -> str:
    """Substitute environment variables in text using ${VAR_NAME} syntax."""
    import re

    def replace_var(match):
        var_name = match.group(1)
        return os.getenv(var_name, match.group(0))  # Return original if not found

    return re.sub(r"\$\{([^}]+)\}", replace_var, text)


def get_provider_endpoint(
    provider_name: str, endpoint_type: str = "completions"
) -> str:
    """
    Get the full endpoint URL for a provider.

    Args:
        provider_name: Provider name
        endpoint_type: Type of endpoint ("completions" or "streaming")

    Returns:
        Full endpoint URL
    """
    config = get_provider_config(provider_name)
    if not config:
        raise ValueError(f"Provider config not found: {provider_name}")

    endpoints = config["endpoints"]
    base_url = _substitute_env_vars(endpoints["base_url"])

    # Check for proxy override
    if config.get("proxy_support", {}).get("enabled", False):
        override_url = config["proxy_support"].get("base_url_override")
        if override_url:
            base_url = _substitute_env_vars(override_url)

    # Get endpoint path
    if endpoint_type == "streaming":
        endpoint_path = endpoints.get("streaming", endpoints.get("completions", ""))
    else:
        endpoint_path = endpoints.get("completions", "")

    # Remove leading slash if present (base_url should end with / or endpoint should start with /)
    if endpoint_path.startswith("/"):
        endpoint_path = endpoint_path[1:]

    if base_url.endswith("/"):
        return f"{base_url}{endpoint_path}"
    else:
        return f"{base_url}/{endpoint_path}"


def get_provider_auth_headers(provider_name: str, api_key: str) -> Dict[str, str]:
    """
    Build authentication headers for a provider.

    Args:
        provider_name: Provider name
        api_key: API key to use

    Returns:
        Dictionary of headers for authentication
    """
    config = get_provider_config(provider_name)
    if not config:
        raise ValueError(f"Provider config not found: {provider_name}")

    auth_config = config["authentication"]
    headers = {}

    header_name = auth_config["header_name"]
    header_format = auth_config.get("header_format", "{api_key}")

    # Format the header value
    header_value = header_format.format(api_key=api_key)
    headers[header_name] = header_value

    # Add additional headers if specified
    if "additional_headers" in auth_config:
        for key, value in auth_config["additional_headers"].items():
            # Substitute environment variables in header values
            headers[key] = _substitute_env_vars(str(value))

    return headers


def get_provider_env_var_patterns(provider_name: str) -> List[str]:
    """
    Get environment variable patterns for a provider's API keys.

    Args:
        provider_name: Provider name

    Returns:
        List of environment variable patterns
    """
    config = get_provider_config(provider_name)
    if not config:
        return []

    api_keys_config = config.get("api_keys", {})
    patterns = api_keys_config.get("env_var_patterns", [])

    # Replace {PROVIDER} placeholder with actual provider name
    provider_upper = provider_name.upper().replace("-", "_")
    resolved_patterns = []

    for pattern in patterns:
        resolved = pattern.replace("{PROVIDER}", provider_upper)
        resolved_patterns.append(resolved)

    return resolved_patterns


def get_provider_wire_protocol(provider_name: str) -> str:
    """
    Resolve the default wire protocol for a provider.

    Uses the provider's endpoints.compatible_format and maps it to a routing
    wire protocol ("openai" or "anthropic"). Non-matching formats default
    to "openai" for routing.
    """
    config = get_provider_config(provider_name)
    if not config:
        return "openai"

    endpoints = config.get("endpoints", {})
    compatible_format = endpoints.get("compatible_format")
    if not compatible_format:
        return "openai"

    normalized = str(compatible_format).strip().lower()
    mapping = {
        "openai": "openai",
        "anthropic": "anthropic",
        "azure": "openai",
        "native": "openai",
    }
    return mapping.get(normalized, "openai")


def is_provider_enabled(provider_name: str) -> bool:
    """
    Check if a provider is enabled.

    Args:
        provider_name: Provider name

    Returns:
        True if enabled, False otherwise
    """
    config = get_provider_config(provider_name)
    if not config:
        return False

    return config.get("enabled", True)
