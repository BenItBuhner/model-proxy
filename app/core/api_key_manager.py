"""
API Key Manager for handling multiple API keys per provider with fallback.
Parses environment variables and manages key rotation with circuit breaker pattern.
Uses provider configuration for environment variable patterns.
"""
import os
import random
import time
from typing import Dict, List, Optional
from collections import defaultdict
from app.core.provider_config import get_provider_env_var_patterns

# Failed keys tracking: {provider: {key: failure_timestamp}}
_failed_keys: Dict[str, Dict[str, float]] = defaultdict(dict)

# Cooldown disabled by default for development (can be overridden via env)
KEY_COOLDOWN_SECONDS = int(os.getenv("KEY_COOLDOWN_SECONDS", "0"))


def _parse_provider_keys(provider_name: str) -> List[str]:
    """
    Parse environment variables for a provider's API keys.
    Uses provider configuration to determine env var patterns.
    Falls back to default pattern if config not available.
    
    Args:
        provider_name: Provider name (e.g., "openai", "anthropic")
        
    Returns:
        List of API keys found
    """
    keys = []
    
    # Try to get patterns from provider config
    try:
        patterns = get_provider_env_var_patterns(provider_name)
    except Exception:
        # Fallback to default pattern if config not available
        patterns = []
    
    # If no patterns from config, use default pattern
    if not patterns:
        env_prefix = provider_name.upper().replace("-", "_")
        patterns = [
            f"{env_prefix}_API_KEY",
            f"{env_prefix}_API_KEY_{{INDEX}}"
        ]
    
    # Parse keys based on patterns
    for pattern in patterns:
        if "{INDEX}" in pattern:
            # Pattern with index placeholder (e.g., OPENAI_API_KEY_{INDEX})
            base_pattern = pattern.replace("{INDEX}", "")
            index = 1
            while True:
                env_var = f"{base_pattern}{index}"
                key = os.getenv(env_var)
                if not key:
                    break
                keys.append(key)
                index += 1
        else:
            # Simple pattern without index (e.g., OPENAI_API_KEY)
            key = os.getenv(pattern)
            if key:
                keys.append(key)
    
    return keys


def get_available_keys(provider: str) -> List[str]:
    """
    Get list of available keys for a provider.
    Cooldown is disabled by default; returns all parsed keys.
    """
    return _parse_provider_keys(provider)


def get_api_key(provider: str) -> Optional[str]:
    """
    Get an available API key for a provider (random selection).
    
    Args:
        provider: Provider name
        
    Returns:
        API key string, or None if no keys available
    """
    available_keys = get_available_keys(provider)
    if not available_keys:
        return None
    
    return random.choice(available_keys)


def mark_key_failed(provider: str, key: str) -> None:
    """
    Mark key failed (no-op while cooldown disabled). Left in place for compatibility.
    """
    return


def get_all_keys(provider: str) -> List[str]:
    """
    Get all keys for a provider (including failed ones).
    
    Args:
        provider: Provider name
        
    Returns:
        List of all API keys
    """
    return _parse_provider_keys(provider)


def reset_failed_keys(provider: Optional[str] = None) -> None:
    """
    Reset failed keys for a provider (or all providers if None).
    Useful for testing or manual recovery.
    
    Args:
        provider: Provider name, or None to reset all
    """
    if provider:
        _failed_keys[provider].clear()
    else:
        _failed_keys.clear()

