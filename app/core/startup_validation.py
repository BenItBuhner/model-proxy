"""
Startup validation for ensuring the application is properly configured.
"""
import os
from typing import List, Tuple
from app.core.provider_config import get_all_provider_configs, is_provider_enabled
from app.core.api_key_manager import get_available_keys
from app.core.model_resolver import get_available_models
from app.database.database import engine
from sqlalchemy import text


class StartupValidationError(Exception):
    """Raised when startup validation fails."""
    pass


def validate_database() -> Tuple[bool, str]:
    """
    Validate database connectivity.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True, ""
    except Exception as e:
        return False, f"Database connection failed: {str(e)}"


def validate_client_api_key() -> Tuple[bool, str]:
    """
    Validate client API key is configured.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    client_key = os.getenv("CLIENT_API_KEY")
    require_key = os.getenv("REQUIRE_CLIENT_API_KEY", "false").lower() == "true"
    
    if require_key and not client_key:
        return False, "CLIENT_API_KEY is required but not set"
    
    return True, ""


def validate_provider_configs() -> Tuple[bool, List[str]]:
    """
    Validate all provider configurations are valid.
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    try:
        configs = get_all_provider_configs()
        if not configs:
            errors.append("No provider configurations found")
            return False, errors
        
        for provider_name, config in configs.items():
            # Check required fields
            required_fields = ["name", "enabled", "endpoints", "authentication"]
            for field in required_fields:
                if field not in config:
                    errors.append(f"Provider '{provider_name}' missing required field: {field}")
            
            # Check endpoints
            if "endpoints" in config and "base_url" not in config["endpoints"]:
                errors.append(f"Provider '{provider_name}' missing base_url in endpoints")
            
            # Check authentication
            if "authentication" in config:
                auth = config["authentication"]
                if "header_name" not in auth:
                    errors.append(f"Provider '{provider_name}' missing header_name in authentication")
        
        return len(errors) == 0, errors
    except Exception as e:
        return False, [f"Failed to load provider configs: {str(e)}"]


def validate_provider_api_keys() -> Tuple[bool, List[str]]:
    """
    Validate at least one provider has API keys configured.
    
    Returns:
        Tuple of (is_valid, list_of_warnings)
    """
    warnings = []
    try:
        configs = get_all_provider_configs()
        providers_with_keys = 0
        
        for provider_name in configs.keys():
            if is_provider_enabled(provider_name):
                available_keys = get_available_keys(provider_name)
                if len(available_keys) == 0:
                    warnings.append(f"Provider '{provider_name}' is enabled but has no API keys configured")
                else:
                    providers_with_keys += 1
        
        if providers_with_keys == 0:
            return False, warnings + ["No providers have API keys configured"]
        
        return True, warnings
    except Exception as e:
        return False, [f"Failed to validate provider API keys: {str(e)}"]


def validate_model_config() -> Tuple[bool, str]:
    """
    Validate model configuration is valid.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        models = get_available_models()
        if len(models) == 0:
            return False, "No models configured"
        return True, ""
    except Exception as e:
        return False, f"Model configuration invalid: {str(e)}"


def validate_startup() -> None:
    """
    Perform all startup validations.
    
    Raises:
        StartupValidationError: If any critical validation fails
    """
    errors = []
    warnings = []
    
    # Validate database
    db_valid, db_error = validate_database()
    if not db_valid:
        errors.append(f"Database: {db_error}")
    
    # Validate client API key (warning only unless required)
    client_key_valid, client_key_error = validate_client_api_key()
    if not client_key_valid:
        errors.append(f"Client API Key: {client_key_error}")
    
    # Validate provider configs
    configs_valid, config_errors = validate_provider_configs()
    if not configs_valid:
        errors.extend([f"Provider Config: {e}" for e in config_errors])
    
    # Validate provider API keys (warnings)
    keys_valid, key_warnings = validate_provider_api_keys()
    if not keys_valid:
        errors.extend([f"Provider API Keys: {w}" for w in key_warnings])
    else:
        warnings.extend(key_warnings)
    
    # Validate model config
    model_valid, model_error = validate_model_config()
    if not model_valid:
        errors.append(f"Model Config: {model_error}")
    
    # Print warnings
    for warning in warnings:
        print(f"Warning: {warning}")
    
    # Raise error if any critical validations failed
    if errors:
        error_msg = "Startup validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        raise StartupValidationError(error_msg)

