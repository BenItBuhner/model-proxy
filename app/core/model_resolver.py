"""
Model resolver that maps requested models to providers.
Loads model configuration from config/models.json at startup.
"""
import json
import os
from pathlib import Path
from typing import Dict, Optional
from fastapi import HTTPException

# Load model configuration at module import time
_config_path = Path(__file__).parent.parent.parent / "config" / "models.json"

if not _config_path.exists():
    raise FileNotFoundError(f"Model configuration file not found: {_config_path}")

with open(_config_path, "r") as f:
    _MODEL_CONFIG: Dict[str, Dict[str, str]] = json.load(f)


def resolve_model(model_name: str) -> Dict[str, str]:
    """
    Resolve a model name to its provider and actual provider model name.
    
    Args:
        model_name: The requested model name
        
    Returns:
        Dictionary with 'provider' and 'provider_model' keys
        
    Raises:
        HTTPException: 400 if model not found in configuration
    """
    if model_name not in _MODEL_CONFIG:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_name}' not found in configuration. Available models: {', '.join(_MODEL_CONFIG.keys())}"
        )
    
    return _MODEL_CONFIG[model_name]


def get_available_models() -> list:
    """
    Get list of all available model names.
    
    Returns:
        List of model names
    """
    return list(_MODEL_CONFIG.keys())

