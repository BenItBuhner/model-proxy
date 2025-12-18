"""
Health check endpoints for system monitoring.
"""

import time
from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from sqlalchemy.orm import Session

from app.core.api_key_manager import get_available_keys
from app.core.provider_config import get_all_provider_configs, get_provider_config
from app.database.database import SessionLocal, engine
from app.routing.config_loader import config_loader

router = APIRouter()

# Track application start time for uptime calculation
_app_start_time = time.time()


def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def check_database() -> Dict[str, Any]:
    """Check database connectivity."""
    try:
        from sqlalchemy import text

        start_time = time.time()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        response_time_ms = int((time.time() - start_time) * 1000)
        return {"status": "healthy", "response_time_ms": response_time_ms}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


def check_providers() -> Dict[str, Any]:
    """Check provider API key availability."""
    providers_status = {}
    try:
        provider_configs = get_all_provider_configs()
        for provider_name in provider_configs.keys():
            try:
                available_keys = get_available_keys(provider_name)
                keys_count = len(available_keys)
                providers_status[provider_name] = {
                    "status": "healthy" if keys_count > 0 else "unhealthy",
                    "keys_available": keys_count,
                }
            except Exception as e:
                providers_status[provider_name] = {
                    "status": "unhealthy",
                    "error": str(e),
                }
    except Exception as e:
        return {"error": str(e)}

    return providers_status


def check_model_config() -> Dict[str, Any]:
    """Check model configuration."""
    try:
        models = config_loader.get_available_models()
        return {"status": "healthy", "models_count": len(models)}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


def check_provider_configs() -> Dict[str, Any]:
    """Check provider configuration files."""
    try:
        configs = get_all_provider_configs()
        enabled_count = sum(1 for cfg in configs.values() if cfg.get("enabled", True))
        return {
            "status": "healthy",
            "providers_loaded": len(configs),
            "providers_enabled": enabled_count,
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


def get_overall_status(components: Dict[str, Any]) -> str:
    """Determine overall system status."""
    unhealthy_count = 0
    degraded_count = 0

    # Check database
    if components.get("database", {}).get("status") != "healthy":
        unhealthy_count += 1

    # Check providers
    providers = components.get("providers", {})
    for provider_status in providers.values():
        if isinstance(provider_status, dict):
            if provider_status.get("status") == "unhealthy":
                unhealthy_count += 1
            elif provider_status.get("keys_available", 0) == 0:
                degraded_count += 1

    # Check model config
    if components.get("model_config", {}).get("status") != "healthy":
        unhealthy_count += 1

    # Check provider configs
    if components.get("provider_configs", {}).get("status") != "healthy":
        unhealthy_count += 1

    if unhealthy_count > 0:
        return "unhealthy"
    elif degraded_count > 0:
        return "degraded"
    else:
        return "healthy"


@router.get("/health")
async def health_check():
    """
    Basic health check endpoint.
    Returns 200 if system is healthy, 503 if unhealthy.
    """
    database_status = check_database()

    if database_status.get("status") != "healthy":
        raise HTTPException(status_code=503, detail="Database unavailable")

    return {
        "status": "healthy",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


@router.get("/health/detailed")
async def detailed_health_check():
    """
    Detailed health check with component status.
    Returns 200 if healthy, 503 if degraded or unhealthy.
    """
    components = {
        "database": check_database(),
        "providers": check_providers(),
        "model_config": check_model_config(),
        "provider_configs": check_provider_configs(),
    }

    overall_status = get_overall_status(components)
    uptime_seconds = int(time.time() - _app_start_time)

    response = {
        "status": overall_status,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "uptime_seconds": uptime_seconds,
        "components": components,
    }

    status_code = 200 if overall_status == "healthy" else 503

    if status_code == 503:
        raise HTTPException(status_code=503, detail=response)

    return response
