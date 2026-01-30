"""
GUI Setup Wizard Routes for Model-Proxy.

Provides FastAPI endpoints for the web-based setup wizard and config export/import.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from app.cli.config_manager import ConfigManager
from app.core.api_key_manager import get_available_keys
from app.core.config_paths import get_writable_config_dir

# Create router
router = APIRouter(prefix="/setup", tags=["setup"])

# Initialize config manager
config_manager = ConfigManager()


# Pydantic models for API requests
class ProviderConfig(BaseModel):
    name: str
    display_name: str
    enabled: bool = True
    api_keys: Dict
    endpoints: Dict
    authentication: Optional[Dict] = None
    request_config: Optional[Dict] = None
    rate_limiting: Optional[Dict] = None
    error_handling: Optional[Dict] = None


class ModelRouting(BaseModel):
    provider: str
    model: str


class ModelConfig(BaseModel):
    logical_name: str
    timeout_seconds: int = 180
    default_cooldown_seconds: int = 180
    model_routings: List[ModelRouting]
    fallback_model_routings: Optional[List[ModelRouting]] = []


class ApiKeyEntry(BaseModel):
    provider: str
    env_var: str
    key_value: str


class SetupExport(BaseModel):
    version: str = "1.0.0"
    providers: List[ProviderConfig]
    models: List[ModelConfig]
    environment: Dict[str, str]


class AuthRequest(BaseModel):
    client_api_key: str


# Authentication dependency
def verify_auth_token(request: Request):
    """Verify the client API key from the Authorization header."""
    auth_header = request.headers.get("Authorization", "")
    stored_key = os.getenv("CLIENT_API_KEY", "")

    if not stored_key:
        raise HTTPException(
            status_code=401, detail="CLIENT_API_KEY not configured on server"
        )

    # Support "Bearer <token>" format or plain token
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
    else:
        token = auth_header

    if token != stored_key:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return True


@router.get("/", response_class=HTMLResponse)
async def setup_wizard_page():
    """Serve the main setup wizard HTML page."""
    html_path = Path(__file__).parent / "static" / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=500, detail="Setup UI not found")

    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


@router.get("/api/status")
async def get_setup_status(auth: bool = Depends(verify_auth_token)):
    """Get the current setup status and configuration statistics."""
    try:
        stats = config_manager.get_config_stats()
        return {
            "status": "ok",
            "stats": stats,
            "config_dir": str(config_manager.config_dir),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/providers")
async def get_providers(auth: bool = Depends(verify_auth_token)):
    """Get all provider configurations."""
    try:
        providers = config_manager.get_providers()
        return {
            "providers": providers,
            "count": len(providers),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/providers/{provider_name}")
async def get_provider(provider_name: str, auth: bool = Depends(verify_auth_token)):
    """Get a specific provider configuration."""
    try:
        provider = config_manager.get_provider(provider_name)
        if not provider:
            raise HTTPException(
                status_code=404, detail=f"Provider '{provider_name}' not found"
            )
        return provider
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/providers")
async def create_provider(
    provider: ProviderConfig,
    overwrite: bool = False,
    auth: bool = Depends(verify_auth_token),
):
    """Create or update a provider configuration."""
    try:
        provider_dict = provider.model_dump()
        config_manager.save_provider(provider_dict, overwrite=overwrite)
        return {
            "status": "success",
            "message": f"Provider '{provider.name}' saved successfully",
            "provider": provider_dict,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/providers/{provider_name}")
async def delete_provider(provider_name: str, auth: bool = Depends(verify_auth_token)):
    """Delete a provider configuration."""
    try:
        success = config_manager.delete_provider(provider_name)
        if not success:
            raise HTTPException(
                status_code=404, detail=f"Provider '{provider_name}' not found"
            )
        return {
            "status": "success",
            "message": f"Provider '{provider_name}' deleted",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/models")
async def get_models(auth: bool = Depends(verify_auth_token)):
    """Get all model configurations."""
    try:
        model_names = config_manager.get_models()
        models = []
        for name in model_names:
            try:
                config = config_manager.get_model_config(name)
                models.append(
                    {
                        "name": name,
                        "config": config,
                    }
                )
            except Exception:
                # Skip invalid models
                pass

        return {
            "models": models,
            "count": len(models),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/models/{model_name}")
async def get_model(model_name: str, auth: bool = Depends(verify_auth_token)):
    """Get a specific model configuration."""
    try:
        config = config_manager.get_model_config(model_name)
        return {
            "name": model_name,
            "config": config,
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/models")
async def create_model(
    model: ModelConfig, overwrite: bool = False, auth: bool = Depends(verify_auth_token)
):
    """Create or update a model configuration."""
    try:
        model_dict = model.model_dump()
        config_manager.save_model_config(
            model.logical_name, model_dict, overwrite=overwrite
        )
        return {
            "status": "success",
            "message": f"Model '{model.logical_name}' saved successfully",
            "model": model_dict,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/models/{model_name}")
async def delete_model(model_name: str, auth: bool = Depends(verify_auth_token)):
    """Delete a model configuration."""
    try:
        success = config_manager.delete_model_config(model_name)
        if not success:
            raise HTTPException(
                status_code=404, detail=f"Model '{model_name}' not found"
            )
        return {
            "status": "success",
            "message": f"Model '{model_name}' deleted",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/providers/{provider_name}/keys")
async def get_provider_keys_status(
    provider_name: str, auth: bool = Depends(verify_auth_token)
):
    """Get API key status for a provider (without exposing key values)."""
    try:
        keys = get_available_keys(provider_name)
        return {
            "provider": provider_name,
            "key_count": len(keys),
            "has_keys": len(keys) > 0,
            # Don't expose actual key values
            "key_preview": [
                f"{k[:10]}...{k[-4:]}" if len(k) > 14 else "***" for k in keys
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/export")
async def export_setup(
    auth: bool = Depends(verify_auth_token),
):
    """
    Export the entire setup configuration as JSON including all API keys.
    """
    try:
        # Get all providers
        providers_dict = config_manager.get_providers()
        providers_list = list(providers_dict.values())

        # Get all models
        model_names = config_manager.get_models()
        models_list = []
        for name in model_names:
            try:
                config = config_manager.get_model_config(name)
                models_list.append(config)
            except Exception:
                pass

        # Get ALL environment variables including API keys
        env_vars = {}
        api_keys = {}

        for provider in providers_list:
            provider_name = provider.get("name", "")
            provider_upper = provider_name.upper()
            patterns = provider.get("api_keys", {}).get("env_var_patterns", [])
            provider_keys = []

            for pattern in patterns:
                if "{INDEX}" in pattern:
                    # Check for indexed keys (up to 20)
                    for i in range(1, 21):
                        env_var = pattern.replace("{PROVIDER}", provider_upper).replace(
                            "{INDEX}", str(i)
                        )
                        value = os.getenv(env_var)
                        if value:
                            env_vars[env_var] = value
                            provider_keys.append({"env_var": env_var, "value": value})
                else:
                    env_var = pattern.replace("{PROVIDER}", provider_upper)
                    value = os.getenv(env_var)
                    if value:
                        env_vars[env_var] = value
                        provider_keys.append({"env_var": env_var, "value": value})

            if provider_keys:
                api_keys[provider_name] = provider_keys

        # Also get other important env vars
        other_vars = [
            "CLIENT_API_KEY",
            "KEY_COOLDOWN_SECONDS",
            "MAX_KEY_RETRY_CYCLES",
            "LOG_LEVEL",
            "VERBOSE_HTTP_ERRORS",
            "CORS_ORIGINS",
            "RATE_LIMIT_REQUESTS_PER_MINUTE",
            "FAIL_ON_STARTUP_VALIDATION",
        ]
        for var in other_vars:
            value = os.getenv(var)
            if value:
                env_vars[var] = value

        export_data = {
            "version": "1.0.0",
            "exported_at": datetime.now().isoformat(),
            "metadata": {
                "total_providers": len(providers_list),
                "total_models": len(models_list),
                "total_api_keys": sum(len(keys) for keys in api_keys.values()),
                "note": "Full configuration export including all API keys. Store securely!",
            },
            "setup": {
                "providers": providers_list,
                "models": models_list,
                "environment": env_vars,
                "api_keys": api_keys,
            },
        }

        return export_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/import")
async def import_setup(
    data: Dict,
    merge: bool = Query(
        default=False, description="Merge with existing instead of replacing"
    ),
    auth: bool = Depends(verify_auth_token),
):
    """
    Import a setup configuration from JSON.

    Note: API keys must be added separately after import.
    """
    try:
        results = {
            "providers_imported": 0,
            "providers_skipped": 0,
            "models_imported": 0,
            "models_skipped": 0,
            "errors": [],
        }

        setup = data.get("setup", {})

        # Import providers
        providers = setup.get("providers", [])
        for provider in providers:
            try:
                provider_name = provider.get("name")
                exists = config_manager.provider_exists(provider_name)

                if exists and not merge:
                    results["providers_skipped"] += 1
                    continue

                config_manager.save_provider(provider, overwrite=merge)
                results["providers_imported"] += 1
            except Exception as e:
                results["errors"].append(
                    f"Provider '{provider.get('name', 'unknown')}': {str(e)}"
                )

        # Import models
        models = setup.get("models", [])
        for model in models:
            try:
                model_name = model.get("logical_name")
                exists = config_manager.model_config_exists(model_name)

                if exists and not merge:
                    results["models_skipped"] += 1
                    continue

                config_manager.save_model_config(model_name, model, overwrite=merge)
                results["models_imported"] += 1
            except Exception as e:
                results["errors"].append(
                    f"Model '{model.get('logical_name', 'unknown')}': {str(e)}"
                )

        # Get environment variables and API keys from import
        env_vars = setup.get("environment", {})
        api_keys = setup.get("api_keys", {})

        # Build complete .env file content with all values
        env_lines = []
        env_lines.append("# ============================================")
        env_lines.append("# Model-Proxy Configuration")
        env_lines.append("# Generated from import - all values included")
        env_lines.append("# ============================================")
        env_lines.append("")

        # Add client API key first
        if "CLIENT_API_KEY" in env_vars:
            env_lines.append("# Client Authentication (REQUIRED)")
            env_lines.append(f"CLIENT_API_KEY={env_vars['CLIENT_API_KEY']}")
            env_lines.append("")

        # Add all API keys by provider
        if api_keys:
            env_lines.append("# Provider API Keys")
            env_lines.append("")
            for provider_name, keys in api_keys.items():
                provider = next(
                    (p for p in providers if p.get("name") == provider_name), None
                )
                display_name = (
                    provider.get("display_name", provider_name)
                    if provider
                    else provider_name
                )
                env_lines.append(f"# {display_name}")
                for key_info in keys:
                    env_lines.append(f"{key_info['env_var']}={key_info['value']}")
                env_lines.append("")

        # Add other settings
        env_lines.append("# Settings")
        settings_vars = [
            "KEY_COOLDOWN_SECONDS",
            "MAX_KEY_RETRY_CYCLES",
            "LOG_LEVEL",
            "VERBOSE_HTTP_ERRORS",
            "CORS_ORIGINS",
            "RATE_LIMIT_REQUESTS_PER_MINUTE",
            "FAIL_ON_STARTUP_VALIDATION",
        ]
        for var in settings_vars:
            if var in env_vars and var != "CLIENT_API_KEY":
                env_lines.append(f"{var}={env_vars[var]}")

        env_content = "\n".join(env_lines)
        results["env_file"] = env_content
        results["env_filename"] = ".env"
        results["note"] = (
            f"Imported {results['providers_imported']} providers and {results['models_imported']} models with {sum(len(keys) for keys in api_keys.values())} API keys. .env file generated with all values."
        )

        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/templates")
async def get_provider_templates(auth: bool = Depends(verify_auth_token)):
    """Get available provider templates."""
    try:
        config_dir = get_writable_config_dir()
        templates_dir = config_dir / "templates"

        templates = {}
        if templates_dir.exists():
            for template_file in templates_dir.glob("*_template.json"):
                try:
                    with open(template_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        template_name = template_file.stem.replace("_template", "")
                        templates[template_name] = data
                except Exception:
                    pass

        return {
            "templates": templates,
            "count": len(templates),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/validate")
async def validate_configuration(auth: bool = Depends(verify_auth_token)):
    """Validate all configurations."""
    try:
        errors = []

        # Validate providers
        providers = config_manager.get_providers()
        for name, config in providers.items():
            try:
                config_manager._validate_provider_config(config)
            except Exception as e:
                errors.append(f"Provider '{name}': {str(e)}")

        # Validate models
        model_names = config_manager.get_models()
        for name in model_names:
            try:
                config = config_manager.get_model_config(name)
                config_manager._validate_model_config(config)
            except Exception as e:
                errors.append(f"Model '{name}': {str(e)}")

        if errors:
            return {
                "valid": False,
                "errors": errors,
            }
        else:
            return {
                "valid": True,
                "message": f"All {len(providers)} provider(s) and {len(model_names)} model(s) are valid",
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Static file serving
# This should be mounted in the main app, not in the router
