"""
Configuration Export/Import module for Model-Proxy CLI.

Provides commands to export the entire setup configuration as JSON
and import it on another device.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import typer

from app.cli.config_manager import ConfigManager
from app.cli.interactive import (
    display_error,
    display_info,
    display_success,
    display_warning,
)


def export_setup(
    output: Optional[str] = None,
    include_keys: bool = False,
    encrypt: bool = False,
) -> Dict:
    """
    Export the entire Model-Proxy setup configuration.

    Args:
        output: Output file path (optional, prints to stdout if not provided)
        include_keys: Whether to include API keys (not recommended for security)
        encrypt: Whether to encrypt the export (not implemented yet)

    Returns:
        Dictionary containing the export data
    """
    config_manager = ConfigManager()

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

    # Get environment variables (excluding API keys by default)
    env_vars = {}
    safe_vars = [
        "CLIENT_API_KEY",
        "KEY_COOLDOWN_SECONDS",
        "MAX_KEY_RETRY_CYCLES",
        "LOG_LEVEL",
        "VERBOSE_HTTP_ERRORS",
        "CORS_ORIGINS",
        "RATE_LIMIT_REQUESTS_PER_MINUTE",
        "FAIL_ON_STARTUP_VALIDATION",
    ]

    for var in safe_vars:
        value = os.getenv(var)
        if value:
            # For CLIENT_API_KEY, only indicate it exists
            if var == "CLIENT_API_KEY":
                env_vars[var] = "[SET]"
            else:
                env_vars[var] = value

    # Build export data
    export_data = {
        "version": "1.0.0",
        "exported_at": datetime.now().isoformat(),
        "metadata": {
            "total_providers": len(providers_list),
            "total_models": len(models_list),
            "note": "API keys are not included for security. Re-enter keys on target device.",
        },
        "setup": {
            "providers": providers_list,
            "models": models_list,
            "environment": env_vars,
        },
    }

    # Include keys only if explicitly requested (not recommended)
    if include_keys:
        display_warning("Including API keys in export is not recommended for security!")
        if not typer.confirm("Are you sure you want to include API keys?"):
            include_keys = False

    if include_keys:
        api_keys = {}
        for provider in providers_list:
            provider_name = provider["name"]
            patterns = provider.get("api_keys", {}).get("env_var_patterns", [])
            keys = []

            for pattern in patterns:
                if "{INDEX}" in pattern:
                    for i in range(1, 10):  # Check up to 9 keys
                        env_var = pattern.format(
                            PROVIDER=provider_name.upper(), INDEX=i
                        )
                        value = os.getenv(env_var)
                        if value:
                            keys.append({"env_var": env_var, "value": value})
                else:
                    env_var = pattern.format(PROVIDER=provider_name.upper())
                    value = os.getenv(env_var)
                    if value:
                        keys.append({"env_var": env_var, "value": value})

            if keys:
                api_keys[provider_name] = keys

        export_data["setup"]["api_keys"] = api_keys
        export_data["metadata"]["note"] = "API keys included. Store this file securely!"

    # Output
    json_output = json.dumps(export_data, indent=2)

    if output:
        output_path = Path(output)
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(json_output)
            display_success(f"Setup exported to: {output_path.absolute()}")
        except Exception as e:
            display_error(f"Failed to write export file: {e}")
            raise typer.Exit(1)
    else:
        # Print to stdout
        print(json_output)

    return export_data


def import_setup(
    input_file: str,
    merge: bool = False,
    skip_keys: bool = True,
) -> Dict:
    """
    Import a Model-Proxy setup configuration.

    Args:
        input_file: Path to the JSON file to import
        merge: Whether to merge with existing config or replace
        skip_keys: Whether to skip importing API keys (recommended)

    Returns:
        Dictionary containing import results
    """
    config_manager = ConfigManager()

    # Read input file
    input_path = Path(input_file)
    if not input_path.exists():
        display_error(f"Import file not found: {input_file}")
        raise typer.Exit(1)

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            import_data = json.load(f)
    except json.JSONDecodeError as e:
        display_error(f"Invalid JSON in import file: {e}")
        raise typer.Exit(1)
    except Exception as e:
        display_error(f"Failed to read import file: {e}")
        raise typer.Exit(1)

    # Validate structure
    if "setup" not in import_data:
        display_error("Invalid import file: missing 'setup' section")
        raise typer.Exit(1)

    setup = import_data["setup"]
    results = {
        "providers_imported": 0,
        "providers_skipped": 0,
        "models_imported": 0,
        "models_skipped": 0,
        "environment_vars_set": 0,
        "errors": [],
    }

    # Import providers
    providers = setup.get("providers", [])
    display_info(f"Importing {len(providers)} provider(s)...")

    for provider in providers:
        try:
            provider_name = provider.get("name")
            exists = config_manager.provider_exists(provider_name)

            if exists and not merge:
                display_warning(
                    f"Provider '{provider_name}' already exists, skipping (use --merge to overwrite)"
                )
                results["providers_skipped"] += 1
                continue

            config_manager.save_provider(provider, overwrite=merge)
            display_success(f"Provider '{provider_name}' imported")
            results["providers_imported"] += 1
        except Exception as e:
            error_msg = (
                f"Failed to import provider '{provider.get('name', 'unknown')}': {e}"
            )
            display_error(error_msg)
            results["errors"].append(error_msg)

    # Import models
    models = setup.get("models", [])
    display_info(f"\nImporting {len(models)} model(s)...")

    for model in models:
        try:
            model_name = model.get("logical_name")
            exists = config_manager.model_config_exists(model_name)

            if exists and not merge:
                display_warning(
                    f"Model '{model_name}' already exists, skipping (use --merge to overwrite)"
                )
                results["models_skipped"] += 1
                continue

            config_manager.save_model_config(model_name, model, overwrite=merge)
            display_success(f"Model '{model_name}' imported")
            results["models_imported"] += 1
        except Exception as e:
            error_msg = (
                f"Failed to import model '{model.get('logical_name', 'unknown')}': {e}"
            )
            display_error(error_msg)
            results["errors"].append(error_msg)

    # Handle environment variables
    environment = setup.get("environment", {})
    if environment:
        display_info("\nEnvironment variables from import:")
        for var, value in environment.items():
            if value == "[SET]":
                display_info(f"  • {var}: [already set on source device]")
            else:
                display_info(f"  • {var}: {value}")

        display_warning(
            "\nEnvironment variables need to be set manually in your .env file"
        )

    # Handle API keys (if present and not skipped)
    api_keys = setup.get("api_keys", {})
    if api_keys and not skip_keys:
        display_warning("\nAPI keys found in import file")
        display_info("API keys need to be added manually via environment variables")

    # Print summary
    print("\n" + "=" * 60)
    display_success("Import complete!")
    print(f"  • Providers imported: {results['providers_imported']}")
    print(f"  • Providers skipped: {results['providers_skipped']}")
    print(f"  • Models imported: {results['models_imported']}")
    print(f"  • Models skipped: {results['models_skipped']}")

    if results["errors"]:
        display_warning(f"\n{len(results['errors'])} error(s) occurred during import")

    display_info("\nNext steps:")
    display_info("1. Add API keys to your .env file for imported providers")
    display_info("2. Run 'model-proxy doctor' to validate the setup")
    display_info("3. Start the server with 'model-proxy start'")

    return results
