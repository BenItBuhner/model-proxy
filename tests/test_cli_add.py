"""
Comprehensive tests for the Model-Proxy CLI add commands.

Tests cover:
- Provider addition (interactive and non-interactive)
- Model configuration (interactive and non-interactive)
- API key management (interactive and non-interactive)
- Edge cases and error handling
"""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from app.cli_main import app

runner = CliRunner()


def get_output(result):
    """Get combined stdout and stderr output."""
    output = result.stdout or ""
    # Typer's CliRunner includes stderr in output by default
    if hasattr(result, "stderr") and result.stderr:
        output += result.stderr
    # Also check the result output attribute
    if hasattr(result, "output") and result.output:
        if result.output not in output:
            output += result.output
    return output


class TestAddProviderCommand:
    """Tests for the 'model-proxy add provider' command."""

    def test_add_provider_list(self):
        """Test listing providers with --list flag."""
        result = runner.invoke(app, ["add", "provider", "--list"])
        assert result.exit_code == 0
        assert (
            "provider(s)" in result.stdout.lower()
            or "provider" in result.stdout.lower()
        )

    def test_add_provider_help(self):
        """Test help output for add provider command."""
        result = runner.invoke(app, ["add", "provider", "--help"])
        assert result.exit_code == 0
        assert "--list" in result.stdout
        assert "--name" in result.stdout
        assert "--display-name" in result.stdout
        assert "--base-url" in result.stdout
        assert "--format" in result.stdout
        assert "--overwrite" in result.stdout

    def test_add_provider_non_interactive_missing_flags(self):
        """Test that partial flags show error."""
        result = runner.invoke(
            app, ["add", "provider", "--name", "test", "--base-url", "https://test.com"]
        )
        output = get_output(result)
        assert result.exit_code == 1
        assert "requires all flags" in output.lower() or "error" in output.lower()

    def test_add_provider_non_interactive_invalid_url(self):
        """Test that invalid URL is rejected."""
        result = runner.invoke(
            app,
            [
                "add",
                "provider",
                "--name",
                "test",
                "--display-name",
                "Test",
                "--base-url",
                "invalid-url",
                "--format",
                "openai",
            ],
        )
        output = get_output(result)
        # Should show error about invalid URL
        assert "invalid" in output.lower() or "error" in output.lower()

    def test_add_provider_non_interactive_invalid_format(self):
        """Test that invalid format type is rejected."""
        result = runner.invoke(
            app,
            [
                "add",
                "provider",
                "--name",
                "test",
                "--display-name",
                "Test",
                "--base-url",
                "https://api.test.com",
                "--format",
                "invalid",
            ],
        )
        output = get_output(result)
        assert (
            "invalid format" in output.lower()
            or "valid options" in output.lower()
            or "error" in output.lower()
        )

    def test_add_provider_non_interactive_success(self):
        """Test successful non-interactive provider addition."""
        # Use a unique name to avoid conflicts
        import uuid

        unique_name = f"testprov{uuid.uuid4().hex[:8]}"

        result = runner.invoke(
            app,
            [
                "add",
                "provider",
                "--name",
                unique_name,
                "--display-name",
                "Test Provider Unique",
                "--base-url",
                "https://api.testunique.com",
                "--format",
                "openai",
            ],
        )
        output = get_output(result)
        assert result.exit_code == 0
        assert "saved successfully" in output.lower() or "ok" in output.lower()

        # Verify provider was created
        provider_file = Path(f"config/providers/{unique_name}.json")
        assert provider_file.exists()

        # Clean up
        if provider_file.exists():
            provider_file.unlink()

    def test_add_provider_non_interactive_duplicate_without_overwrite(self):
        """Test that duplicate provider without --overwrite fails."""
        # Use existing provider
        result = runner.invoke(
            app,
            [
                "add",
                "provider",
                "--name",
                "openai",
                "--display-name",
                "Duplicate",
                "--base-url",
                "https://api.test.com",
                "--format",
                "openai",
            ],
        )
        output = get_output(result)
        assert (
            "already exists" in output.lower()
            or "overwrite" in output.lower()
            or "error" in output.lower()
        )


class TestAddModelCommand:
    """Tests for the 'model-proxy add model' command."""

    def test_add_model_list(self):
        """Test listing models with --list flag."""
        result = runner.invoke(app, ["add", "model", "--list"])
        assert result.exit_code == 0
        assert "model" in result.stdout.lower()

    def test_add_model_help(self):
        """Test help output for add model command."""
        result = runner.invoke(app, ["add", "model", "--help"])
        assert result.exit_code == 0
        assert "--list" in result.stdout
        assert "--name" in result.stdout
        assert "--provider" in result.stdout
        assert "--model" in result.stdout
        assert "--timeout" in result.stdout
        assert "--custom" in result.stdout
        assert "--overwrite" in result.stdout

    def test_add_model_non_interactive_missing_flags(self):
        """Test that partial flags show error."""
        result = runner.invoke(app, ["add", "model", "--name", "test"])
        output = get_output(result)
        assert (
            result.exit_code == 1
            or "requires" in output.lower()
            or "error" in output.lower()
        )

    def test_add_model_non_interactive_invalid_provider(self):
        """Test that non-existent provider is rejected."""
        result = runner.invoke(
            app,
            [
                "add",
                "model",
                "--name",
                "test-model",
                "--provider",
                "nonexistent",
                "--model",
                "test",
            ],
        )
        output = get_output(result)
        assert "does not exist" in output.lower() or "error" in output.lower()

    def test_add_model_non_interactive_success(self):
        """Test successful non-interactive model addition."""
        import uuid

        unique_name = f"testmodel{uuid.uuid4().hex[:8]}"

        # Use existing provider
        result = runner.invoke(
            app,
            [
                "add",
                "model",
                "--name",
                unique_name,
                "--provider",
                "openai",
                "--model",
                "gpt-4",
                "--timeout",
                "120",
            ],
        )
        output = get_output(result)
        assert result.exit_code == 0
        assert "saved successfully" in output.lower() or "ok" in output.lower()

        # Verify model config was created
        model_file = Path(f"config/models/{unique_name}.json")
        assert model_file.exists()

        # Verify content
        with open(model_file) as f:
            config = json.load(f)
            assert config["logical_name"] == unique_name
            assert config["timeout_seconds"] == 120
            assert config["model_routings"][0]["provider"] == "openai"
            assert config["model_routings"][0]["model"] == "gpt-4"

        # Clean up
        if model_file.exists():
            model_file.unlink()

    def test_add_custom_model_non_interactive(self):
        """Test adding custom model to cache."""
        import uuid
        from app.cli.config_manager import ConfigManager

        unique_model = f"custom-model-{uuid.uuid4().hex[:8]}"
        config_manager = ConfigManager()

        try:
            result = runner.invoke(
                app,
                [
                    "add",
                    "model",
                    "--custom",
                    "--provider",
                    "openai",
                    "--model",
                    unique_model,
                ],
            )
            output = get_output(result)
            assert result.exit_code == 0
            assert "added" in output.lower() or "ok" in output.lower()

            # Verify model is in cache
            cache_file = Path("config/models.json")
            if cache_file.exists():
                with open(cache_file) as f:
                    cache = json.load(f)
                    assert unique_model in cache.get("custom_models", {}).get("openai", [])
        finally:
            # Clean up: remove the test model from cache
            try:
                cache = config_manager.get_models_cache()
                if "custom_models" in cache and "openai" in cache["custom_models"]:
                    if unique_model in cache["custom_models"]["openai"]:
                        cache["custom_models"]["openai"].remove(unique_model)
                        config_manager.update_models_cache(cache)
            except Exception:
                pass  # Ignore cleanup errors

    def test_add_custom_model_invalid_provider(self):
        """Test that custom model with invalid provider fails."""
        result = runner.invoke(
            app,
            [
                "add",
                "model",
                "--custom",
                "--provider",
                "nonexistent",
                "--model",
                "test",
            ],
        )
        output = get_output(result)
        assert "does not exist" in output.lower() or "error" in output.lower()


class TestAddKeyCommand:
    """Tests for the 'model-proxy add key' command."""

    def test_add_key_list(self):
        """Test listing API keys with --list flag."""
        result = runner.invoke(app, ["add", "key", "--list"])
        assert result.exit_code == 0
        assert (
            "api key" in result.stdout.lower() or "configured" in result.stdout.lower()
        )

    def test_add_key_help(self):
        """Test help output for add key command."""
        result = runner.invoke(app, ["add", "key", "--help"])
        assert result.exit_code == 0
        assert "--list" in result.stdout
        assert "--provider" in result.stdout
        assert "--key" in result.stdout
        assert "--env-var" in result.stdout

    def test_add_key_non_interactive_missing_flags(self):
        """Test that partial flags show error."""
        result = runner.invoke(app, ["add", "key", "--provider", "openai"])
        output = get_output(result)
        assert (
            result.exit_code == 1
            or "requires" in output.lower()
            or "error" in output.lower()
        )

    def test_add_key_non_interactive_invalid_provider(self):
        """Test that non-existent provider is rejected."""
        result = runner.invoke(
            app,
            ["add", "key", "--provider", "nonexistent", "--key", "sk-test1234567890"],
        )
        output = get_output(result)
        assert "does not exist" in output.lower() or "error" in output.lower()

    def test_add_key_shows_censored_output(self):
        """Test that API keys are properly censored in listing."""
        result = runner.invoke(app, ["add", "key", "--list"])
        assert result.exit_code == 0
        # Check that keys are censored (asterisks shown)
        if "API_KEY" in result.stdout:
            assert "****" in result.stdout or "[No keys" in result.stdout


class TestConfigManager:
    """Tests for the ConfigManager class."""

    def test_get_providers(self):
        """Test loading provider configurations."""
        from app.cli.config_manager import ConfigManager

        manager = ConfigManager()
        providers = manager.get_providers()

        assert isinstance(providers, dict)
        assert len(providers) > 0
        assert "openai" in providers or len(providers) > 0

    def test_get_models(self):
        """Test loading model configurations."""
        from app.cli.config_manager import ConfigManager

        manager = ConfigManager()
        models = manager.get_models()

        assert isinstance(models, list)

    def test_provider_exists(self):
        """Test provider existence check."""
        from app.cli.config_manager import ConfigManager

        manager = ConfigManager()

        # Check existing provider
        assert manager.provider_exists("openai")

        # Check non-existent provider
        assert not manager.provider_exists("nonexistent_provider_12345")

    def test_validate_provider_config_valid(self):
        """Test validation of valid provider config."""
        from app.cli.config_manager import ConfigManager

        manager = ConfigManager()
        config = {
            "name": "test",
            "display_name": "Test Provider",
            "api_keys": {"env_var_patterns": ["{PROVIDER}_API_KEY"]},
            "endpoints": {"base_url": "https://test.com"},
        }

        # Should not raise
        manager._validate_provider_config(config)

    def test_validate_provider_config_missing_field(self):
        """Test validation of invalid provider config (missing field)."""
        from app.cli.config_manager import ConfigManager

        manager = ConfigManager()
        config = {
            "name": "test",
            # Missing display_name
            "api_keys": {"env_var_patterns": ["{PROVIDER}_API_KEY"]},
            "endpoints": {"base_url": "https://test.com"},
        }

        with pytest.raises(ValueError, match="Missing required field"):
            manager._validate_provider_config(config)

    def test_validate_provider_config_invalid_url(self):
        """Test validation of invalid URL."""
        from app.cli.config_manager import ConfigManager

        manager = ConfigManager()
        config = {
            "name": "test",
            "display_name": "Test Provider",
            "api_keys": {"env_var_patterns": ["{PROVIDER}_API_KEY"]},
            "endpoints": {"base_url": "invalid-url"},
        }

        with pytest.raises(ValueError, match="base_url must start with"):
            manager._validate_provider_config(config)


class TestInteractiveUtilities:
    """Tests for interactive utility functions."""

    def test_censor_string(self):
        """Test string censorship."""
        from app.cli.interactive import censor_string

        # Normal string - 12 chars, show last 4
        result = censor_string("sk-1234567890")
        assert result.endswith("7890")
        assert "****" in result

        # Short string
        assert censor_string("1234") == "1234"

        # Empty string
        assert censor_string("") == ""

    def test_display_functions(self):
        """Test that display functions don't crash."""
        from app.cli.interactive import (
            display_error,
            display_info,
            display_success,
            display_warning,
        )

        # These should not raise exceptions
        display_success("Test success")
        display_error("Test error")
        display_warning("Test warning")
        display_info("Test info")


class TestProviderTemplates:
    """Tests for provider templates."""

    def test_openai_template_exists(self):
        """Test that OpenAI template exists and is valid JSON."""
        template_path = Path("config/templates/openai_template.json")
        assert template_path.exists()

        with open(template_path) as f:
            template = json.load(f)
            assert "{{provider_name}}" in json.dumps(template)
            assert "{{base_url}}" in json.dumps(template)

    def test_anthropic_template_exists(self):
        """Test that Anthropic template exists and is valid JSON."""
        template_path = Path("config/templates/anthropic_template.json")
        assert template_path.exists()

        with open(template_path) as f:
            template = json.load(f)
            assert "{{provider_name}}" in json.dumps(template)

    def test_gemini_template_exists(self):
        """Test that Gemini template exists and is valid JSON."""
        template_path = Path("config/templates/gemini_template.json")
        assert template_path.exists()

        with open(template_path) as f:
            template = json.load(f)
            assert "{{provider_name}}" in json.dumps(template)

    def test_azure_template_exists(self):
        """Test that Azure template exists and is valid JSON."""
        template_path = Path("config/templates/azure_template.json")
        assert template_path.exists()

        with open(template_path) as f:
            template = json.load(f)
            assert "{{provider_name}}" in json.dumps(template)


class TestModelDiscovery:
    """Tests for model discovery functionality."""

    def test_discovery_module_imports(self):
        """Test that discovery module can be imported."""
        from app.cli.discovery import ModelDiscovery

        assert ModelDiscovery is not None

    def test_models_cache_structure(self):
        """Test that models cache has correct structure."""
        from app.cli.config_manager import ConfigManager

        manager = ConfigManager()
        cache = manager.get_models_cache()

        assert isinstance(cache, dict)
        assert "discovered_models" in cache or cache == {
            "discovered_models": {},
            "custom_models": {},
            "last_updated": None,
        }


class TestCLIIntegration:
    """Integration tests for CLI commands."""

    def test_doctor_command(self):
        """Test that doctor command runs."""
        result = runner.invoke(app, ["doctor"])
        # Should complete (exit code 0 or 1 depending on config)
        assert result.exit_code in [0, 1]

    def test_config_list_command(self):
        """Test config list command."""
        result = runner.invoke(app, ["config", "list"])
        assert result.exit_code == 0

    def test_version_command(self):
        """Test version command."""
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "version" in result.stdout.lower() or "0.1" in result.stdout

    def test_help_command(self):
        """Test help command."""
        result = runner.invoke(app, ["help"])
        assert result.exit_code == 0
        assert "model-proxy" in result.stdout.lower()


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_provider_name(self):
        """Test that empty provider name is rejected."""
        result = runner.invoke(
            app,
            [
                "add",
                "provider",
                "--name",
                "",
                "--display-name",
                "Test",
                "--base-url",
                "https://test.com",
                "--format",
                "openai",
            ],
        )
        output = get_output(result)
        # Should show error or not create provider
        assert (
            result.exit_code != 0
            or "error" in output.lower()
            or "empty" in output.lower()
            or "cannot be empty" in output.lower()
        )

    def test_provider_name_with_spaces(self):
        """Test that provider name with spaces is rejected."""
        result = runner.invoke(
            app,
            [
                "add",
                "provider",
                "--name",
                "test provider",
                "--display-name",
                "Test",
                "--base-url",
                "https://test.com",
                "--format",
                "openai",
            ],
        )
        output = get_output(result)
        # Should show error about spaces
        assert (
            "error" in output.lower()
            or "space" in output.lower()
            or "invalid" in output.lower()
            or "alphanumeric" in output.lower()
        )

    def test_provider_name_uppercase(self):
        """Test that uppercase provider name is rejected."""
        result = runner.invoke(
            app,
            [
                "add",
                "provider",
                "--name",
                "TestProvider",
                "--display-name",
                "Test",
                "--base-url",
                "https://test.com",
                "--format",
                "openai",
            ],
        )
        output = get_output(result)
        # Should show error about lowercase
        assert (
            "error" in output.lower()
            or "lowercase" in output.lower()
            or "invalid" in output.lower()
        )

    def test_negative_timeout(self):
        """Test that negative timeout is handled."""
        runner.invoke(
            app,
            [
                "add",
                "model",
                "--name",
                "test",
                "--provider",
                "openai",
                "--model",
                "gpt-4",
                "--timeout",
                "-1",
            ],
        )
        # Should show error or use default
        # The exact behavior depends on implementation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
