"""
Unit tests for model routing and fallback functionality.
"""
import pytest
import asyncio
import json
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from app.routing.models import ModelRoutingConfig, RouteConfig, ResolvedRoute, Attempt
from app.routing.config_loader import ModelConfigLoader
from app.routing.router import FallbackRouter, RoutingError


class TestModelRoutingConfig:
    """Test routing configuration models."""

    def test_valid_config(self):
        """Test parsing a valid routing config."""
        config_data = {
            "logical_name": "glm-4.6",
            "timeout_seconds": 60,
            "model_routings": [
                {
                    "id": "primary",
                    "wire_protocol": "openai",
                    "provider": "cerebras",
                    "model": "zai-glm-4.6",
                    "api_key_env": ["CEREBRAS_API_KEY"]
                }
            ],
            "fallback_model_routings": ["glm-4.5"]
        }

        config = ModelRoutingConfig(**config_data)
        assert config.logical_name == "glm-4.6"
        assert config.timeout_seconds == 60
        assert len(config.model_routings) == 1
        assert config.fallback_model_routings == ["glm-4.5"]

    def test_config_validation(self):
        """Test config validation."""
        # Missing required fields
        with pytest.raises(Exception):
            ModelRoutingConfig(logical_name="test")  # Missing model_routings

        # Invalid wire protocol
        with pytest.raises(Exception):
            RouteConfig(
                wire_protocol="invalid",  # Should be "openai" or "anthropic"
                provider="test",
                model="test",
                api_key_env=["TEST_KEY"]
            )


class TestConfigLoader:
    """Test configuration loading."""

    def test_load_config(self, tmp_path):
        """Test loading a config file."""
        config_dir = tmp_path / "models"
        config_dir.mkdir()

        config_file = config_dir / "test-model.json"
        config_data = {
            "logical_name": "test-model",
            "model_routings": [
                {
                    "wire_protocol": "openai",
                    "provider": "test",
                    "model": "test-model",
                    "api_key_env": ["TEST_API_KEY"]
                }
            ]
        }

        with open(config_file, 'w') as f:
            json.dump(config_data, f)

        loader = ModelConfigLoader(config_dir)
        config = loader.load_config("test-model")

        assert config.logical_name == "test-model"
        assert len(config.model_routings) == 1

    def test_missing_config(self, tmp_path):
        """Test loading non-existent config."""
        loader = ModelConfigLoader(tmp_path / "models")

        with pytest.raises(FileNotFoundError):
            loader.load_config("nonexistent-model")

    def test_invalid_json(self, tmp_path):
        """Test loading invalid JSON."""
        config_dir = tmp_path / "models"
        config_dir.mkdir()

        config_file = config_dir / "bad-model.json"
        with open(config_file, 'w') as f:
            f.write("invalid json {")

        loader = ModelConfigLoader(config_dir)

        with pytest.raises(ValueError):
            loader.load_config("bad-model")


class TestFallbackRouter:
    """Test the fallback router logic."""

    @pytest.fixture
    def mock_config_loader(self):
        """Mock config loader."""
        with patch('app.routing.router.config_loader') as mock_loader:
            # Mock glm-4.6 config
            glm46_config = ModelRoutingConfig(
                logical_name="glm-4.6",
                model_routings=[
                    RouteConfig(
                        wire_protocol="openai",
                        provider="cerebras",
                        model="zai-glm-4.6",
                        api_key_env=["CEREBRAS_KEY1", "CEREBRAS_KEY2"]
                    ),
                    RouteConfig(
                        wire_protocol="openai",
                        provider="nahcrof",
                        model="glm-4.6",
                        api_key_env=["NAHCROF_KEY1"]
                    )
                ],
                fallback_model_routings=["glm-4.5"]
            )

            # Mock glm-4.5 config
            glm45_config = ModelRoutingConfig(
                logical_name="glm-4.5",
                model_routings=[
                    RouteConfig(
                        wire_protocol="openai",
                        provider="nahcrof",
                        model="glm-4.5",
                        api_key_env=["NAHCROF_KEY1"]
                    )
                ]
            )

            def mock_load_config(logical_model):
                if logical_model == "glm-4.6":
                    return glm46_config
                elif logical_model == "glm-4.5":
                    return glm45_config
                else:
                    raise FileNotFoundError(f"No config for {logical_model}")

            mock_loader.load_config.side_effect = mock_load_config
            yield mock_loader

    def test_resolve_attempts_basic(self, mock_config_loader):
        """Test basic attempt resolution."""
        router = FallbackRouter()

        with patch.dict(os.environ, {"CEREBRAS_KEY1": "key1", "NAHCROF_KEY1": "key2"}):
            attempts = router.resolve_attempts("glm-4.6")

        assert len(attempts) == 3  # 2 from cerebras + 1 from nahcrof
        assert attempts[0].route.provider == "cerebras"
        assert attempts[0].route.api_key == "key1"
        assert attempts[1].route.provider == "cerebras"
        assert attempts[1].route.api_key == "key2"
        assert attempts[2].route.provider == "nahcrof"
        assert attempts[2].route.api_key == "key2"

    def test_resolve_attempts_with_fallbacks(self, mock_config_loader):
        """Test attempt resolution with logical model fallbacks."""
        router = FallbackRouter()

        with patch.dict(os.environ, {"CEREBRAS_KEY1": "key1", "NAHCROF_KEY1": "key2"}):
            attempts = router.resolve_attempts("glm-4.6")

        # Should include glm-4.5 fallback attempts
        fallback_attempts = [a for a in attempts if a.is_fallback_route]
        assert len(fallback_attempts) == 1
        assert fallback_attempts[0].route.source_logical_model == "glm-4.5"

    @pytest.mark.asyncio
    async def test_successful_call(self, mock_config_loader):
        """Test successful call with no fallback needed."""
        router = FallbackRouter()

        call_count = 0
        async def mock_exec(route: ResolvedRoute):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"success": True, "provider": route.provider}
            else:
                raise Exception("Should not be called")

        with patch.dict(os.environ, {"CEREBRAS_KEY1": "key1"}):
            result = await router.call_with_fallback("glm-4.6", mock_exec)

        assert result["success"] is True
        assert result["provider"] == "cerebras"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_fallback_on_failure(self, mock_config_loader):
        """Test fallback when first attempt fails."""
        router = FallbackRouter()

        call_count = 0
        async def mock_exec(route: ResolvedRoute):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("First attempt failed")
            elif call_count == 2:
                return {"success": True, "provider": route.provider}
            else:
                raise Exception("Should not be called")

        with patch.dict(os.environ, {"CEREBRAS_KEY1": "key1", "NAHCROF_KEY1": "key2"}):
            result = await router.call_with_fallback("glm-4.6", mock_exec)

        assert result["success"] is True
        assert result["provider"] == "nahcrof"  # Second provider
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_all_routes_fail(self, mock_config_loader):
        """Test when all routes fail."""
        router = FallbackRouter()

        async def mock_exec(route: ResolvedRoute):
            raise Exception("All routes fail")

        with patch.dict(os.environ, {"CEREBRAS_KEY1": "key1"}):
            with pytest.raises(RoutingError) as exc_info:
                await router.call_with_fallback("glm-4.6", mock_exec)

        error = exc_info.value
        assert "glm-4.6" in error.message
        assert len(error.attempted_routes) > 0
        assert len(error.errors) > 0

    def test_cycle_detection(self, mock_config_loader):
        """Test detection of cycles in fallback configs."""
        # Mock a cycle: glm-4.6 -> glm-4.5 -> glm-4.6
        glm46_config = ModelRoutingConfig(
            logical_name="glm-4.6",
            model_routings=[],
            fallback_model_routings=["glm-4.5"]
        )

        glm45_config = ModelRoutingConfig(
            logical_name="glm-4.5",
            model_routings=[],
            fallback_model_routings=["glm-4.6"]  # Creates cycle
        )

        def mock_load_config(logical_model):
            if logical_model == "glm-4.6":
                return glm46_config
            elif logical_model == "glm-4.5":
                return glm45_config

        mock_config_loader.load_config.side_effect = mock_load_config

        router = FallbackRouter()
        attempts = router.resolve_attempts("glm-4.6")

        # Should not include any fallback attempts due to cycle
        assert len(attempts) == 0

    def test_fallback_worthy_errors(self):
        """Test classification of fallback-worthy errors."""
        router = FallbackRouter()

        # HTTP errors should be fallback-worthy
        assert router._is_fallback_worthy_error(Exception("HTTP 500 Internal Server Error"))
        assert router._is_fallback_worthy_error(Exception("HTTP 429 Too Many Requests"))

        # Network errors should be fallback-worthy
        assert router._is_fallback_worthy_error(Exception("Connection refused"))

        # Timeout errors should be fallback-worthy
        assert router._is_fallback_worthy_error(asyncio.TimeoutError())

        # Auth errors should NOT be fallback-worthy (permanent failures)
        assert not router._is_fallback_worthy_error(Exception("HTTP 401 Unauthorized"))
        assert not router._is_fallback_worthy_error(Exception("HTTP 403 Forbidden"))
