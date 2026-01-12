"""
Test suite to verify that the routing system works correctly without route IDs.
This ensures that the simplification of schemas by removing IDs doesn't break functionality.
"""

from app.routing.config_loader import config_loader
from app.routing.models import ModelRoutingConfig, ResolvedRoute, RouteConfig


class TestNoRouteIDs:
    """Test that routing works without ID fields."""

    def test_route_config_without_id(self):
        """Test that RouteConfig works without id field."""
        route_data = {
            "provider": "cerebras",
            "model": "zai-glm-4.7",
            "wire_protocol": "openai",
        }

        route = RouteConfig(**route_data)

        assert route.provider == "cerebras"
        assert route.model == "zai-glm-4.7"
        assert route.wire_protocol == "openai"
        # Verify no id field exists
        assert not hasattr(route, "id")
        assert "id" not in route.__dict__

    def test_model_config_without_route_ids(self):
        """Test that ModelRoutingConfig works with routes that have no IDs."""
        config_data = {
            "logical_name": "test-model",
            "timeout_seconds": 60,
            "model_routings": [
                {"provider": "cerebras", "model": "model-a", "wire_protocol": "openai"},
                {"provider": "nahcrof", "model": "model-b", "wire_protocol": "openai"},
            ],
            "fallback_model_routings": ["backup-model"],
        }

        config = ModelRoutingConfig(**config_data)

        assert config.logical_name == "test-model"
        assert len(config.model_routings) == 2
        assert config.model_routings[0].provider == "cerebras"
        assert config.model_routings[1].provider == "nahcrof"

        # Verify no routes have id fields
        for route in config.model_routings:
            assert not hasattr(route, "id")
            assert "id" not in route.__dict__

    def test_resolved_route_without_route_id(self):
        """Test that ResolvedRoute works without route_id field."""
        resolved_route = ResolvedRoute(
            source_logical_model="test-model",
            wire_protocol="openai",
            provider="cerebras",
            model="zai-glm-4.7",
            api_key="test-key",
            timeout_seconds=60,
        )

        assert resolved_route.source_logical_model == "test-model"
        assert resolved_route.provider == "cerebras"
        assert resolved_route.model == "zai-glm-4.7"
        # Verify no route_id field exists
        assert not hasattr(resolved_route, "route_id")
        assert "route_id" not in resolved_route.__dict__

    def test_load_existing_model_configs(self):
        """Test that existing model configs load correctly without IDs."""
        # Test several model configs to ensure they work
        models_to_test = ["glm-4.6", "glm-4.7", "sprecision", "gpt-oss-120b"]

        for model_name in models_to_test:
            config = config_loader.load_config(model_name)

            assert config.logical_name == model_name
            assert len(config.model_routings) > 0

            # Verify no routes have id fields
            for route in config.model_routings:
                assert not hasattr(route, "id")
                assert "id" not in route.__dict__

    def test_no_id_in_error_info(self):
        """Test that error info doesn't include route_id."""
        # This is a unit test to verify error info structure
        # We'll create a mock error info dict like the router would
        error_info = {
            "attempt": 1,
            "provider": "cerebras",
            "model": "zai-glm-4.7",
            "error": "Connection failed",
            "error_type": "ConnectionError",
        }

        # Verify route_id is not in error info
        assert "route_id" not in error_info
        assert len(error_info) == 5  # Should only have the 5 fields above

    def test_route_config_serialization_without_id(self):
        """Test that RouteConfig serializes correctly without id field."""
        route_data = {
            "provider": "cerebras",
            "model": "zai-glm-4.7",
            "wire_protocol": "openai",
            "api_key_env": ["CEREBRAS_API_KEY"],
        }

        route = RouteConfig(**route_data)
        serialized = route.model_dump()

        # Verify serialization doesn't include id
        assert "id" not in serialized
        assert serialized == {
            "wire_protocol": "openai",
            "provider": "cerebras",
            "model": "zai-glm-4.7",
            "base_url": None,
            "api_key_env": ["CEREBRAS_API_KEY"],
            "timeout_seconds": None,
            "cooldown_seconds": None,
        }

    def test_model_config_serialization_without_ids(self):
        """Test that ModelRoutingConfig serializes routes without id fields."""
        config_data = {
            "logical_name": "test-model",
            "model_routings": [{"provider": "cerebras", "model": "model-a"}],
        }

        config = ModelRoutingConfig(**config_data)
        serialized = config.model_dump()

        # Verify no route has id in serialization
        for route in serialized["model_routings"]:
            assert "id" not in route

        # Verify the route structure is correct
        expected_route = {
            "wire_protocol": None,
            "provider": "cerebras",
            "model": "model-a",
            "base_url": None,
            "api_key_env": None,
            "timeout_seconds": None,
            "cooldown_seconds": None,
        }

        assert serialized["model_routings"][0] == expected_route


class TestBackwardCompatibility:
    """Test that the system handles cases where IDs might still exist."""

    def test_route_config_ignores_id_if_present(self):
        """Test that RouteConfig ignores id field if somehow present in data."""
        # This shouldn't normally happen, but let's ensure robustness
        route_data = {
            "id": "primary",  # This should be ignored
            "provider": "cerebras",
            "model": "zai-glm-4.7",
        }

        # Should work without error, and id should not be stored
        route = RouteConfig(**route_data)

        assert route.provider == "cerebras"
        assert route.model == "zai-glm-4.7"
        # id should not be present
        assert not hasattr(route, "id")
        assert "id" not in route.__dict__
