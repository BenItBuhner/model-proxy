#!/usr/bin/env python3
"""
Fallback Routing Tester
Command-line tool to test multi-level model fallback routing.
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
import argparse
import time

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.routing.router import call_with_fallback
from app.routing.models import ResolvedRoute, RoutingError
from app.routing.config_loader import config_loader


class MockProvider:
    """Mock provider that can simulate different failure modes."""

    def __init__(self, name: str, fail_mode: Optional[str] = None, delay: float = 0.0):
        self.name = name
        self.fail_mode = fail_mode
        self.delay = delay
        self.call_count = 0

    async def call(self, **kwargs) -> Dict[str, Any]:
        """Mock call that may fail based on fail_mode."""
        self.call_count += 1
        if self.delay > 0:
            await asyncio.sleep(self.delay)

        if self.fail_mode == "timeout":
            await asyncio.sleep(30)  # Force timeout
        elif self.fail_mode == "http_500":
            raise Exception("HTTP 500 Internal Server Error")
        elif self.fail_mode == "http_429":
            raise Exception("HTTP 429 Too Many Requests")
        elif self.fail_mode == "connection":
            raise Exception("Connection refused")
        elif self.fail_mode == "auth":
            raise Exception("HTTP 401 Unauthorized")

        # Success case
        return {
            "id": f"mock-{self.name}-{self.call_count}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": kwargs.get("model", "mock-model"),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": f"Mock response from {self.name} (attempt {self.call_count})",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }


class FallbackTester:
    """Tester for fallback routing scenarios."""

    def __init__(self):
        self.mock_providers: Dict[str, MockProvider] = {}
        self.call_log: List[Dict[str, Any]] = []

    def setup_mock_scenario(self, scenario: str):
        """Set up mock providers for different test scenarios."""
        self.mock_providers.clear()

        if scenario == "basic_success":
            # All routes work
            self.mock_providers["cerebras"] = MockProvider("cerebras")
            self.mock_providers["nahcrof"] = MockProvider("nahcrof")

        elif scenario == "api_key_fallback":
            # First API key fails, second succeeds
            self.mock_providers["cerebras"] = MockProvider("cerebras", fail_mode="auth")
            self.mock_providers["nahcrof"] = MockProvider("nahcrof")

        elif scenario == "provider_fallback":
            # Primary provider fails, secondary succeeds
            self.mock_providers["cerebras"] = MockProvider(
                "cerebras", fail_mode="http_500"
            )
            self.mock_providers["nahcrof"] = MockProvider("nahcrof")

        elif scenario == "logical_model_fallback":
            # All primary routes fail, fallback to another logical model
            self.mock_providers["cerebras"] = MockProvider(
                "cerebras", fail_mode="http_500"
            )
            self.mock_providers["nahcrof"] = MockProvider(
                "nahcrof", fail_mode="http_500"
            )
            # glm-4.5 routes work
            self.mock_providers["cerebras_45"] = MockProvider("cerebras-glm45")
            self.mock_providers["nahcrof_45"] = MockProvider("nahcrof-glm45")

        elif scenario == "timeout_fallback":
            # First route times out, second succeeds
            self.mock_providers["cerebras"] = MockProvider(
                "cerebras", fail_mode="timeout", delay=0.1
            )
            self.mock_providers["nahcrof"] = MockProvider("nahcrof")

        elif scenario == "all_fail":
            # All routes fail
            self.mock_providers["cerebras"] = MockProvider(
                "cerebras", fail_mode="http_500"
            )
            self.mock_providers["nahcrof"] = MockProvider(
                "nahcrof", fail_mode="connection"
            )

    async def execute_route(self, resolved_route: ResolvedRoute) -> Dict[str, Any]:
        """Execute a route using mock providers."""
        provider_key = f"{resolved_route.provider}_{resolved_route.model.replace('.', '').replace('-', '')}"

        # Special handling for fallback models
        if resolved_route.source_logical_model == "glm-4.5":
            provider_key = f"{resolved_route.provider}_45"

        provider = self.mock_providers.get(provider_key)
        if not provider:
            # Fallback to base provider name
            provider = self.mock_providers.get(resolved_route.provider)

        if not provider:
            raise Exception(
                f"No mock provider configured for {resolved_route.provider}"
            )

        # Log the attempt
        self.call_log.append(
            {
                "timestamp": time.time(),
                "logical_model": resolved_route.source_logical_model,
                "provider": resolved_route.provider,
                "model": resolved_route.model,
                "wire_protocol": resolved_route.wire_protocol,
                "api_key": resolved_route.api_key[:8] + "..."
                if resolved_route.api_key
                else None,
            }
        )

        print(
            f"[Attempt] {resolved_route.source_logical_model} -> {resolved_route.provider}/{resolved_route.model}"
        )
        return await provider.call(model=resolved_route.model)

    async def test_fallback(self, logical_model: str, scenario: str) -> Dict[str, Any]:
        """Test fallback routing for a logical model."""
        print(f"\n{'=' * 60}")
        print(
            f"Testing fallback routing for '{logical_model}' with scenario '{scenario}'"
        )
        print(f"{'=' * 60}")

        self.call_log.clear()
        self.setup_mock_scenario(scenario)

        start_time = time.time()

        try:
            result = await call_with_fallback(logical_model, self.execute_route)
            duration = time.time() - start_time

            print("\n✅ SUCCESS")
            print(f"   Duration: {duration:.2f}s")
            print(f"   Final provider: {result['model']}")
            print(f"   Response: {result['choices'][0]['message']['content']}")

            return {
                "success": True,
                "duration": duration,
                "attempts": len(self.call_log),
                "final_provider": result.get("model"),
                "call_log": self.call_log.copy(),
            }

        except RoutingError as e:
            duration = time.time() - start_time
            print("\n❌ ALL ROUTES FAILED")
            print(f"   Duration: {duration:.2f}s")
            print(f"   Attempts made: {len(e.attempted_routes)}")
            print(f"   Error: {e.message}")

            return {
                "success": False,
                "duration": duration,
                "attempts": len(e.attempted_routes),
                "errors": [
                    {"attempt": a.attempt_number, "error": str(err)}
                    for a, err in zip(e.attempted_routes, e.errors)
                ],
                "call_log": self.call_log.copy(),
            }

    def show_call_log(self):
        """Display the call log."""
        if not self.call_log:
            print("No calls made.")
            return

        print("\n📋 Call Log:")
        print("-" * 80)
        for i, call in enumerate(self.call_log, 1):
            print(
                f"{i:2d}. {call['logical_model']} -> {call['provider']}/{call['model']} ({call['wire_protocol']})"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Test multi-level model fallback routing"
    )
    parser.add_argument(
        "logical_model", nargs="?", help="Logical model name to test (e.g., glm-4.6)"
    )
    parser.add_argument(
        "--scenario",
        choices=[
            "basic_success",
            "api_key_fallback",
            "provider_fallback",
            "logical_model_fallback",
            "timeout_fallback",
            "all_fail",
        ],
        default="basic_success",
        help="Test scenario to run",
    )
    parser.add_argument(
        "--list-models", action="store_true", help="List available logical models"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.list_models:
        print("Available logical models:")
        models = config_loader.get_available_models()
        for model in sorted(models):
            print(f"  - {model}")
        return

    if not args.logical_model:
        print("Error: logical_model is required unless --list-models is used")
        print("Use --list-models to see available models")
        return

    # Validate that the logical model exists
    try:
        config_loader.load_config(args.logical_model)
        print(f"✅ Logical model '{args.logical_model}' is configured")
    except (FileNotFoundError, ValueError) as e:
        print(f"❌ Error: {e}")
        print("Use --list-models to see available models")
        return

    # Run the test
    tester = FallbackTester()

    try:
        result = asyncio.run(tester.test_fallback(args.logical_model, args.scenario))

        if args.verbose:
            tester.show_call_log()

        print(f"\n{'=' * 60}")
        if result["success"]:
            print("🎉 TEST PASSED")
        else:
            print("💥 TEST FAILED")
        print(f"   Attempts: {result['attempts']}")
        print(f"   Duration: {result['duration']:.2f}s")
        print(f"{'=' * 60}")

    except KeyboardInterrupt:
        print("\n🛑 Test interrupted")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
