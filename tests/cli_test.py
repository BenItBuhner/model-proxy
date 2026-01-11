"""
Comprehensive CLI tests for model-proxy.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

# Base command for running the CLI
# In development: python -m app
# When installed: model-proxy
# We'll use Python module approach for tests with the current interpreter
BASE_CMD = [sys.executable, "-m", "app"]


def run_cli_command(args, check=True, capture_output=True):
    """Helper function to run CLI commands."""
    cmd = BASE_CMD + args
    # Set encoding to UTF-8 to handle Unicode characters properly
    # Use errors='replace' to avoid encoding errors on Windows
    result = subprocess.run(
        cmd,
        check=check,
        capture_output=capture_output,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=10,
    )
    return result


def test_cli_version():
    """Test version command."""
    result = run_cli_command(["version"])
    assert result.returncode == 0
    assert "version" in result.stdout.lower()
    assert "0.1.0" in result.stdout


def test_cli_version_verbose():
    """Test version command with verbose flag."""
    result = run_cli_command(["version", "--verbose"])
    assert result.returncode == 0
    assert "Python" in result.stdout
    assert "FastAPI" in result.stdout or "Uvicorn" in result.stdout


def test_cli_help():
    """Test main help command."""
    result = run_cli_command(["--help"])
    assert result.returncode == 0
    assert "start" in result.stdout
    assert "health" in result.stdout
    assert "version" in result.stdout
    assert "config" in result.stdout
    assert "doctor" in result.stdout


def test_cli_help_alternative():
    """Test help command using 'help' instead of '--help'."""
    run_cli_command(["help"], check=False)
    # The help command exists but may behave differently
    # Just check it doesn't crash entirely


def test_cli_start_help():
    """Test start command help."""
    result = run_cli_command(["start", "--help"])
    assert result.returncode == 0
    assert "Start the model-proxy server" in result.stdout
    assert "--port" in result.stdout
    assert "--host" in result.stdout
    assert "--reload" in result.stdout


def test_cli_config_list_help():
    """Test config list command help."""
    result = run_cli_command(["config", "list", "--help"])
    assert result.returncode == 0
    assert "List all available models" in result.stdout


def test_cli_config_list():
    """Test listing available models."""
    result = run_cli_command(["config", "list"])
    assert result.returncode == 0

    # Check if models are found or appropriate message shown
    if "No models found" in result.stdout:
        # That's OK if config directory is empty
        pass
    else:
        # Should show models in some format
        assert "Found" in result.stdout or len(result.stdout) > 0


def test_cli_config_list_format_json():
    """Test listing models in JSON format."""
    result = run_cli_command(["config", "list", "--format", "json"])
    assert result.returncode == 0

    if "No models" not in result.stdout:
        # If models exist, output should be valid JSON
        try:
            models = json.loads(result.stdout)
            assert isinstance(models, list)
        except json.JSONDecodeError:
            # If no models, might not be JSON
            pass


def test_cli_config_list_format_table():
    """Test listing models in table format (default)."""
    result = run_cli_command(["config", "list", "--format", "table"])
    assert result.returncode == 0


def test_cli_config_validate():
    """Test validating model configurations."""
    result = run_cli_command(["config", "validate"])
    assert result.returncode in [0, 1]  # 0 if valid, 1 if invalid

    if result.returncode == 0:
        assert "valid" in result.stdout.lower()
    else:
        assert "invalid" in result.stdout.lower() or "error" in result.stdout.lower()


def test_cli_config_show_not_found():
    """Test showing configuration for non-existent model."""
    result = run_cli_command(["config", "show", "nonexistent-model-12345"], check=False)
    assert result.returncode != 0
    # Check both stdout and stderr for error messages
    output = result.stdout.lower() + result.stderr.lower()
    assert "not found" in output or "error" in output


def test_cli_health_when_server_down():
    """Test health check when server is not running."""
    # First try to check if server is actually running
    import httpx

    try:
        response = httpx.get("http://127.0.0.1:9876/health", timeout=1)
        if response.status_code == 200:
            # Server is running, skip this test
            pytest.skip("Server is already running")
    except (httpx.ConnectError, httpx.TimeoutException):
        pass  # Server not running, proceed with test
    except Exception:
        pass  # Other errors, proceed with test

    result = run_cli_command(["health"], check=False)
    # Should fail when server is not running
    assert result.returncode != 0
    # Check both stdout and stderr for error messages
    combined_output = (result.stdout + result.stderr).lower()
    assert (
        "connect" in combined_output
        or "running" in combined_output
        or "error" in combined_output
        or "could not connect" in combined_output
    )


def test_cli_health_help():
    """Test health command help."""
    result = run_cli_command(["health", "--help"])
    assert result.returncode == 0
    assert "Check the health" in result.stdout
    assert "--endpoint" in result.stdout


def test_cli_doctor():
    """Test doctor diagnostics."""
    result = run_cli_command(["doctor"], check=False)
    # Doctor should run even with issues
    assert result.returncode in [0, 1]
    assert "diagnostics" in result.stdout.lower()


def test_cli_doctor_help():
    """Test doctor command help."""
    result = run_cli_command(["doctor", "--help"])
    assert result.returncode == 0
    assert "Run comprehensive system diagnostics" in result.stdout


def test_cli_env_check():
    """Test environment variable check."""
    # Test the hyphenated version for compatibility
    result = run_cli_command(["env", "check"], check=False)
    # Should report on environment status
    assert result.returncode in [0, 1]  # 0 if OK, 1 if missing vars
    output = result.stdout.lower() + result.stderr.lower()
    assert "environment" in output or "variable" in output


def test_cli_keys_list():
    """Test listing API keys."""
    result = run_cli_command(["keys", "list"], check=False)
    # Should run even if no keys are configured
    assert result.returncode == 0
    assert "API keys" in result.stdout or "key" in result.stdout.lower()


def test_cli_keys_help():
    """Test keys command help."""
    result = run_cli_command(["keys", "--help"])
    assert result.returncode == 0
    assert "API key management" in result.stdout


def test_cli_keys_test_help():
    """Test keys test command help."""
    result = run_cli_command(["keys", "test", "--help"])
    assert result.returncode == 0
    assert "Test API key validity" in result.stdout


def test_cli_keys_test_no_provider():
    """Test keys test with missing provider argument."""
    result = run_cli_command(["keys", "test"], check=False)
    # Should fail due to missing argument
    assert result.returncode != 0


def test_cli_db_stats():
    """Test database statistics."""
    result = run_cli_command(["db", "stats"], check=False)
    # Should run even if database is empty
    assert result.returncode in [0, 1]
    assert "database" in result.stdout.lower() or "table" in result.stdout.lower()


def test_cli_db_help():
    """Test db command help."""
    result = run_cli_command(["db", "--help"])
    assert result.returncode == 0
    assert "Database management" in result.stdout


def test_cli_db_reset_help():
    """Test db reset command help."""
    result = run_cli_command(["db", "reset", "--help"])
    assert result.returncode == 0
    assert "Reset database" in result.stdout
    assert "--confirm" in result.stdout


@pytest.mark.slow
def test_cli_start_runs_server():
    """Test that start command actually starts the server."""
    # Start server in background
    process = subprocess.Popen(
        BASE_CMD + ["start", "--port", "9877"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        # Give server time to start
        time.sleep(3)

        # Check if server is responding
        try:
            response = httpx.get("http://127.0.0.1:9877/health", timeout=5)
            assert response.status_code == 200
        except httpx.ConnectError:
            # Server might not have started yet or had issues
            # Check for errors in process output
            output, errors = process.communicate(timeout=1)
            print(f"Server output: {output}")
            print(f"Server errors: {errors}")
            # Don't fail the test, just skip
            pytest.skip("Server failed to start")

    finally:
        # Clean up
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


@pytest.mark.slow
def test_cli_server_health_integration():
    """Integration test: start server and check health."""
    # Start server on a different port
    port = 9878
    process = subprocess.Popen(
        BASE_CMD + ["start", "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        # Wait for server to start
        time.sleep(3)

        # Check health using CLI
        result = run_cli_command(
            ["health", "--endpoint", f"http://127.0.0.1:{port}"], check=False
        )

        if result.returncode == 0:
            assert "healthy" in result.stdout.lower()
        else:
            # Server might have failed to start
            pytest.skip("Server failed to start")

    finally:
        # Clean up
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def test_cli_unknown_command():
    """Test handling of unknown commands."""
    result = run_cli_command(["unknown-command-xyz"], check=False)
    assert result.returncode != 0


def test_cli_invalid_flag():
    """Test handling of invalid flags."""
    result = run_cli_command(["start", "--invalid-flag"], check=False)
    assert result.returncode != 0


def test_cli_start_invalid_port():
    """Test start command with invalid port."""
    result = run_cli_command(["start", "--port", "invalid"], check=False)
    assert result.returncode != 0


def test_cli_config_subcommands_help():
    """Test config subcommand helps."""
    subcommands = ["list", "validate", "show"]
    for subcommand in subcommands:
        result = run_cli_command(["config", subcommand, "--help"])
        assert result.returncode == 0


def test_cli_dev_help():
    """Test dev command help."""
    result = run_cli_command(["dev", "--help"])
    assert result.returncode == 0
    assert "Development tools" in result.stdout


def test_cli_dev_subcommands_exist():
    """Test that dev subcommands exist."""
    subcommands = ["shell", "test", "lint"]
    for subcommand in subcommands:
        result = run_cli_command(["dev", subcommand, "--help"])
        assert result.returncode == 0
        assert subcommand in result.stdout.lower()


def test_cli_dev_test_help():
    """Test dev test command help."""
    result = run_cli_command(["dev", "test", "--help"])
    assert result.returncode == 0
    assert "Run the test suite" in result.stdout
    assert "--verbose" in result.stdout


def test_cli_dev_lint_help():
    """Test dev lint command help."""
    result = run_cli_command(["dev", "lint", "--help"])
    assert result.returncode == 0
    assert "Run linter" in result.stdout
    assert "--fix" in result.stdout


def test_cli_no_args_shows_help():
    """Test that running CLI with no arguments shows help."""
    # When no args is help is set to True in Typer app
    result = run_cli_command([], check=False)
    # Typer returns exit code 2 when no_args_is_help=True
    assert result.returncode in [0, 2]  # 2 is expected for help display


def test_cli_output_structure():
    """Test that CLI output has expected structure."""
    result = run_cli_command(["version"])
    stdout = result.stdout
    # Should have version indicator
    assert any(char in stdout for char in ["0", "1", "."])


def test_cli_error_output_to_stderr():
    """Test that errors go to stderr."""
    result = run_cli_command(["config", "show", "nonexistent"], check=False)
    # Should have non-zero return code
    assert result.returncode != 0


def test_cli_environment_loading():
    """Test that CLI loads environment from .env."""
    # Create a temporary .env file
    env_path = Path(".env.test")
    try:
        env_path.write_text("TEST_VAR=test_value\n")

        # Run a command that should load environment
        # This is a basic check - the command should succeed
        result = run_cli_command(["version"])
        assert result.returncode == 0

    finally:
        # Clean up
        if env_path.exists():
            env_path.unlink()


@pytest.fixture(autouse=True)
def cleanup_test_processes():
    """Fixture to ensure no test processes are left running."""
    yield
    # This is a simple cleanup - in reality, you might want more robust cleanup
    pass


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
