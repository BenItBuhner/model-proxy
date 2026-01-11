"""
Interactive CLI module for Model-Proxy configuration management.

This module provides interactive command-line interfaces for managing
providers, models, and API keys with arrow key navigation and guided workflows.

All interactive functions handle Ctrl+C (KeyboardInterrupt) gracefully,
allowing users to exit at any point without crashing.
"""

from app.cli.config_manager import ConfigManager
from app.cli.interactive import (
    UserCancelled,
    ask_batch_addition,
    ask_password,
    ask_provider_selection,
    ask_text,
    ask_yes_no,
    display_error,
    display_existing_items,
    display_info,
    display_success,
    display_warning,
    handle_user_cancelled,
    safe_select,
    select_existing_provider,
)

__all__ = [
    "ConfigManager",
    "UserCancelled",
    "ask_provider_selection",
    "ask_yes_no",
    "ask_text",
    "ask_password",
    "display_existing_items",
    "ask_batch_addition",
    "display_success",
    "display_error",
    "display_warning",
    "display_info",
    "handle_user_cancelled",
    "safe_select",
    "select_existing_provider",
]
