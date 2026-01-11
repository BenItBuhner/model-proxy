"""
Entry point for running the app module as a script.

Usage:
    python -m app
    python -m app --help
    python -m app start --port 8000
"""

from app.cli_main import main

if __name__ == "__main__":
    main()
