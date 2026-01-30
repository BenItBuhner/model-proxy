# Load environment variables from .env file FIRST, before any other imports
import os

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.api_key_manager import get_available_keys
from app.core.provider_config import get_all_provider_configs, is_provider_enabled
from app.core.startup_validation import validate_startup
from app.database import logging_models, models
from app.database.database import engine
from app.middleware.logging_middleware import LoggingMiddleware
from app.routers.anthropic import router as anthropic_router
from app.routers.health import router as health_router

# Import routers AFTER environment is loaded (they depend on auth which uses CLIENT_API_KEY)
from app.routers.openai import router as openai_router
from app.setup_ui import router as setup_ui_router

# Create all database tables
models.Base.metadata.create_all(bind=engine)
logging_models.Base.metadata.create_all(bind=engine)

# Load provider configurations at startup
provider_configs = {}
try:
    provider_configs = get_all_provider_configs()
except Exception as e:
    print(f"Warning: Failed to load some provider configs: {e}")
    provider_configs = {}

# Print provider key counts before startup validation
if provider_configs:
    printed = False
    for provider_name in sorted(provider_configs.keys()):
        if not is_provider_enabled(provider_name):
            continue
        key_count = len(get_available_keys(provider_name))
        if key_count == 0:
            continue
        if not printed:
            print("Provider API keys:")
            printed = True
        print(f"- {provider_name}: {key_count}")
    if not printed:
        print("Provider API keys: none configured")
else:
    print("Provider API keys: no provider configs loaded")

# Perform startup validation
try:
    validate_startup()
except Exception as e:
    # In production, you might want to fail fast
    # For now, we'll log and continue
    print(f"Startup validation error: {e}")
    if os.getenv("FAIL_ON_STARTUP_VALIDATION", "false").lower() == "true":
        raise

app = FastAPI(
    title="Centralized Inference Endpoint",
    description="Multi-provider LLM inference proxy with API key fallback",
    version="1.0.0",
)

# Add CORS middleware (first - outermost)
cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add logging middleware (second - sets request_id and client_api_key_hash)
app.add_middleware(LoggingMiddleware)

# Rate limiting removed per request

# Include routers
app.include_router(openai_router)
app.include_router(anthropic_router)
app.include_router(health_router)
app.include_router(setup_ui_router)

# Mount static files for setup UI
from fastapi.staticfiles import StaticFiles
from pathlib import Path

static_dir = Path(__file__).parent / "setup_ui" / "static"
if static_dir.exists():
    app.mount(
        "/setup/static", StaticFiles(directory=str(static_dir)), name="setup_static"
    )
