# Load environment variables from .env file FIRST, before any other imports
import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import routers AFTER environment is loaded (they depend on auth which uses CLIENT_API_KEY)
from app.routers.openai import router as openai_router
from app.routers.anthropic import router as anthropic_router
from app.routers.health import router as health_router
from app.database import models
from app.database import logging_models
from app.database.database import engine
from app.middleware.logging_middleware import LoggingMiddleware
from app.core.provider_config import get_all_provider_configs
from app.core.startup_validation import validate_startup

# Create all database tables
models.Base.metadata.create_all(bind=engine)
logging_models.Base.metadata.create_all(bind=engine)

# Load provider configurations at startup
try:
    get_all_provider_configs()
except Exception as e:
    print(f"Warning: Failed to load some provider configs: {e}")

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
    version="1.0.0"
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
