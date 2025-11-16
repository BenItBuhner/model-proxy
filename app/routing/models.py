"""
Pydantic models for routing configuration.
Defines the JSON schema for config/models/<logical_model>.json files.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal


class RouteConfig(BaseModel):
    """Configuration for a single provider route."""

    id: Optional[str] = Field(default=None, description="Optional identifier for this route (e.g., 'primary', 'secondary')")
    wire_protocol: Literal["openai", "anthropic"] = Field(..., description="API protocol this route uses")
    provider: str = Field(..., description="Provider name (e.g., 'cerebras', 'nahcrof', 'openai')")
    model: str = Field(..., description="Concrete model name to send to the provider")
    base_url: Optional[str] = Field(default=None, description="Override base URL for this provider")
    api_key_env: List[str] = Field(..., description="Ordered list of environment variable names to try for API keys")
    timeout_seconds: Optional[int] = Field(default=None, description="Timeout for this specific route")


class ModelRoutingConfig(BaseModel):
    """Configuration for a logical model with fallback routing."""

    logical_name: str = Field(..., description="The logical model name (e.g., 'glm-4.6')")
    timeout_seconds: Optional[int] = Field(default=60, description="Default timeout for all routes in seconds")
    model_routings: List[RouteConfig] = Field(..., description="Ordered list of provider routes to try")
    fallback_model_routings: List[str] = Field(default_factory=list, description="Ordered list of logical model names to fall back to")


class ResolvedRoute(BaseModel):
    """A concrete route resolved from a logical model config."""

    source_logical_model: str = Field(..., description="The logical model this route came from")
    wire_protocol: Literal["openai", "anthropic"] = Field(..., description="API protocol to use")
    provider: str = Field(..., description="Provider name")
    model: str = Field(..., description="Concrete model name to send")
    base_url: Optional[str] = Field(..., description="Base URL to use")
    api_key: str = Field(..., description="Resolved API key to use")
    timeout_seconds: int = Field(..., description="Timeout in seconds")
    route_id: Optional[str] = Field(default=None, description="Optional route identifier")


class Attempt(BaseModel):
    """An attempt to execute a route with fallback metadata."""

    route: ResolvedRoute
    attempt_number: int = Field(..., description="Sequential attempt number across all routes")
    is_fallback_route: bool = Field(default=False, description="Whether this is from a fallback logical model")


class RoutingError(Exception):
    """Error information when all routes fail."""

    def __init__(
        self,
        logical_model: str,
        attempted_routes: List[Attempt],
        errors: List[Dict[str, Any]],
        message: str
    ):
        super().__init__(message)
        self.logical_model = logical_model
        self.attempted_routes = attempted_routes
        self.errors = errors
        self.message = message
