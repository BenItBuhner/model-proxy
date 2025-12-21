"""
Database models for structured request/response logging.
"""

from sqlalchemy import Column, Integer, String, Boolean, DateTime, JSON, Text, Index
from sqlalchemy.sql import func
from .database import Base


class RequestLog(Base):
    """
    Comprehensive logging model for all API requests and responses.
    Stores all context including parameters, messages, responses, and metadata.
    """

    __tablename__ = "request_logs"

    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(
        String, unique=True, nullable=False, index=True
    )  # UUID for request tracking
    timestamp = Column(DateTime, nullable=False, server_default=func.now(), index=True)

    # Request metadata
    endpoint = Column(String, nullable=False)  # e.g., "/v1/chat/completions"
    method = Column(String, nullable=False)  # "POST"
    client_api_key_hash = Column(String)  # Hashed client API key for security

    # Model and provider info
    requested_model = Column(
        String, nullable=False, index=True
    )  # Model requested by client
    resolved_provider = Column(
        String, nullable=False, index=True
    )  # Actual provider used
    resolved_model = Column(String, nullable=False)  # Actual model name used

    # Request parameters (stored as JSON for flexibility)
    parameters = Column(JSON, nullable=False)  # {temperature, max_tokens, top_p, etc.}

    # Request content
    messages = Column(JSON, nullable=False)  # Full message array

    # Response metadata
    response_status = Column(Integer)  # HTTP status code
    response_time_ms = Column(Integer)  # Time taken in milliseconds

    # Response content
    response_content = Column(Text)  # Full response text (for non-streaming)
    response_usage = Column(
        JSON
    )  # Token usage: {prompt_tokens, completion_tokens, total_tokens}

    # Provider details
    provider_api_key_used = Column(String)  # Which API key was used (hashed)
    provider_retries = Column(Integer, default=0)  # Number of retries needed

    # Streaming flag
    is_streaming = Column(Boolean, default=False)

    # Error tracking
    error_message = Column(Text)  # If request failed
    error_type = Column(String)  # Exception type if applicable

    # Create indexes for common queries
    __table_args__ = (
        Index("idx_timestamp", "timestamp"),
        Index("idx_provider", "resolved_provider"),
        Index("idx_model", "requested_model"),
        Index("idx_request_id", "request_id"),
    )
