"""
Tests for logging database models and CRUD operations.
"""

import pytest
from datetime import datetime, timedelta, timezone
from sqlalchemy.orm import Session
from app.database.database import SessionLocal, engine
from app.database import logging_models
from app.database import logging_crud
from app.core.logging import generate_request_id


@pytest.fixture(scope="function")
def db_session():
    """Create a database session for testing."""
    # Create tables
    logging_models.Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        # Clean up tables
        logging_models.Base.metadata.drop_all(bind=engine)


def test_create_request_log(db_session: Session):
    """Test creating a request log entry."""
    request_id = generate_request_id()
    log = logging_crud.create_request_log(
        db=db_session,
        request_id=request_id,
        endpoint="/v1/chat/completions",
        method="POST",
        requested_model="gpt-4",
        resolved_provider="openai",
        resolved_model="gpt-4",
        parameters={"temperature": 0.7, "max_tokens": 100},
        messages=[{"role": "user", "content": "Hello"}],
        client_api_key_hash="test_hash",
    )

    assert log.request_id == request_id
    assert log.endpoint == "/v1/chat/completions"
    assert log.requested_model == "gpt-4"
    assert log.resolved_provider == "openai"
    assert not log.is_streaming


def test_update_request_log(db_session: Session):
    """Test updating a request log entry."""
    request_id = generate_request_id()
    logging_crud.create_request_log(
        db=db_session,
        request_id=request_id,
        endpoint="/v1/chat/completions",
        method="POST",
        requested_model="gpt-4",
        resolved_provider="openai",
        resolved_model="gpt-4",
        parameters={},
        messages=[],
    )

    updated = logging_crud.update_request_log(
        db=db_session,
        request_id=request_id,
        response_status=200,
        response_time_ms=150,
        response_content="Hello there!",
        response_usage={
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        },
    )

    assert updated.response_status == 200
    assert updated.response_time_ms == 150
    assert updated.response_content == "Hello there!"
    assert updated.response_usage["total_tokens"] == 30


def test_get_request_by_id(db_session: Session):
    """Test retrieving a request log by ID."""
    request_id = generate_request_id()
    logging_crud.create_request_log(
        db=db_session,
        request_id=request_id,
        endpoint="/v1/chat/completions",
        method="POST",
        requested_model="gpt-4",
        resolved_provider="openai",
        resolved_model="gpt-4",
        parameters={},
        messages=[],
    )

    log = logging_crud.get_request_by_id(db_session, request_id)
    assert log is not None
    assert log.request_id == request_id


def test_get_request_by_id_not_found(db_session: Session):
    """Test retrieving non-existent request log."""
    log = logging_crud.get_request_by_id(db_session, "nonexistent-id")
    assert log is None


def test_get_requests_by_time_range(db_session: Session):
    """Test retrieving requests by time range."""
    request_id1 = generate_request_id()
    request_id2 = generate_request_id()

    logging_crud.create_request_log(
        db=db_session,
        request_id=request_id1,
        endpoint="/v1/chat/completions",
        method="POST",
        requested_model="gpt-4",
        resolved_provider="openai",
        resolved_model="gpt-4",
        parameters={},
        messages=[],
    )

    logging_crud.create_request_log(
        db=db_session,
        request_id=request_id2,
        endpoint="/v1/chat/completions",
        method="POST",
        requested_model="gpt-4",
        resolved_provider="openai",
        resolved_model="gpt-4",
        parameters={},
        messages=[],
    )

    start_time = datetime.now(timezone.utc) - timedelta(hours=1)
    end_time = datetime.now(timezone.utc) + timedelta(hours=1)

    logs = logging_crud.get_requests_by_time_range(db_session, start_time, end_time)
    assert len(logs) >= 2


def test_get_requests_by_provider(db_session: Session):
    """Test retrieving requests by provider."""
    request_id = generate_request_id()
    logging_crud.create_request_log(
        db=db_session,
        request_id=request_id,
        endpoint="/v1/chat/completions",
        method="POST",
        requested_model="gpt-4",
        resolved_provider="openai",
        resolved_model="gpt-4",
        parameters={},
        messages=[],
    )

    logs = logging_crud.get_requests_by_provider(db_session, "openai")
    assert len(logs) >= 1
    assert all(log.resolved_provider == "openai" for log in logs)


def test_get_requests_by_model(db_session: Session):
    """Test retrieving requests by model."""
    request_id = generate_request_id()
    logging_crud.create_request_log(
        db=db_session,
        request_id=request_id,
        endpoint="/v1/chat/completions",
        method="POST",
        requested_model="gpt-4",
        resolved_provider="openai",
        resolved_model="gpt-4",
        parameters={},
        messages=[],
    )

    logs = logging_crud.get_requests_by_model(db_session, "gpt-4")
    assert len(logs) >= 1
    assert all(log.requested_model == "gpt-4" for log in logs)


def test_get_usage_stats(db_session: Session):
    """Test getting usage statistics."""
    request_id1 = generate_request_id()
    request_id2 = generate_request_id()

    logging_crud.create_request_log(
        db=db_session,
        request_id=request_id1,
        endpoint="/v1/chat/completions",
        method="POST",
        requested_model="gpt-4",
        resolved_provider="openai",
        resolved_model="gpt-4",
        parameters={},
        messages=[],
    )

    logging_crud.update_request_log(
        db=db_session,
        request_id=request_id1,
        response_status=200,
        response_time_ms=100,
        response_usage={
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        },
    )

    logging_crud.create_request_log(
        db=db_session,
        request_id=request_id2,
        endpoint="/v1/chat/completions",
        method="POST",
        requested_model="gpt-4",
        resolved_provider="openai",
        resolved_model="gpt-4",
        parameters={},
        messages=[],
    )

    logging_crud.update_request_log(
        db=db_session,
        request_id=request_id2,
        response_status=200,
        response_time_ms=200,
        response_usage={
            "prompt_tokens": 15,
            "completion_tokens": 25,
            "total_tokens": 40,
        },
    )

    stats = logging_crud.get_usage_stats(db_session)
    assert stats["total_requests"] >= 2
    assert stats["successful_requests"] >= 2
    assert stats["total_tokens"] >= 70
    assert stats["avg_latency_ms"] > 0


def test_get_error_logs(db_session: Session):
    """Test retrieving error logs."""
    request_id = generate_request_id()
    logging_crud.create_request_log(
        db=db_session,
        request_id=request_id,
        endpoint="/v1/chat/completions",
        method="POST",
        requested_model="gpt-4",
        resolved_provider="openai",
        resolved_model="gpt-4",
        parameters={},
        messages=[],
    )

    logging_crud.update_request_log(
        db=db_session,
        request_id=request_id,
        response_status=500,
        error_message="Test error",
        error_type="Exception",
    )

    error_logs = logging_crud.get_error_logs(db_session)
    assert len(error_logs) >= 1
    assert any(log.error_message == "Test error" for log in error_logs)
