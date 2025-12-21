"""
CRUD operations for structured request/response logging.
"""

from sqlalchemy.orm import Session
from sqlalchemy import and_
from datetime import datetime
from typing import Dict, Any, List, Optional
from . import logging_models


def create_request_log(
    db: Session,
    request_id: str,
    endpoint: str,
    method: str,
    requested_model: str,
    resolved_provider: str,
    resolved_model: str,
    parameters: Dict[str, Any],
    messages: List[Dict[str, Any]],
    client_api_key_hash: Optional[str] = None,
    **kwargs,
) -> logging_models.RequestLog:
    """
    Create a new request log entry.

    Args:
        db: Database session
        request_id: Unique request ID (UUID)
        endpoint: API endpoint path
        method: HTTP method
        requested_model: Model requested by client
        resolved_provider: Actual provider used
        resolved_model: Actual model name used
        parameters: Request parameters dict
        messages: Messages array
        client_api_key_hash: Hashed client API key
        **kwargs: Additional fields (response_status, response_time_ms, etc.)

    Returns:
        Created RequestLog instance
    """
    log_entry = logging_models.RequestLog(
        request_id=request_id,
        endpoint=endpoint,
        method=method,
        requested_model=requested_model,
        resolved_provider=resolved_provider,
        resolved_model=resolved_model,
        parameters=parameters,
        messages=messages,
        client_api_key_hash=client_api_key_hash,
        **kwargs,
    )
    db.add(log_entry)
    db.commit()
    db.refresh(log_entry)
    return log_entry


def update_request_log(
    db: Session, request_id: str, **updates
) -> Optional[logging_models.RequestLog]:
    """
    Update an existing request log entry.

    Args:
        db: Database session
        request_id: Request ID to update
        **updates: Fields to update (response_status, response_content, etc.)

    Returns:
        Updated RequestLog instance or None if not found
    """
    log_entry = (
        db.query(logging_models.RequestLog)
        .filter(logging_models.RequestLog.request_id == request_id)
        .first()
    )

    if not log_entry:
        return None

    for key, value in updates.items():
        if hasattr(log_entry, key):
            setattr(log_entry, key, value)

    db.commit()
    db.refresh(log_entry)
    return log_entry


def get_request_by_id(
    db: Session, request_id: str
) -> Optional[logging_models.RequestLog]:
    """
    Retrieve a request log by request ID.

    Args:
        db: Database session
        request_id: Request ID to lookup

    Returns:
        RequestLog instance or None if not found
    """
    return (
        db.query(logging_models.RequestLog)
        .filter(logging_models.RequestLog.request_id == request_id)
        .first()
    )


def get_requests_by_time_range(
    db: Session, start_time: datetime, end_time: datetime, limit: int = 1000
) -> List[logging_models.RequestLog]:
    """
    Get requests within a time range.

    Args:
        db: Database session
        start_time: Start datetime
        end_time: End datetime
        limit: Maximum number of results

    Returns:
        List of RequestLog instances
    """
    return (
        db.query(logging_models.RequestLog)
        .filter(
            and_(
                logging_models.RequestLog.timestamp >= start_time,
                logging_models.RequestLog.timestamp <= end_time,
            )
        )
        .order_by(logging_models.RequestLog.timestamp.desc())
        .limit(limit)
        .all()
    )


def get_requests_by_provider(
    db: Session,
    provider: str,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: int = 1000,
) -> List[logging_models.RequestLog]:
    """
    Get requests filtered by provider.

    Args:
        db: Database session
        provider: Provider name
        start_time: Optional start datetime
        end_time: Optional end datetime
        limit: Maximum number of results

    Returns:
        List of RequestLog instances
    """
    query = db.query(logging_models.RequestLog).filter(
        logging_models.RequestLog.resolved_provider == provider
    )

    if start_time:
        query = query.filter(logging_models.RequestLog.timestamp >= start_time)
    if end_time:
        query = query.filter(logging_models.RequestLog.timestamp <= end_time)

    return query.order_by(logging_models.RequestLog.timestamp.desc()).limit(limit).all()


def get_requests_by_model(
    db: Session,
    model: str,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: int = 1000,
) -> List[logging_models.RequestLog]:
    """
    Get requests filtered by model.

    Args:
        db: Database session
        model: Model name
        start_time: Optional start datetime
        end_time: Optional end datetime
        limit: Maximum number of results

    Returns:
        List of RequestLog instances
    """
    query = db.query(logging_models.RequestLog).filter(
        logging_models.RequestLog.requested_model == model
    )

    if start_time:
        query = query.filter(logging_models.RequestLog.timestamp >= start_time)
    if end_time:
        query = query.filter(logging_models.RequestLog.timestamp <= end_time)

    return query.order_by(logging_models.RequestLog.timestamp.desc()).limit(limit).all()


def get_usage_stats(
    db: Session,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Get aggregate usage statistics.

    Args:
        db: Database session
        start_time: Optional start datetime
        end_time: Optional end datetime

    Returns:
        Dictionary with usage statistics
    """
    query = db.query(logging_models.RequestLog)

    if start_time:
        query = query.filter(logging_models.RequestLog.timestamp >= start_time)
    if end_time:
        query = query.filter(logging_models.RequestLog.timestamp <= end_time)

    logs = query.all()

    total_requests = len(logs)
    successful_requests = sum(
        1 for log in logs if log.response_status and 200 <= log.response_status < 300
    )
    failed_requests = total_requests - successful_requests

    total_tokens = 0
    prompt_tokens = 0
    completion_tokens = 0
    total_latency_ms = 0

    for log in logs:
        if log.response_usage:
            usage = log.response_usage
            if isinstance(usage, dict):
                total_tokens += usage.get("total_tokens", 0)
                prompt_tokens += usage.get("prompt_tokens", 0)
                completion_tokens += usage.get("completion_tokens", 0)

        if log.response_time_ms:
            total_latency_ms += log.response_time_ms

    avg_latency_ms = total_latency_ms / total_requests if total_requests > 0 else 0
    error_rate = (failed_requests / total_requests * 100) if total_requests > 0 else 0

    return {
        "total_requests": total_requests,
        "successful_requests": successful_requests,
        "failed_requests": failed_requests,
        "error_rate_percent": round(error_rate, 2),
        "total_tokens": total_tokens,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "avg_latency_ms": round(avg_latency_ms, 2),
        "total_latency_ms": total_latency_ms,
    }


def get_error_logs(
    db: Session,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: int = 100,
) -> List[logging_models.RequestLog]:
    """
    Get error logs for analysis.

    Args:
        db: Database session
        start_time: Optional start datetime
        end_time: Optional end datetime
        limit: Maximum number of results

    Returns:
        List of RequestLog instances with errors
    """
    query = db.query(logging_models.RequestLog).filter(
        logging_models.RequestLog.error_message.isnot(None)
    )

    if start_time:
        query = query.filter(logging_models.RequestLog.timestamp >= start_time)
    if end_time:
        query = query.filter(logging_models.RequestLog.timestamp <= end_time)

    return query.order_by(logging_models.RequestLog.timestamp.desc()).limit(limit).all()
