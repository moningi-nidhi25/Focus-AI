import json
from typing import List

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.api.deps import owned_session, owned_session_or_404, require_roles
from app.models import ActivityEvent, FocusMetric, StudySession, User, UserRole
from app.schemas.analytics import FocusMetricRead
from app.schemas.tracking import EventBatch, EventBatchResult, MetricsSummary
from app.services import metrics_service
from app.utils.database import get_db

activity_router = APIRouter(prefix="/activity", tags=["Activity Tracking"])
analytics_router = APIRouter(prefix="/analytics", tags=["Learning Analytics"])
student_only = require_roles(UserRole.STUDENT)


@activity_router.post("/events", response_model=EventBatchResult, status_code=status.HTTP_201_CREATED)
def ingest_events(batch: EventBatch, user: User = Depends(student_only), db: Session = Depends(get_db)):
    """Store a batch of browser events and, at most once a minute, a FocusMetric snapshot."""
    session = owned_session_or_404(db, user, batch.session_id)
    for ev in batch.events:
        raw = json.dumps(ev.metadata)
        db.add(ActivityEvent(session_id=session.id, event_type=ev.event_type,
                             metadata_json=raw if len(raw) <= 4000 else "{}"))
    db.commit()
    snap = metrics_service.snapshot(db, session)
    return EventBatchResult(stored=len(batch.events), snapshot_created=snap is not None)


@analytics_router.get("/sessions/{session_id}/summary", response_model=MetricsSummary)
def session_summary(session: StudySession = Depends(owned_session), db: Session = Depends(get_db)):
    return MetricsSummary(session_id=session.id, **metrics_service.compute(db, session))


@analytics_router.get("/sessions/{session_id}/metrics", response_model=List[FocusMetricRead])
def session_metrics(limit: int = 500, session: StudySession = Depends(owned_session), db: Session = Depends(get_db)):
    """Focus/productivity time series for a session (oldest first)."""
    return (db.query(FocusMetric).filter(FocusMetric.session_id == session.id)
            .order_by(FocusMetric.calculated_at).limit(min(limit, 2000)).all())


@analytics_router.post("/sessions/{session_id}/snapshot", response_model=FocusMetricRead)
def force_snapshot(session: StudySession = Depends(owned_session), db: Session = Depends(get_db)):
    return metrics_service.snapshot(db, session, force=True)