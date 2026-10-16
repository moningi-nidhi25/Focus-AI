import json
from datetime import datetime
from typing import List, Optional

from sqlalchemy.orm import Session

from app.ai.context import build_profile
from app.models.study_session import SessionStatus, StudyContext, StudySession
from app.schemas.study_session import StudyContextCreate, StudySessionCreate, StudySessionUpdate


def _refresh_profile(ctx: StudyContext) -> None:
    """Re-derive the structured representation (keywords) and invalidate the cached embedding."""
    profile = build_profile(ctx.topic, ctx.learning_objectives or "", ctx.course_material or "")
    ctx.keywords = json.dumps(profile.keywords)
    ctx.context_embedding = None  # recomputed lazily by the relevance service


def create_study_session(db: Session, session_in: StudySessionCreate) -> StudySession:
    """Creates a new study session and initializes its study context."""
    db_session = StudySession(
        student_id=session_in.student_id,
        cohort_id=session_in.cohort_id,
        title=session_in.title,
        status=SessionStatus.ACTIVE,
    )
    db.add(db_session)
    db.commit()
    db.refresh(db_session)

    db_context = StudyContext(
        session_id=db_session.id,
        topic=session_in.topic,
        learning_objectives=session_in.learning_objectives,
        course_material=session_in.course_material,
    )
    _refresh_profile(db_context)
    db.add(db_context)
    db.commit()
    db.refresh(db_session)
    return db_session


def list_study_sessions(db: Session, student_id: Optional[str] = None, limit: int = 50) -> List[StudySession]:
    q = db.query(StudySession)
    if student_id:
        q = q.filter(StudySession.student_id == student_id)
    return q.order_by(StudySession.created_at.desc()).limit(limit).all()


def get_study_session(db: Session, session_id: str) -> Optional[StudySession]:
    return db.query(StudySession).filter(StudySession.id == session_id).first()


def update_study_session(db: Session, session_id: str, update_in: StudySessionUpdate) -> Optional[StudySession]:
    """Updates status/details; completing a session stamps its end time and duration."""
    db_session = get_study_session(db, session_id)
    if not db_session:
        return None
    for field, value in update_in.model_dump(exclude_unset=True).items():
        setattr(db_session, field, value)
    if db_session.status == SessionStatus.COMPLETED and db_session.end_time is None:
        db_session.end_time = datetime.utcnow()
        if not update_in.total_duration_seconds and db_session.start_time:
            db_session.total_duration_seconds = int((db_session.end_time - db_session.start_time).total_seconds())
    db.commit()
    db.refresh(db_session)
    return db_session


def create_or_update_study_context(db: Session, context_in: StudyContextCreate) -> StudyContext:
    """Creates or updates the active study context for a session."""
    ctx = db.query(StudyContext).filter(StudyContext.session_id == context_in.session_id).first()
    if ctx:
        ctx.topic = context_in.topic
        ctx.learning_objectives = context_in.learning_objectives
        if context_in.course_material is not None:
            ctx.course_material = context_in.course_material
    else:
        ctx = StudyContext(
            session_id=context_in.session_id,
            topic=context_in.topic,
            learning_objectives=context_in.learning_objectives,
            course_material=context_in.course_material,
        )
        db.add(ctx)
    _refresh_profile(ctx)
    db.commit()
    db.refresh(ctx)
    return ctx


def add_course_material(db: Session, session_id: str, text: str) -> Optional[StudyContext]:
    """Append extracted document text to the session's course material and rebuild the profile."""
    ctx = get_study_context_by_session(db, session_id)
    if not ctx:
        return None
    ctx.course_material = f"{ctx.course_material}\n\n{text}" if ctx.course_material else text
    _refresh_profile(ctx)
    db.commit()
    db.refresh(ctx)
    return ctx


def get_study_context_by_session(db: Session, session_id: str) -> Optional[StudyContext]:
    return db.query(StudyContext).filter(StudyContext.session_id == session_id).first()
