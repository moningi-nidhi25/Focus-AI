from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.api.deps import owned_session, require_roles
from app.models import StudySession, User, UserRole
from app.schemas.study_session import (
    StudyContextCreate,
    StudyContextRead,
    StudySessionCreate,
    StudySessionRead,
    StudySessionUpdate,
)
from app.services import context_service
from app.services.content_extractor import MAX_UPLOAD_BYTES, extract_document
from app.utils.database import get_db

router = APIRouter(prefix="/sessions", tags=["Study Sessions & Context"])
student_only = require_roles(UserRole.STUDENT)


@router.post("", response_model=StudySessionRead, status_code=status.HTTP_201_CREATED)
def create_session(session_in: StudySessionCreate, student: User = Depends(student_only),
                   db: Session = Depends(get_db)):
    """Create a new study session with topic and learning objectives (always for the logged-in student)."""
    if session_in.student_id != student.id:
        raise HTTPException(status_code=403, detail="You can only create sessions for yourself")
    return context_service.create_study_session(db, session_in)


@router.get("", response_model=List[StudySessionRead])
def list_sessions(student: User = Depends(student_only), db: Session = Depends(get_db)):
    """The logged-in student's recent study sessions."""
    return context_service.list_study_sessions(db, student.id)


@router.get("/{session_id}", response_model=StudySessionRead)
def get_session(session: StudySession = Depends(owned_session)):
    return session


@router.patch("/{session_id}", response_model=StudySessionRead)
def update_session(update_in: StudySessionUpdate, session: StudySession = Depends(owned_session),
                   db: Session = Depends(get_db)):
    updated = context_service.update_study_session(db, session.id, update_in)
    if not updated:
        raise HTTPException(status_code=404, detail="Study session not found")
    return updated


@router.post("/{session_id}/context", response_model=StudyContextRead)
def set_study_context(context_in: StudyContextCreate, session: StudySession = Depends(owned_session),
                      db: Session = Depends(get_db)):
    """Create or update the active study context for a session."""
    context_in.session_id = session.id
    return context_service.create_or_update_study_context(db, context_in)


@router.get("/{session_id}/context", response_model=StudyContextRead)
def get_study_context(session: StudySession = Depends(owned_session), db: Session = Depends(get_db)):
    db_context = context_service.get_study_context_by_session(db, session.id)
    if not db_context:
        raise HTTPException(status_code=404, detail="Study context not found for this session")
    return db_context


@router.post("/{session_id}/context/material", response_model=StudyContextRead)
async def upload_course_material(file: UploadFile = File(...), session: StudySession = Depends(owned_session),
                                 db: Session = Depends(get_db)):
    """Attach a course document (PDF, DOCX, TXT, MD); its key terms enrich the study context."""
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    try:
        extracted = extract_document(file.filename or "", data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        raise HTTPException(status_code=422, detail="The file could not be read.")
    if not extracted["cleaned_text"]:
        raise HTTPException(status_code=422, detail=extracted.get("error", "No text found in the file."))
    ctx = context_service.add_course_material(db, session.id, extracted["cleaned_text"])
    if not ctx:
        raise HTTPException(status_code=404, detail="Study context not found for this session")
    return ctx