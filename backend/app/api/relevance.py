from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.ai.explainability import METHOD
from app.ai.relevance_engine import get_engine
from app.api.deps import current_user, owned_session, owned_session_or_404
from app.models import StudySession, User
from app.schemas.relevance import (
    AnalyzeRequest, AnalyzeResponse, EngineInfo, ExplainTokensRequest, ExplainTokensResponse, HistoryItem, OverrideRequest,
)
from app.services import relevance_service
from app.utils.database import get_db

router = APIRouter(prefix="/relevance", tags=["Relevance Engine & Explainability"])


@router.get("/engine", response_model=EngineInfo)
def engine_info():
    """Which embedding backend / LLM is active and the decision thresholds."""
    e = get_engine()
    return EngineInfo(
        embedding_backend=e.embedder.name,
        dense_embeddings=e.embedder.dense,
        llm_enabled=e.llm is not None,
        llm_provider=getattr(e.llm, "provider", None),
        llm_model=getattr(e.llm, "model", None),
        allow_threshold=e.config.allow_threshold,
        warn_threshold=e.config.warn_threshold,
        explanation_method=METHOD,
    )


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze_content(payload: AnalyzeRequest, user: User = Depends(current_user), db: Session = Depends(get_db)):
    """Score content against the study context and return ALLOW / WARN / BLOCK with an explanation,
    matching concepts and a sentence-level relevance heatmap."""
    if payload.session_id:
        owned_session_or_404(db, user, payload.session_id)
    try:
        return relevance_service.analyze(db, payload)
    except relevance_service.SessionNotFound:
        raise HTTPException(status_code=404, detail="Study session or context not found")
    except relevance_service.ContentUnavailable as exc:
        raise HTTPException(status_code=422, detail=f"Could not read the content: {exc}")


@router.post("/explain-tokens", response_model=ExplainTokensResponse)
def explain_tokens(payload: ExplainTokensRequest, user: User = Depends(current_user), db: Session = Depends(get_db)):
    """Optional word-level Integrated Gradients for ONE sentence (target: cosine similarity to the study context).

    Always answers 200 for supported requests: ``available=false`` + ``reason`` when the model cannot provide it,
    so callers can fall back to the sentence-level heatmap."""
    if payload.session_id:
        owned_session_or_404(db, user, payload.session_id)
    try:
        return relevance_service.explain_tokens(db, payload)
    except relevance_service.SessionNotFound:
        raise HTTPException(status_code=404, detail="Study session or context not found")


@router.post("/override", status_code=status.HTTP_204_NO_CONTENT)
def override_block(payload: OverrideRequest, user: User = Depends(current_user), db: Session = Depends(get_db)):
    """Log that the student continued past a block (counted as a distraction attempt)."""
    owned_session_or_404(db, user, payload.session_id)
    try:
        relevance_service.record_override(db, payload.session_id, payload.url, payload.title)
    except relevance_service.SessionNotFound:
        raise HTTPException(status_code=404, detail="Study session not found")


@router.get("/sessions/{session_id}/history", response_model=List[HistoryItem])
def analysis_history(limit: int = 50, session: StudySession = Depends(owned_session), db: Session = Depends(get_db)):
    rows = relevance_service.history(db, session.id, min(limit, 200))
    return [
        HistoryItem(
            id=r.id, url=r.url, title=r.title, relevance_score=r.relevance_score,
            classification=r.classification, decision=r.decision.value, analyzed_at=r.analyzed_at,
        )
        for r in rows
    ]