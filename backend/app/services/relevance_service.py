"""Glue between the API, the relevance engine and the database (Phases 4-6)."""
import json
import logging
from typing import Optional, Tuple

import numpy as np
from sqlalchemy.orm import Session

from app.ai.context import ContextProfile, build_profile
from app.ai.explainability import level_for
from app.ai.relevance_engine import AnalysisResult, ContentInput, get_engine
from app.models.activity import ActivityEvent, BlockedContent
from app.models.content_analysis import ContentAnalysis, DecisionEnum
from app.schemas.relevance import (
    AnalyzeRequest, AnalyzeResponse, ExplainTokensRequest, ExplainTokensResponse, SentenceScore,
    TokenPart, WordAttributionOut,
)
from app.services import context_service
from app.services.content_extractor import extract_and_normalize, extract_youtube_content, extract_youtube_id

logger = logging.getLogger("focusai.relevance")
EVENT_FOR_DECISION = {"ALLOW": "content_allowed", "WARN": "warning_shown", "BLOCK": "content_blocked"}


class SessionNotFound(Exception):
    pass


class ContentUnavailable(Exception):
    pass


def _resolve_content(req: AnalyzeRequest) -> dict:
    """Turn the request into analysable parts: title, description, metadata text and body.

    Normal pages: use what the extension sent; only go to the network when only a URL was given.
    YouTube: ALWAYS analyse the individual video. The extension supplies what the page shows (title,
    description, channel, hashtags); the backend adds the authoritative title, tags/category and the
    transcript. Each part is best-effort, so a missing transcript degrades gracefully to
    title + description + metadata instead of failing.
    """
    yt_id = extract_youtube_id(req.url) if req.url else None
    if yt_id:
        meta = req.metadata
        client_desc = (meta.description if meta and meta.description else "") or (req.text or "")  # older extensions sent it as text
        yt = extract_youtube_content(
            yt_id, req.url,
            client_title=req.title or "",
            client_description=client_desc,
            client_channel=(meta.channel if meta and meta.channel else ""),
            client_keywords=(meta.keywords if meta else None),
        )
        parts = yt["parts"]
        return {
            "title": parts["title"],  # empty when unknown: never analyse a placeholder like "YouTube Video (id)"
            "description": parts["description"],
            "metadata_text": parts["metadata_text"],
            "cleaned_text": parts["transcript"],
            "source_type": "youtube",
            "url": req.url,
            "sources": {
                "title": bool(parts["title"]), "description": bool(parts["description"]),
                "metadata": bool(parts["metadata_text"]), "transcript": bool(parts["transcript"]),
                "transcript_status": yt["metadata"].get("transcript_status", ""),
                "body_words": len((" ".join((parts["description"], parts["transcript"]))).split()),
            },
        }
    out = extract_and_normalize(url=req.url, html=req.html, text=req.text, title=req.title)
    if req.title:
        out["title"] = req.title
    if out.get("error") and not out["cleaned_text"] and not req.title:
        raise ContentUnavailable(out["error"])
    out.setdefault("description", "")
    out.setdefault("metadata_text", "")
    out["sources"] = {
        "title": bool(out.get("title")), "description": False, "metadata": False,
        "transcript": False, "body_words": out.get("word_count", 0),
    }
    return out


def _context_vector(engine, db: Session, ctx_row, profile: ContextProfile) -> np.ndarray:
    """Use the cached embedding for dense models; recompute (and cache) when missing or model changed."""
    emb = engine.embedder
    if ctx_row is not None and emb.dense and ctx_row.context_embedding:
        try:
            cached = json.loads(ctx_row.context_embedding)
            if cached.get("model") == emb.name:
                return np.asarray(cached["vector"], dtype=np.float32)
        except (ValueError, KeyError):
            pass
    vec = engine.embed_context(profile)
    if ctx_row is not None and emb.dense:
        ctx_row.context_embedding = json.dumps({"model": emb.name, "vector": [round(float(x), 6) for x in vec]})
        db.commit()
    return vec


def analyze(db: Session, req: AnalyzeRequest) -> AnalyzeResponse:
    engine = get_engine()
    ctx_row = None
    if req.session_id:
        ctx_row = context_service.get_study_context_by_session(db, req.session_id)
        if not ctx_row:
            raise SessionNotFound(req.session_id)
        profile = build_profile(ctx_row.topic, ctx_row.learning_objectives or "", ctx_row.course_material or "")
    else:
        profile = build_profile(req.topic or "", req.learning_objectives or "")

    content = _resolve_content(req)
    ctx_vec = _context_vector(engine, db, ctx_row, profile)
    result: AnalysisResult = engine.analyze(
        profile,
        ContentInput(
            title=content["title"], text=content["cleaned_text"], url=req.url or "",
            description=content["description"], metadata=content["metadata_text"], source_type=content["source_type"],
        ),
        ctx_vec,
    )

    heatmap = [SentenceScore(text=t, score=round(s * 100, 1), level=level_for(s)) for t, s in result.heatmap]
    highlighted = [SentenceScore(text=t, score=round(s * 100, 1), level=level_for(s)) for t, s in result.highlighted]

    analysis_id = None
    if req.persist and req.session_id:
        analysis_id = _persist(db, req, content, result, heatmap)

    return AnalyzeResponse(
        analysis_id=analysis_id,
        url=req.url,
        title=content["title"],
        source_type=content["source_type"],
        relevance_score=result.score,
        classification=result.classification,
        decision=result.decision,
        explanation=result.explanation,
        reasons=result.reasons,
        matching_concepts=result.concepts,
        highlighted_sentences=highlighted,
        heatmap=heatmap,
        signals=result.signals,
        explanation_method=result.method,
        embedding_backend=result.embedding_backend,
        llm_used=result.llm_used,
        low_content=result.low_content,
        insufficient_content=result.insufficient_content,
        word_count=result.word_count,
        evaluation_path=result.evaluation_path,
        llm_status=result.llm_status,
        content_quality=result.content_quality,
        content_sources=content.get("sources", {}),
    )


def _persist(db: Session, req: AnalyzeRequest, content: dict, result: AnalysisResult, heatmap) -> str:
    url = req.url or content.get("url") or "(pasted text)"
    row = ContentAnalysis(
        session_id=req.session_id,
        url=url,
        title=(content["title"] or "")[:255],
        extracted_text=" ".join(p for p in (content["description"], content["cleaned_text"]) if p)[:20_000],
        relevance_score=result.score,
        decision=DecisionEnum(result.decision),
        explanation=result.explanation,
        key_concepts=json.dumps(result.concepts),
        classification=result.classification,
        heatmap=json.dumps([h.model_dump() for h in heatmap]),
        signals=json.dumps(result.signals),
    )
    db.add(row)
    db.add(ActivityEvent(
        session_id=req.session_id,
        event_type=EVENT_FOR_DECISION[result.decision],
        metadata_json=json.dumps({"url": url, "score": result.score}),
    ))
    if result.decision == "BLOCK":
        db.add(BlockedContent(session_id=req.session_id, url=url, title=(content["title"] or "")[:255], reason=result.explanation))
    db.commit()
    db.refresh(row)
    return row.id


def record_override(db: Session, session_id: str, url: str, title: Optional[str]) -> None:
    """The student chose to continue after a block: log it as a distraction attempt."""
    if not context_service.get_study_session(db, session_id):
        raise SessionNotFound(session_id)
    db.add(ActivityEvent(
        session_id=session_id,
        event_type="distraction_attempt",
        metadata_json=json.dumps({"url": url, "title": title, "action": "block_overridden"}),
    ))
    db.commit()


def history(db: Session, session_id: str, limit: int = 50):
    return (
        db.query(ContentAnalysis)
        .filter(ContentAnalysis.session_id == session_id)
        .order_by(ContentAnalysis.analyzed_at.desc())
        .limit(limit)
        .all()
    )


# ---------------------------------------------------------------- token-level Integrated Gradients (optional extra)
_ig_cache = None


def _get_ig_cache():
    global _ig_cache
    if _ig_cache is None:
        from app.ai.integrated_gradients import AttributionCache
        _ig_cache = AttributionCache(256)
    return _ig_cache


def _ig_supported(embedder) -> bool:
    return bool(getattr(embedder, "dense", False)) and getattr(embedder, "st_model", None) is not None


def explain_tokens(db: Session, req: ExplainTokensRequest) -> ExplainTokensResponse:
    """Word-level Integrated Gradients for one sentence. Independent of /analyze: it never changes a score,
    and any failure is reported as ``available=False`` so the core analysis is unaffected."""
    from app.ai import integrated_gradients as ig

    engine = get_engine()
    ctx_row = None
    if req.session_id:
        ctx_row = context_service.get_study_context_by_session(db, req.session_id)
        if not ctx_row:
            raise SessionNotFound(req.session_id)
        profile = build_profile(ctx_row.topic, ctx_row.learning_objectives or "", ctx_row.course_material or "")
    else:
        profile = build_profile(req.topic or "", req.learning_objectives or "")

    meta = dict(
        method=ig.METHOD, target=ig.TARGET_DESCRIPTION, scope=ig.SCOPE_DESCRIPTION, baseline=ig.BASELINE_DESCRIPTION,
        score_semantics=ig.SCORE_SEMANTICS, embedding_backend=engine.embedder.name,
    )

    def unavailable(reason: str, message: str) -> ExplainTokensResponse:
        return ExplainTokensResponse(available=False, reason=reason, message=message, **meta)

    if not _ig_supported(engine.embedder):
        return unavailable(
            "unsupported_backend",
            "Word-level explanations need the sentence-transformers embedding model (not the lexical fallback).",
        )

    steps = req.steps or ig.DEFAULT_STEPS
    key = ig.cache_key(engine.embedder.name, req.sentence, profile.text, steps)
    cache = _get_ig_cache()
    hit = cache.get(key)
    if hit is not None:
        return hit.model_copy(update={"cached": True})

    try:
        ctx_vec = _context_vector(engine, db, ctx_row, profile)  # reuses the cached study-context embedding
        expl = ig.explain(engine.embedder, req.sentence, ctx_vec, steps)
    except ig.AttributionUnavailable as exc:
        return unavailable(exc.reason, exc.message)
    except Exception as exc:  # numerical failure, model error, out of memory...
        logger.warning("token attribution failed: %s", type(exc).__name__)  # never log the text itself
        return unavailable("attribution_failed", "The word-level explanation could not be computed.")

    resp = ExplainTokensResponse(
        available=True, n_tokens=expl.n_tokens, n_tokens_total=expl.n_tokens_total, truncated=expl.truncated, steps=expl.steps,
        target_value=round(expl.f_input, 6), baseline_value=round(expl.f_baseline, 6),
        sum_attributions=round(expl.sum_attributions, 6), completeness_error=round(expl.completeness_error, 6),
        converged=expl.converged,
        words=[
            WordAttributionOut(
                word=w.word, start=w.start, end=w.end, attribution=round(w.attribution, 6),
                normalized=round(w.normalized, 4),
                tokens=[TokenPart(token=t, attribution=round(a, 6)) for t, a in w.tokens],
            )
            for w in expl.words
        ],
        **meta,
    )
    cache.put(key, resp)
    return resp
