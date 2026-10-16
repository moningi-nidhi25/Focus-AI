"""Focus Score and Productivity Score (Phase 7).

All inputs are normalised to 0..1 and combined with fixed, documented weights.

relevance_ratio  = relevant dwell seconds / total dwell seconds
                   (ALLOW dwell counts 1.0, WARN dwell counts 0.5, BLOCK dwell counts 0).
                   If no dwell data exists, falls back to the share of ALLOW/WARN analyses.
distraction_rate = (blocked pages + distraction attempts) per 10 minutes
tab_switch_rate  = tab switches per 10 minutes

FOCUS SCORE (0-100) = 100 * ( 0.50 * relevance_ratio
                            + 0.25 * (1 - min(1, distraction_rate / 3))
                            + 0.25 * (1 - min(1, tab_switch_rate / 10)) )

PRODUCTIVITY SCORE (0-100) = 100 * ( 0.60 * relevant_seconds / max(duration, dwell)
                                   + 0.40 * min(1, relevant_minutes / min(25, duration_minutes)) )
  (25 minutes is one Pomodoro block.)
"""
import json
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from app.models import ActivityEvent, ContentAnalysis, FocusMetric, StudySession

SNAPSHOT_INTERVAL = timedelta(seconds=60)
IDLE_CAP = timedelta(minutes=5)
WARN_WEIGHT = 0.5
POMODORO_MIN = 25.0
MAX_DWELL_PER_EVENT = 600.0
MAX_DISTRACTIONS_PER_10MIN = 3.0
MAX_TAB_SWITCHES_PER_10MIN = 10.0


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _meta(ev: ActivityEvent) -> Dict[str, Any]:
    try:
        data = json.loads(ev.metadata_json) if ev.metadata_json else {}
        return data if isinstance(data, dict) else {}
    except ValueError:
        return {}


def compute(db: Session, session: StudySession, since: Optional[datetime] = None,
            until: Optional[datetime] = None) -> Dict[str, Any]:
    until = until or datetime.utcnow()
    start = since or session.start_time or until

    ev_q = db.query(ActivityEvent).filter(ActivityEvent.session_id == session.id, ActivityEvent.timestamp <= until)
    an_q = db.query(ContentAnalysis).filter(ContentAnalysis.session_id == session.id,
                                            ContentAnalysis.analyzed_at <= until)
    if since:
        ev_q = ev_q.filter(ActivityEvent.timestamp > since)
        an_q = an_q.filter(ContentAnalysis.analyzed_at > since)
    events, analyses = ev_q.all(), an_q.all()

    counts: Dict[str, int] = {}
    total_dwell = relevant = 0.0
    for ev in events:
        counts[ev.event_type] = counts.get(ev.event_type, 0) + 1
        if ev.event_type == "dwell_time":
            m = _meta(ev)
            secs = min(max(_num(m.get("seconds")), 0.0), MAX_DWELL_PER_EVENT)
            total_dwell += secs
            d = str(m.get("decision", "")).upper()
            relevant += secs * (1.0 if d == "ALLOW" else WARN_WEIGHT if d == "WARN" else 0.0)

    decisions = [a.decision.value if hasattr(a.decision, "value") else str(a.decision) for a in analyses]
    allowed, warned, blocked = decisions.count("ALLOW"), decisions.count("WARN"), decisions.count("BLOCK")
    distractions = counts.get("distraction_attempt", 0) + blocked
    tabs = counts.get("tab_switch", 0)

    last_ts = max((e.timestamp for e in events), default=None)
    if session.end_time and session.end_time <= until:
        end = session.end_time
    elif last_ts:
        end = min(until, last_ts + IDLE_CAP)
    else:
        end = start
    duration = max(0.0, (end - start).total_seconds(), total_dwell)
    minutes = max(duration / 60.0, 1.0)
    per10 = 10.0 / minutes

    if total_dwell > 0:
        rel_ratio = relevant / total_dwell
    elif analyses:
        rel_ratio = (allowed + WARN_WEIGHT * warned) / len(analyses)
    else:
        rel_ratio = 0.0

    focus = 100.0 * (
        0.50 * rel_ratio
        + 0.25 * (1 - min(1.0, distractions * per10 / MAX_DISTRACTIONS_PER_10MIN))
        + 0.25 * (1 - min(1.0, tabs * per10 / MAX_TAB_SWITCHES_PER_10MIN))
    )
    share = relevant / max(duration, total_dwell) if total_dwell > 0 else 0.0
    target_min = min(POMODORO_MIN, minutes)
    productivity = 100.0 * (0.60 * share + 0.40 * min(1.0, (relevant / 60.0) / target_min))

    insights = []
    if not events and not analyses:
        insights.append("No activity recorded yet. Browse with the FocusAI extension on to start tracking.")
    else:
        if rel_ratio < 0.5:
            insights.append("Less than half of your time was on relevant content. Try closing unrelated tabs.")
        elif rel_ratio >= 0.8:
            insights.append("Great job: most of your time was spent on relevant content.")
        if distractions * per10 > MAX_DISTRACTIONS_PER_10MIN:
            insights.append("You hit many distractions. Consider a short break, then restart the session.")
        if tabs * per10 > MAX_TAB_SWITCHES_PER_10MIN:
            insights.append("Frequent tab switching is hurting your focus. Try staying on one tab for 10 minutes.")
        if not insights:
            insights.append("Steady session. Keep going.")

    return {
        "study_duration_seconds": int(duration),
        "total_dwell_seconds": int(total_dwell),
        "relevant_seconds": int(relevant),
        "allowed_count": allowed,
        "warning_count": warned,
        "blocked_count": blocked,
        "distraction_count": distractions,
        "tab_switch_count": tabs,
        "click_count": counts.get("click", 0),
        "scroll_count": counts.get("scroll", 0),
        "focus_score": round(focus, 1),
        "productivity_score": round(productivity, 1),
        "insights": insights,
        "event_total": len(events) + len(analyses),
    }


def snapshot(db: Session, session: StudySession, force: bool = False) -> Optional[FocusMetric]:
    """Store one windowed FocusMetric row (the time series the LSTM will train on)."""
    now = datetime.utcnow()
    last = (db.query(FocusMetric).filter(FocusMetric.session_id == session.id)
            .order_by(FocusMetric.calculated_at.desc()).first())
    if not force and last and now - last.calculated_at < SNAPSHOT_INTERVAL:
        return None
    since = last.calculated_at if last else None
    m = compute(db, session, since, now)
    if m["event_total"] == 0 and not force:
        return None
    row = FocusMetric(
        session_id=session.id,
        focus_score=m["focus_score"],
        productivity_score=m["productivity_score"],
        dwell_time_seconds=m["total_dwell_seconds"],
        tab_switch_count=m["tab_switch_count"],
        distraction_count=m["distraction_count"],
        calculated_at=now,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row