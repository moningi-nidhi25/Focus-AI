"""Dashboard analytics (Phase 9).

Student view: the student's own data.
Teacher view: cohort AGGREGATES only (no names, ids or browsing history), hidden for cohorts
              smaller than MIN_COHORT_SIZE so no individual can be singled out.
Admin view:   platform totals only.

Study time = sum of visible page time (dwell) recorded in focus_metrics snapshots.
"""
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from app.models import (Cohort, CohortMembership, ContentAnalysis, FocusMetric, FocusPrediction,
                        StudySession, User, UserRole)

MIN_COHORT_SIZE = 3
MODEL_FILE = Path(__file__).resolve().parent.parent / "ml" / "artifacts" / "focus_lstm.pt"


def _avg(values) -> Optional[float]:
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 1) if values else None


def _rows(db: Session, since: datetime, student_id: Optional[str] = None, cohort_id: Optional[str] = None):
    q = (db.query(
            FocusMetric.calculated_at.label("ts"),
            FocusMetric.focus_score.label("focus"),
            FocusMetric.productivity_score.label("prod"),
            FocusMetric.dwell_time_seconds.label("dwell"),
            FocusMetric.tab_switch_count.label("tabs"),
            FocusMetric.distraction_count.label("dist"),
            StudySession.student_id.label("student_id"),
            StudySession.id.label("session_id"))
         .join(StudySession, StudySession.id == FocusMetric.session_id)
         .filter(FocusMetric.calculated_at >= since))
    if student_id:
        q = q.filter(StudySession.student_id == student_id)
    if cohort_id:
        members = select(CohortMembership.student_id).where(CohortMembership.cohort_id == cohort_id)
        q = q.filter(StudySession.student_id.in_(members))
    return q.all()


def _summary(rows) -> Dict[str, Any]:
    return {
        "students": len({r.student_id for r in rows}),
        "snapshots": len(rows),
        "avg_focus": _avg([r.focus for r in rows]),
        "avg_productivity": _avg([r.prod for r in rows]),
        "study_minutes": round(sum((r.dwell or 0) for r in rows) / 60, 1),
        "distractions": sum((r.dist or 0) for r in rows),
        "tab_switches": sum((r.tabs or 0) for r in rows),
    }


def _trend(rows, days: int, now: datetime) -> List[Dict[str, Any]]:
    by_day = defaultdict(list)
    for r in rows:
        by_day[r.ts.date()].append(r)
    out = []
    for i in range(days - 1, -1, -1):
        d = (now - timedelta(days=i)).date()
        rs = by_day.get(d, [])
        out.append({"date": d.isoformat(), "avg_focus": _avg([r.focus for r in rs]),
                    "study_minutes": round(sum((r.dwell or 0) for r in rs) / 60, 1),
                    "distractions": sum((r.dist or 0) for r in rs)})
    return out


def _week_change(rows, now: datetime) -> Optional[float]:
    cut = now - timedelta(days=7)
    this_week = [r.focus for r in rows if r.ts >= cut]
    before = [r.focus for r in rows if r.ts < cut]
    if this_week and before:
        return round(sum(this_week) / len(this_week) - sum(before) / len(before), 1)
    return None


def _decision(a: ContentAnalysis) -> str:
    return a.decision.value if hasattr(a.decision, "value") else str(a.decision)


def student_overview(db: Session, user: User, days: int) -> Dict[str, Any]:
    now = datetime.utcnow()
    since = now - timedelta(days=days)
    rows = _rows(db, since, student_id=user.id)
    summary, change = _summary(rows), _week_change(rows, now)

    focus_by_session = defaultdict(list)
    for r in rows:
        focus_by_session[r.session_id].append(r.focus)
    sessions = (db.query(StudySession).filter(StudySession.student_id == user.id)
                .order_by(StudySession.start_time.desc()).limit(8).all())

    analyses = (db.query(ContentAnalysis).join(StudySession, StudySession.id == ContentAnalysis.session_id)
                .filter(StudySession.student_id == user.id, ContentAnalysis.analyzed_at >= since)
                .order_by(ContentAnalysis.analyzed_at.desc()).limit(500).all())

    pred = (db.query(FocusPrediction).join(StudySession, StudySession.id == FocusPrediction.session_id)
            .filter(StudySession.student_id == user.id).order_by(FocusPrediction.created_at.desc()).first())
    curve = None
    if pred and pred.curve_data:
        try:
            curve = json.loads(pred.curve_data)
        except ValueError:
            curve = None

    insights: List[str] = []
    if summary["avg_focus"] is None:
        insights.append("No tracked activity yet. Start a study session and browse with the FocusAI extension on.")
    else:
        level = "strong" if summary["avg_focus"] >= 70 else "moderate" if summary["avg_focus"] >= 50 else "low"
        insights.append(f"Your average focus over the last {days} days is {summary['avg_focus']} ({level}).")
        if change is not None:
            insights.append(f"Compared with the week before, your focus is {'up' if change >= 0 else 'down'} "
                            f"{abs(change)} points this week.")
        if summary["study_minutes"] >= 5 and summary["distractions"] / (summary["study_minutes"] / 60) > 6:
            insights.append("You hit more than 6 distractions per study hour. Try closing unrelated tabs before you start.")

    return {
        "days": days,
        "summary": summary,
        "blocked_count": sum(1 for a in analyses if _decision(a) == "BLOCK"),
        "focus_change_vs_previous_week": change,
        "trend": _trend(rows, days, now),
        "recent_sessions": [{"id": s.id, "title": s.title,
                             "status": getattr(s.status, "value", str(s.status)),
                             "start_time": s.start_time, "avg_focus": _avg(focus_by_session.get(s.id, []))}
                            for s in sessions],
        "relevance_history": [{"title": a.title, "url": a.url, "relevance_score": a.relevance_score,
                               "decision": _decision(a), "analyzed_at": a.analyzed_at} for a in analyses[:10]],
        "focus_decay": ({"created_at": pred.created_at, "curve": curve} if curve else None),
        "insights": insights,
    }


def cohort_analytics(db: Session, cohort: Cohort, days: int) -> Dict[str, Any]:
    now = datetime.utcnow()
    size = db.query(func.count()).select_from(CohortMembership).filter(CohortMembership.cohort_id == cohort.id).scalar() or 0
    base = {"cohort_id": cohort.id, "name": cohort.name, "description": cohort.description,
            "student_count": size, "days": days}
    if size < MIN_COHORT_SIZE:
        return {**base, "suppressed": True,
                "message": f"Analytics stay hidden until a cohort has at least {MIN_COHORT_SIZE} students, "
                           "to protect individual privacy."}

    rows = _rows(db, now - timedelta(days=days), cohort_id=cohort.id)
    per_student = defaultdict(list)
    for r in rows:
        per_student[r.student_id].append(r.focus)
    avgs = [sum(v) / len(v) for v in per_student.values()]
    buckets = [("0-40", 0, 40), ("40-60", 40, 60), ("60-80", 60, 80), ("80-100", 80, 101)]
    summary = _summary(rows)
    return {
        **base,
        "suppressed": False,
        "active_students_7d": len({r.student_id for r in rows if r.ts >= now - timedelta(days=7)}),
        "summary": summary,
        "avg_study_minutes_per_student": round(summary["study_minutes"] / len(per_student), 1) if per_student else None,
        "focus_change_vs_previous_week": _week_change(rows, now),
        "trend": _trend(rows, days, now),
        "focus_distribution": [{"range": n, "students": sum(1 for a in avgs if lo <= a < hi)} for n, lo, hi in buckets],
    }


def admin_overview(db: Session, days: int) -> Dict[str, Any]:
    now = datetime.utcnow()
    rows = _rows(db, now - timedelta(days=days))

    def count_role(role: UserRole) -> int:
        return db.query(func.count(User.id)).filter(User.role == role).scalar() or 0

    db_ok = True
    try:
        db.execute(text("select 1"))
    except Exception:
        db_ok = False
    total_dwell = db.query(func.coalesce(func.sum(FocusMetric.dwell_time_seconds), 0)).scalar() or 0
    return {
        "days": days,
        "total_students": count_role(UserRole.STUDENT),
        "total_teachers": count_role(UserRole.TEACHER),
        "active_teachers": db.query(func.count(func.distinct(Cohort.teacher_id))).scalar() or 0,
        "total_cohorts": db.query(func.count(Cohort.id)).scalar() or 0,
        "total_sessions": db.query(func.count(StudySession.id)).scalar() or 0,
        "active_students_7d": len({r.student_id for r in rows if r.ts >= now - timedelta(days=7)}),
        "platform_study_hours_all_time": round(total_dwell / 3600, 1),
        "summary": _summary(rows),
        "trend": _trend(rows, days, now),
        "system": {"database": "ok" if db_ok else "error", "focus_model_trained": MODEL_FILE.exists()},
    }