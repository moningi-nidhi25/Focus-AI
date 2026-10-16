"""Demo teacher + 6 students with SYNTHETIC activity:  uv run python scripts/seed_demo_dashboard.py"""
import getpass
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from app.ml.synthetic import simulate_session
from app.models import (Cohort, CohortMembership, FocusMetric, SessionStatus, StudyContext,
                        StudySession, User, UserRole)
from app.utils.database import SessionLocal, init_db
from app.utils.security import hash_password

init_db()
password = getpass.getpass("One password for ALL demo accounts (at least 8 characters, hidden): ")
if len(password) < 8:
    sys.exit("Password is too short.")

db = SessionLocal()
if db.query(User).filter(User.email == "teacher@demo.focusai.dev").first():
    sys.exit("Demo data already exists.")

rng = np.random.default_rng(7)
pw_hash = hash_password(password)
now = datetime.utcnow()

teacher = User(email="teacher@demo.focusai.dev", full_name="Demo Teacher", role=UserRole.TEACHER, password_hash=pw_hash)
db.add(teacher)
db.flush()
cohort = Cohort(name="Operating Systems - Batch A", description="Synthetic demo cohort", teacher_id=teacher.id)
db.add(cohort)
db.flush()

for i in range(1, 7):
    student = User(email=f"student{i}@demo.focusai.dev", full_name=f"Demo Student {i}",
                   role=UserRole.STUDENT, password_hash=pw_hash)
    db.add(student)
    db.flush()
    db.add(CohortMembership(cohort_id=cohort.id, student_id=student.id))
    bias = float(rng.normal(0, 8))
    days_used = sorted(rng.choice(14, size=int(rng.integers(5, 9)), replace=False), reverse=True)
    for k, off in enumerate(days_used):
        rows = simulate_session(rng)[:40]
        start = now - timedelta(days=int(off), hours=int(rng.integers(1, 9)))
        n = len(rows)
        sess = StudySession(student_id=student.id, title=f"OS study session {k + 1}", status=SessionStatus.COMPLETED,
                            start_time=start, end_time=start + timedelta(minutes=n), total_duration_seconds=n * 60)
        db.add(sess)
        db.flush()
        db.add(StudyContext(session_id=sess.id, topic="Operating Systems", keywords=json.dumps(["operating systems"])))
        for j, r in enumerate(rows):
            db.add(FocusMetric(session_id=sess.id,
                               focus_score=float(np.clip(r[0] + bias, 0, 100)),
                               productivity_score=float(np.clip(r[1] + bias, 0, 100)),
                               dwell_time_seconds=int(r[2]), tab_switch_count=int(r[3]),
                               distraction_count=int(r[4]),
                               calculated_at=start + timedelta(minutes=j + 1)))
db.commit()
print("Done: teacher@demo.focusai.dev and student1..6@demo.focusai.dev created.")