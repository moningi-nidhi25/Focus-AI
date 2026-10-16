import os
import sys
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.utils.database import init_db, SessionLocal
from app.models import (
    User,
    UserRole,
    Cohort,
    CohortMembership,
    StudySession,
    SessionStatus,
    StudyContext,
    ContentAnalysis,
    DecisionEnum,
    ActivityEvent,
    FocusMetric,
)


def seed_database():
    print("Initializing database tables...")
    init_db()

    db = SessionLocal()
    try:
        # Check if database is already seeded
        existing_admin = db.query(User).filter_by(email="admin@focusai.edu").first()
        if existing_admin:
            print("Database already contains seed data. Skipping seed step.")
            return

        print("Seeding initial multi-tenant roles and data...")

        # 1. Create Users (ADMIN, TEACHER, STUDENT)
        admin = User(
            email="admin@focusai.edu",
            full_name="System Administrator",
            role=UserRole.ADMIN,
        )
        teacher = User(
            email="prof.smith@focusai.edu",
            full_name="Prof. John Smith",
            role=UserRole.TEACHER,
        )
        student = User(
            email="alice@focusai.edu",
            full_name="Alice Student",
            role=UserRole.STUDENT,
        )

        db.add_all([admin, teacher, student])
        db.commit()

        # Refresh to get IDs
        db.refresh(admin)
        db.refresh(teacher)
        db.refresh(student)

        # 2. Create Cohort
        cohort = Cohort(
            name="CS 401: Operating Systems",
            description="Introduction to OS principles, processes, deadlock prevention, and scheduling.",
            teacher_id=teacher.id,
        )
        db.add(cohort)
        db.commit()
        db.refresh(cohort)

        # 3. Enroll Student in Cohort
        membership = CohortMembership(
            cohort_id=cohort.id,
            student_id=student.id,
        )
        db.add(membership)

        # 4. Create Study Session & Context for Student
        session = StudySession(
            student_id=student.id,
            cohort_id=cohort.id,
            title="Deadlock Prevention and Resource Allocation",
            status=SessionStatus.ACTIVE,
        )
        db.add(session)
        db.commit()
        db.refresh(session)

        context = StudyContext(
            session_id=session.id,
            topic="Operating System Deadlocks",
            learning_objectives="Understand Banker's Algorithm and Coffman conditions for deadlock prevention.",
        )
        db.add(context)

        # 5. Create Sample Content Analysis
        analysis = ContentAnalysis(
            session_id=session.id,
            url="https://en.wikipedia.org/wiki/Banker%27s_algorithm",
            title="Banker's Algorithm - Wikipedia",
            extracted_text="The Banker's algorithm is a resource allocation and deadlock avoidance algorithm...",
            relevance_score=94.0,
            decision=DecisionEnum.ALLOW,
            explanation="Content directly explains deadlock prevention algorithms matching the active topic.",
            key_concepts="deadlock prevention, resource allocation, Banker's algorithm",
        )
        db.add(analysis)

        # 6. Create Initial Activity Event & Focus Metric
        event = ActivityEvent(
            session_id=session.id,
            event_type="session_start",
            metadata_json='{"source": "student_dashboard"}',
        )
        db.add(event)

        metric = FocusMetric(
            session_id=session.id,
            focus_score=92.5,
            productivity_score=88.0,
            dwell_time_seconds=1200,
            tab_switch_count=3,
            distraction_count=0,
        )
        db.add(metric)

        db.commit()
        print("Database successfully seeded!")
        print(f"- Created Admin: {admin.email}")
        print(f"- Created Teacher: {teacher.email}")
        print(f"- Created Student: {student.email}")
        print(f"- Created Cohort: {cohort.name}")
        print(f"- Created Study Session: {session.title}")

    except Exception as e:
        db.rollback()
        print(f"Error seeding database: {e}")
        raise e
    finally:
        db.close()


if __name__ == "__main__":
    seed_database()
