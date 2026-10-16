import uuid
from datetime import datetime
from sqlalchemy import Column, String, Text, DateTime, ForeignKey, PrimaryKeyConstraint
from sqlalchemy.orm import relationship
from app.utils.database import Base


class Cohort(Base):
    __tablename__ = "cohorts"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    teacher_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    teacher = relationship("User", back_populates="taught_cohorts")
    memberships = relationship("CohortMembership", back_populates="cohort", cascade="all, delete-orphan")
    study_sessions = relationship("StudySession", back_populates="cohort")


class CohortMembership(Base):
    __tablename__ = "cohort_memberships"

    cohort_id = Column(String(36), ForeignKey("cohorts.id", ondelete="CASCADE"), nullable=False)
    student_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    joined_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        PrimaryKeyConstraint("cohort_id", "student_id"),
    )

    # Relationships
    cohort = relationship("Cohort", back_populates="memberships")
    student = relationship("User", back_populates="cohort_memberships")
