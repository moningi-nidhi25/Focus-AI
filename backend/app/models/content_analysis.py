import uuid
from enum import Enum
from datetime import datetime
from sqlalchemy import Column, String, Text, Float, DateTime, ForeignKey, Enum as SQLEnum
from sqlalchemy.orm import relationship
from app.utils.database import Base


class DecisionEnum(str, Enum):
    ALLOW = "ALLOW"
    WARN = "WARN"
    BLOCK = "BLOCK"


class ContentAnalysis(Base):
    __tablename__ = "content_analysis"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(36), ForeignKey("study_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    url = Column(Text, nullable=False)
    title = Column(String(255), nullable=True)
    extracted_text = Column(Text, nullable=True)
    relevance_score = Column(Float, nullable=False)
    decision = Column(SQLEnum(DecisionEnum), nullable=False)
    explanation = Column(Text, nullable=True)
    key_concepts = Column(Text, nullable=True)  # JSON list of matching concepts
    classification = Column(String(30), nullable=True)  # RELEVANT | PARTIALLY_RELEVANT | DISTRACTING
    heatmap = Column(Text, nullable=True)  # JSON list of {text, score, level} per sentence (Phase 6)
    signals = Column(Text, nullable=True)  # JSON of the individual relevance signals
    analyzed_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    session = relationship("StudySession", back_populates="content_analyses")
