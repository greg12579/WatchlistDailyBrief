"""SQLAlchemy models for brief storage."""

from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Brief(Base):
    """A daily brief that was sent."""

    __tablename__ = "briefs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date_sent = Column(DateTime, nullable=False, default=datetime.utcnow)
    subject = Column(String(255), nullable=False)
    delivery_mode = Column(String(50), nullable=False)  # "email" or "slack"

    # Relationship to items
    items = relationship("BriefItem", back_populates="brief", cascade="all, delete-orphan")


class BriefItem(Base):
    """An individual item (ticker) within a brief."""

    __tablename__ = "brief_items"

    id = Column(Integer, primary_key=True, autoincrement=True)
    brief_id = Column(Integer, ForeignKey("briefs.id"), nullable=False)
    ticker = Column(String(20), nullable=False)
    label = Column(String(20), nullable=False)  # "ACTIONABLE" or "MONITOR"
    rank = Column(Integer, nullable=False)
    score = Column(Float, nullable=False)
    facts_json = Column(Text, nullable=False)  # JSON string of trigger facts
    llm_json = Column(Text, nullable=False)  # JSON string of LLM explanation

    # Relationships
    brief = relationship("Brief", back_populates="items")
    feedback = relationship("Feedback", back_populates="brief_item", cascade="all, delete-orphan")


class Feedback(Base):
    """User feedback on a brief item."""

    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, autoincrement=True)
    brief_item_id = Column(Integer, ForeignKey("brief_items.id"), nullable=False)
    vote = Column(String(10), nullable=True)  # "up" or "down"
    impact = Column(String(10), nullable=True)  # "yes" or "no" (decision impact)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationship
    brief_item = relationship("BriefItem", back_populates="feedback")
