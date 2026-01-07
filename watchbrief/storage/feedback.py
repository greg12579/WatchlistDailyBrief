"""Feedback recording functions."""

from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from watchbrief.storage.models import BriefItem, Feedback


def record_vote(
    session: Session,
    brief_id: int,
    ticker: str,
    vote: str,
) -> Optional[Feedback]:
    """Record a thumbs up/down vote for a brief item.

    Args:
        session: Database session
        brief_id: ID of the brief
        ticker: Ticker symbol
        vote: "up" or "down"

    Returns:
        Created Feedback object, or None if item not found
    """
    # Find the brief item
    item = (
        session.query(BriefItem)
        .filter(BriefItem.brief_id == brief_id, BriefItem.ticker == ticker)
        .first()
    )

    if not item:
        return None

    # Check for existing feedback
    existing = (
        session.query(Feedback)
        .filter(Feedback.brief_item_id == item.id)
        .first()
    )

    if existing:
        # Update existing
        existing.vote = vote
        existing.created_at = datetime.utcnow()
        return existing
    else:
        # Create new
        feedback = Feedback(
            brief_item_id=item.id,
            vote=vote,
            created_at=datetime.utcnow(),
        )
        session.add(feedback)
        return feedback


def record_impact(
    session: Session,
    brief_id: int,
    ticker: str,
    impact: str,
) -> Optional[Feedback]:
    """Record decision impact for a brief item.

    Args:
        session: Database session
        brief_id: ID of the brief
        ticker: Ticker symbol
        impact: "yes" or "no"

    Returns:
        Updated Feedback object, or None if item not found
    """
    # Find the brief item
    item = (
        session.query(BriefItem)
        .filter(BriefItem.brief_id == brief_id, BriefItem.ticker == ticker)
        .first()
    )

    if not item:
        return None

    # Check for existing feedback
    existing = (
        session.query(Feedback)
        .filter(Feedback.brief_item_id == item.id)
        .first()
    )

    if existing:
        # Update existing
        existing.impact = impact
        existing.created_at = datetime.utcnow()
        return existing
    else:
        # Create new with just impact
        feedback = Feedback(
            brief_item_id=item.id,
            impact=impact,
            created_at=datetime.utcnow(),
        )
        session.add(feedback)
        return feedback


def get_feedback_stats(session: Session) -> dict:
    """Get aggregate feedback statistics.

    Returns:
        Dictionary with vote and impact counts
    """
    total_items = session.query(BriefItem).count()
    total_feedback = session.query(Feedback).count()

    votes_up = session.query(Feedback).filter(Feedback.vote == "up").count()
    votes_down = session.query(Feedback).filter(Feedback.vote == "down").count()

    impact_yes = session.query(Feedback).filter(Feedback.impact == "yes").count()
    impact_no = session.query(Feedback).filter(Feedback.impact == "no").count()

    return {
        "total_items": total_items,
        "total_feedback": total_feedback,
        "votes_up": votes_up,
        "votes_down": votes_down,
        "impact_yes": impact_yes,
        "impact_no": impact_no,
    }
