"""Database setup and session management."""

from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from watchbrief.storage.models import Base


_engine = None
_SessionLocal = None


def init_db(sqlite_path: str) -> None:
    """Initialize the database engine and create tables.

    Args:
        sqlite_path: Path to SQLite database file
    """
    global _engine, _SessionLocal

    # Ensure directory exists
    db_path = Path(sqlite_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Create engine
    _engine = create_engine(
        f"sqlite:///{sqlite_path}",
        connect_args={"check_same_thread": False},
    )

    # Create session factory
    _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)

    # Create tables
    Base.metadata.create_all(bind=_engine)


def get_engine():
    """Get the database engine."""
    if _engine is None:
        raise RuntimeError("Database not initialized. Call init_db first.")
    return _engine


def get_session() -> Session:
    """Create a new database session."""
    if _SessionLocal is None:
        raise RuntimeError("Database not initialized. Call init_db first.")
    return _SessionLocal()


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """Provide a transactional scope around a series of operations.

    Usage:
        with session_scope() as session:
            session.add(obj)
            # commit happens automatically on exit
    """
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
