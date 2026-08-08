"""
database.py - Database Models & Session Factory
================================================
Defines the SQLAlchemy ORM table for storing emotion predictions.

ORM = Object-Relational Mapper: lets us work with the database using
Python objects instead of raw SQL strings.  Every row in the
`emotion_results` table becomes an instance of the EmotionResult class.
"""

import logging
from datetime import datetime

from sqlalchemy import (
    Column, Integer, String, Float, Text,
    DateTime, create_engine
)
from sqlalchemy.orm import declarative_base, sessionmaker

from config import DATABASE_URL, LOG_FORMAT, LOG_DATE_FORMAT, LOG_FILE, LOG_LEVEL

# ─────────────────────────────────────────────
#  LOGGING SETUP
# ─────────────────────────────────────────────
logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    datefmt=LOG_DATE_FORMAT,
    handlers=[
        logging.FileHandler(LOG_FILE),   # write to file
        logging.StreamHandler(),         # also print to console
    ]
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
#  SQLALCHEMY BASE
# ─────────────────────────────────────────────
# declarative_base() returns a base class.  Every ORM model must inherit
# from this base so SQLAlchemy can discover and manage the table.
Base = declarative_base()


# ─────────────────────────────────────────────
#  ORM MODEL  →  emotion_results table
# ─────────────────────────────────────────────
class EmotionResult(Base):
    """
    Represents a single emotion-detection result stored in MySQL.

    Each attribute decorated with Column() maps to one column in the table.
    """

    __tablename__ = "emotion_results"   # exact table name in MySQL

    # Auto-incrementing primary key – every row gets a unique integer id
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Name of the uploaded image file (e.g. "selfie.jpg")
    image_name = Column(String(255), nullable=False)

    # The emotion label with the highest confidence score (e.g. "happy")
    predicted_emotion = Column(String(50), nullable=False)

    # Confidence value between 0.0 and 1.0 (e.g. 0.9234 = 92.34 %)
    confidence = Column(Float, nullable=False)

    # JSON string storing all emotion probabilities, e.g.:
    # '{"happy": 0.92, "neutral": 0.05, "sad": 0.02, ...}'
    all_scores = Column(Text, nullable=True)

    # Original image dimensions in pixels
    image_width  = Column(Integer, nullable=True)
    image_height = Column(Integer, nullable=True)

    # How long (in seconds) the full detection pipeline took
    detection_time = Column(Float, nullable=True)

    # Automatically set to the current UTC time when the row is created
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self) -> str:
        """Human-readable summary for debugging."""
        return (
            f"<EmotionResult(id={self.id}, image='{self.image_name}', "
            f"emotion='{self.predicted_emotion}', confidence={self.confidence:.2%})>"
        )

    def to_dict(self) -> dict:
        """Convert the ORM object to a plain Python dictionary."""
        return {
            "id":               self.id,
            "image_name":       self.image_name,
            "predicted_emotion":self.predicted_emotion,
            "confidence":       self.confidence,
            "all_scores":       self.all_scores,
            "image_width":      self.image_width,
            "image_height":     self.image_height,
            "detection_time":   self.detection_time,
            "created_at":       str(self.created_at),
        }


# ─────────────────────────────────────────────
#  ENGINE & SESSION FACTORY
# ─────────────────────────────────────────────
def get_engine(database_url: str = DATABASE_URL):
    """
    Create a SQLAlchemy Engine.

    The engine is the low-level connection pool.  It does NOT open a
    connection immediately – it waits until you actually query the DB.

    Args:
        database_url: SQLAlchemy connection string from config.py

    Returns:
        sqlalchemy.engine.Engine
    """
    try:
        engine = create_engine(
            database_url,
            pool_pre_ping=True,    # test connections before using them
            pool_recycle=3600,     # recycle connections older than 1 hour
            echo=False,            # set True to log every SQL statement
        )
        logger.info("SQLAlchemy engine created successfully.")
        return engine
    except Exception as exc:
        logger.error("Failed to create database engine: %s", exc)
        raise


def get_session_factory(engine):
    """
    Return a session factory bound to the given engine.

    A session is a "unit of work" – open a session, perform queries,
    then commit or rollback.  Think of it like a database transaction.
    """
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal


def init_db(engine) -> None:
    """
    Create all tables that don't yet exist in MySQL.

    Safe to call on every startup – SQLAlchemy skips tables that already
    exist, so it won't wipe your data.
    """
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables initialised (created if missing).")
    except Exception as exc:
        logger.error("Database initialisation failed: %s", exc)
        raise
