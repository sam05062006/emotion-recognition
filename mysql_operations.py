"""
mysql_operations.py - DatabaseManager Class
============================================
All MySQL interactions are encapsulated here.  The rest of the application
never writes raw SQL — it calls methods on this class instead.

WHY OOP HERE?
  Wrapping DB logic in a class means:
  - One place to change connection settings.
  - Easy to mock/test without a real database.
  - Methods are self-documenting (save_result, get_history, delete_result…).
"""

import json
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, date

import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import desc, and_

from config import HISTORY_LIMIT
from database import EmotionResult, get_engine, get_session_factory, init_db, DATABASE_URL

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages all Create / Read / Update / Delete operations on MySQL.

    Usage
    -----
    db = DatabaseManager()
    db.connect()
    record_id = db.save_result(result_data)
    history   = db.get_history()
    db.close()
    """

    def __init__(self, database_url: str = DATABASE_URL):
        self.database_url = database_url
        self.engine       = None
        self._SessionLocal = None   # session factory (callable)
        logger.info("DatabaseManager created (not yet connected).")

    # ──────────────────────────────────────────
    #  CONNECTION LIFECYCLE
    # ──────────────────────────────────────────
    def connect(self) -> bool:
        """
        Open the connection pool and initialise the schema.

        Returns True on success, False on failure (so the app can gracefully
        degrade if MySQL is unavailable).
        """
        try:
            self.engine        = get_engine(self.database_url)
            self._SessionLocal = get_session_factory(self.engine)
            init_db(self.engine)          # CREATE TABLE IF NOT EXISTS …
            logger.info("Connected to MySQL and schema ready.")
            return True
        except Exception as exc:
            logger.error("Database connection failed: %s", exc)
            return False

    def close(self) -> None:
        """Dispose the connection pool (call on app shutdown)."""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connection pool closed.")

    def _get_session(self) -> Session:
        """Return a new session.  Caller is responsible for closing it."""
        if self._SessionLocal is None:
            raise RuntimeError("DatabaseManager.connect() must be called first.")
        return self._SessionLocal()

    # ──────────────────────────────────────────
    #  CREATE
    # ──────────────────────────────────────────
    def save_result(self, result_data: Dict[str, Any]) -> Optional[int]:
        """
        Insert one emotion-detection result into the database.

        Args:
            result_data: dictionary produced by EmotionDetector.detect()
                Keys: image_name, predicted_emotion, confidence,
                      all_scores (dict), image_width, image_height,
                      detection_time

        Returns:
            The auto-generated integer ID of the new row, or None on error.
        """
        session = self._get_session()
        try:
            # Convert the all_scores dict to a JSON string for storage
            all_scores_json = json.dumps(result_data.get("all_scores", {}))

            record = EmotionResult(
                image_name       = result_data["image_name"],
                predicted_emotion= result_data["predicted_emotion"],
                confidence       = float(result_data["confidence"]),
                all_scores       = all_scores_json,
                image_width      = result_data.get("image_width"),
                image_height     = result_data.get("image_height"),
                detection_time   = result_data.get("detection_time"),
            )

            session.add(record)    # stage the INSERT
            session.commit()       # send to MySQL
            session.refresh(record)  # reload to get the auto-generated id

            logger.info("Saved result id=%s for image '%s'.",
                        record.id, record.image_name)
            return record.id

        except Exception as exc:
            session.rollback()     # undo if anything went wrong
            logger.error("save_result failed: %s", exc)
            return None
        finally:
            session.close()        # always release the session

    # ──────────────────────────────────────────
    #  READ
    # ──────────────────────────────────────────
    def get_history(self, limit: int = HISTORY_LIMIT) -> List[Dict]:
        """
        Fetch the most recent `limit` predictions, newest first.

        Returns a list of plain dicts so the caller doesn't need to
        import SQLAlchemy ORM models.
        """
        session = self._get_session()
        try:
            rows = (
                session.query(EmotionResult)
                .order_by(desc(EmotionResult.created_at))
                .limit(limit)
                .all()
            )
            return [r.to_dict() for r in rows]
        except Exception as exc:
            logger.error("get_history failed: %s", exc)
            return []
        finally:
            session.close()

    def get_by_emotion(self, emotion: str) -> List[Dict]:
        """Return all predictions where predicted_emotion == emotion."""
        session = self._get_session()
        try:
            rows = (
                session.query(EmotionResult)
                .filter(EmotionResult.predicted_emotion == emotion.lower())
                .order_by(desc(EmotionResult.created_at))
                .all()
            )
            return [r.to_dict() for r in rows]
        except Exception as exc:
            logger.error("get_by_emotion failed: %s", exc)
            return []
        finally:
            session.close()

    def get_by_date_range(
        self, start_date: date, end_date: date
    ) -> List[Dict]:
        """Return predictions whose created_at falls within the date range."""
        session = self._get_session()
        try:
            start_dt = datetime.combine(start_date, datetime.min.time())
            end_dt   = datetime.combine(end_date,   datetime.max.time())
            rows = (
                session.query(EmotionResult)
                .filter(and_(
                    EmotionResult.created_at >= start_dt,
                    EmotionResult.created_at <= end_dt,
                ))
                .order_by(desc(EmotionResult.created_at))
                .all()
            )
            return [r.to_dict() for r in rows]
        except Exception as exc:
            logger.error("get_by_date_range failed: %s", exc)
            return []
        finally:
            session.close()

    def get_by_id(self, record_id: int) -> Optional[Dict]:
        """Fetch a single record by primary key."""
        session = self._get_session()
        try:
            row = session.query(EmotionResult).filter_by(id=record_id).first()
            return row.to_dict() if row else None
        except Exception as exc:
            logger.error("get_by_id failed: %s", exc)
            return None
        finally:
            session.close()

    def get_total_count(self) -> int:
        """Return the total number of stored predictions."""
        session = self._get_session()
        try:
            return session.query(EmotionResult).count()
        except Exception as exc:
            logger.error("get_total_count failed: %s", exc)
            return 0
        finally:
            session.close()

    def get_emotion_stats(self) -> Dict[str, int]:
        """
        Return a dict mapping each emotion label to how many times it was
        the top prediction.  Useful for drawing a summary bar chart.
        """
        session = self._get_session()
        try:
            from sqlalchemy import func
            rows = (
                session.query(
                    EmotionResult.predicted_emotion,
                    func.count(EmotionResult.id).label("count")
                )
                .group_by(EmotionResult.predicted_emotion)
                .all()
            )
            return {r.predicted_emotion: r.count for r in rows}
        except Exception as exc:
            logger.error("get_emotion_stats failed: %s", exc)
            return {}
        finally:
            session.close()

    # ──────────────────────────────────────────
    #  DELETE
    # ──────────────────────────────────────────
    def delete_result(self, record_id: int) -> bool:
        """
        Delete a single prediction record by ID.

        Returns True if the row existed and was deleted, False otherwise.
        """
        session = self._get_session()
        try:
            row = session.query(EmotionResult).filter_by(id=record_id).first()
            if row is None:
                logger.warning("delete_result: id=%s not found.", record_id)
                return False
            session.delete(row)
            session.commit()
            logger.info("Deleted record id=%s.", record_id)
            return True
        except Exception as exc:
            session.rollback()
            logger.error("delete_result failed: %s", exc)
            return False
        finally:
            session.close()

    def delete_all(self) -> int:
        """
        DANGER: Delete every row in emotion_results.
        Returns the number of rows deleted.
        """
        session = self._get_session()
        try:
            count = session.query(EmotionResult).delete()
            session.commit()
            logger.warning("Deleted ALL %d records from emotion_results.", count)
            return count
        except Exception as exc:
            session.rollback()
            logger.error("delete_all failed: %s", exc)
            return 0
        finally:
            session.close()

    # ──────────────────────────────────────────
    #  EXPORT
    # ──────────────────────────────────────────
    def export_to_dataframe(self, records: Optional[List[Dict]] = None) -> pd.DataFrame:
        """
        Convert a list of record dicts to a Pandas DataFrame.

        If no list is provided, exports the full history.
        Useful for CSV download or further analysis.
        """
        if records is None:
            records = self.get_history(limit=10_000)  # all rows

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)

        # Parse the all_scores JSON string back to a dict for readability
        if "all_scores" in df.columns:
            df["all_scores"] = df["all_scores"].apply(
                lambda x: json.loads(x) if isinstance(x, str) else x
            )

        # Make confidence a percentage string for human-friendly export
        if "confidence" in df.columns:
            df["confidence_pct"] = df["confidence"].apply(
                lambda x: f"{x:.1%}" if x is not None else ""
            )

        return df
