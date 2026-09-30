import logging
import re
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)

# Sort keys are (rank, microseconds-since-epoch). Rank 0 is for timestamps
# that cannot be turned into an instant, rank 1 for real ones.
#
# This is a behaviour change against 94c3fe9f, where the raw string
# `ORDER BY created_at ASC` put a NULL first but a non-NULL unparseable value
# at its lexicographic position, i.e. after the real timestamps, because
# letters sort after digits. Both now share rank 0 and sort ahead of real
# history. That is deliberate -- where malformed timestamps belong is still
# an open question upstream -- so keep it stated rather than silent.
_UNPARSEABLE_RANK = 0
_INSTANT_RANK = 1

_OFFSET_COLON = re.compile(r"(\d{2}:\d{2}(?::\d{2})?)([+-])(\d{2})(\d{2})$")
_OFFSET_HOUR = re.compile(r"(\d{2}:\d{2}(?::\d{2})?)([+-])(\d{2})$")
_BASIC_DATETIME = re.compile(r"^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})(.*)$")
_BASIC_DATE = re.compile(r"^(\d{4})(\d{2})(\d{2})(.*)$")


def _expand_offset(text: str) -> str:
    """Write a UTC offset as ``+HH:MM``, whatever width it arrived in.

    The offset has to be anchored to a clock reading, otherwise the ``-01``
    in a date like ``2026-01-01`` looks like a ``-01`` offset.
    """
    text = _OFFSET_COLON.sub(r"\1\2\3:\4", text)
    return _OFFSET_HOUR.sub(r"\1\2\3:00", text)


def _expand_basic_format(text: str) -> str:
    """Write an ISO-8601 *basic* (unseparated) date or datetime in extended form.

    A trailing UTC offset is carried over untouched; ``_expand_offset`` runs
    afterwards and fixes its width.
    """
    match = _BASIC_DATETIME.match(text)
    if match:
        year, month, day, hour, minute, second, rest = match.groups()
        return f"{year}-{month}-{day}T{hour}:{minute}:{second}{rest}"
    match = _BASIC_DATE.match(text)
    if match:
        year, month, day, rest = match.groups()
        return f"{year}-{month}-{day}{rest}"
    return text


def _normalise_timestamp(value: str) -> str:
    """Rewrite an ISO-8601 string into a form every supported Python accepts.

    ``datetime.fromisoformat`` only learned to read ``Z``, short offsets and
    the *basic* (unseparated) formats in 3.11, and this project supports 3.10.
    SQLite's date functions accept all of them, so without this the same
    database would order the same rows differently on 3.10 than on 3.11/3.12.
    """
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    head, dot, tail = text.partition(".")
    head = _expand_offset(_expand_basic_format(head))
    if not dot:
        return head
    digits = ""
    for char in tail:
        if not char.isdigit():
            break
        digits += char
    return head + "." + (digits + "000000")[:6] + _expand_offset(tail[len(digits) :])


def _instant_key(value: Optional[str]) -> tuple:
    """Build an exact, offset-aware sort key for an ISO-8601 timestamp.

    Ordering in SQL cannot be both exact and portable here. SQLite's
    ``JULIANDAY()`` keeps only three fractional digits, so two records inside
    the same millisecond collapse onto one key, and the double it returns
    costs another ~40 microseconds on top at present-day magnitudes. Neither
    SQL workaround is usable either:

    * ``strftime(..., 'utc')`` keeps full seconds but reinterprets a naive
      value as local time, so a naive ``09:00`` and an explicit ``01:00+00:00``
      stop comparing as the same moment;
    * ``CAST(substr(ts, 20) AS REAL)`` reads the offset instead of the
      fraction -- it returns ``8.0`` for a ``+08:00`` suffix.

    So the ordering is computed here on integer microseconds. Naive values are
    read as UTC, which is what ``DATETIME()`` already did, so existing rows
    keep their relative order.
    """
    if not value:
        return (_UNPARSEABLE_RANK, 0)
    try:
        moment = datetime.fromisoformat(_normalise_timestamp(value))
    except (AttributeError, TypeError, ValueError):
        logger.debug("history: unparseable timestamp %r, sorted ahead of real history", value)
        return (_UNPARSEABLE_RANK, 0)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    delta = moment - _EPOCH
    micros = (delta.days * 86_400_000_000) + (delta.seconds * 1_000_000) + delta.microseconds
    return (_INSTANT_RANK, micros)


class SQLiteManager:
    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self._lock = threading.Lock()
        self._migrate_history_table()
        self._create_history_table()
        self._create_messages_table()

    def _migrate_history_table(self) -> None:
        """
        If a pre-existing history table had the old group-chat columns,
        rename it, create the new schema, copy the intersecting data, then
        drop the old table.
        """
        with self._lock:
            try:
                # Start a transaction
                self.connection.execute("BEGIN")
                cur = self.connection.cursor()

                cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='history'")
                if cur.fetchone() is None:
                    self.connection.execute("COMMIT")
                    return  # nothing to migrate

                cur.execute("PRAGMA table_info(history)")
                old_cols = {row[1] for row in cur.fetchall()}

                expected_cols = {
                    "id",
                    "memory_id",
                    "old_memory",
                    "new_memory",
                    "event",
                    "created_at",
                    "updated_at",
                    "is_deleted",
                    "actor_id",
                    "role",
                }

                if old_cols == expected_cols:
                    self.connection.execute("COMMIT")
                    return

                logger.info("Migrating history table to new schema (no convo columns).")

                # Clean up any existing history_old table from previous failed migration
                cur.execute("DROP TABLE IF EXISTS history_old")

                # Rename the current history table
                cur.execute("ALTER TABLE history RENAME TO history_old")

                # Create the new history table with updated schema
                cur.execute(
                    """
                    CREATE TABLE history (
                        id           TEXT PRIMARY KEY,
                        memory_id    TEXT,
                        old_memory   TEXT,
                        new_memory   TEXT,
                        event        TEXT,
                        created_at   DATETIME,
                        updated_at   DATETIME,
                        is_deleted   INTEGER,
                        actor_id     TEXT,
                        role         TEXT
                    )
                """
                )

                # Copy data from old table to new table
                intersecting = list(expected_cols & old_cols)
                if intersecting:
                    cols_csv = ", ".join(intersecting)
                    cur.execute(f"INSERT INTO history ({cols_csv}) SELECT {cols_csv} FROM history_old")

                # Drop the old table
                cur.execute("DROP TABLE history_old")

                # Commit the transaction
                self.connection.execute("COMMIT")
                logger.info("History table migration completed successfully.")

            except Exception as e:
                # Rollback the transaction on any error
                self.connection.execute("ROLLBACK")
                logger.error(f"History table migration failed: {e}")
                raise

    def _create_history_table(self) -> None:
        with self._lock:
            try:
                self.connection.execute("BEGIN")
                self.connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS history (
                        id           TEXT PRIMARY KEY,
                        memory_id    TEXT,
                        old_memory   TEXT,
                        new_memory   TEXT,
                        event        TEXT,
                        created_at   DATETIME,
                        updated_at   DATETIME,
                        is_deleted   INTEGER,
                        actor_id     TEXT,
                        role         TEXT
                    )
                """
                )
                self.connection.execute("COMMIT")
            except Exception as e:
                self.connection.execute("ROLLBACK")
                logger.error(f"Failed to create history table: {e}")
                raise

    def _create_messages_table(self) -> None:
        with self._lock:
            try:
                self.connection.execute("BEGIN")
                self.connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS messages (
                        id TEXT PRIMARY KEY,
                        session_scope TEXT,
                        role TEXT,
                        content TEXT,
                        name TEXT,
                        created_at DATETIME
                    )
                """
                )
                self.connection.execute("COMMIT")
            except Exception as e:
                self.connection.execute("ROLLBACK")
                logger.error(f"Failed to create messages table: {e}")
                raise

    def add_history(
        self,
        memory_id: str,
        old_memory: Optional[str],
        new_memory: Optional[str],
        event: str,
        *,
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None,
        is_deleted: int = 0,
        actor_id: Optional[str] = None,
        role: Optional[str] = None,
    ) -> None:
        with self._lock:
            try:
                self.connection.execute("BEGIN")
                self.connection.execute(
                    """
                    INSERT INTO history (
                        id, memory_id, old_memory, new_memory, event,
                        created_at, updated_at, is_deleted, actor_id, role
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        str(uuid.uuid4()),
                        memory_id,
                        old_memory,
                        new_memory,
                        event,
                        created_at,
                        updated_at,
                        is_deleted,
                        actor_id,
                        role,
                    ),
                )
                self.connection.execute("COMMIT")
            except Exception as e:
                self.connection.execute("ROLLBACK")
                logger.error(f"Failed to add history record: {e}")
                raise

    def batch_add_history(self, records: List[Dict[str, Any]]) -> None:
        with self._lock:
            try:
                self.connection.execute("BEGIN")
                self.connection.executemany(
                    """
                    INSERT INTO history (
                        id, memory_id, old_memory, new_memory, event,
                        created_at, updated_at, is_deleted, actor_id, role
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    [
                        (
                            str(uuid.uuid4()),
                            record.get("memory_id"),
                            record.get("old_memory"),
                            record.get("new_memory"),
                            record.get("event"),
                            record.get("created_at"),
                            record.get("updated_at"),
                            record.get("is_deleted", 0),
                            record.get("actor_id"),
                            record.get("role"),
                        )
                        for record in records
                    ],
                )
                self.connection.execute("COMMIT")
            except Exception as e:
                self.connection.execute("ROLLBACK")
                logger.error(f"Failed to batch add history records: {e}")
                raise

    def get_history(self, memory_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            cur = self.connection.execute(
                """
                SELECT id, memory_id, old_memory, new_memory, event,
                       created_at, updated_at, is_deleted, actor_id, role
                FROM history
                WHERE memory_id = ?
                ORDER BY created_at ASC, updated_at ASC, rowid ASC
            """,
                (memory_id,),
            )
            rows = cur.fetchall()

        # created_at first, then updated_at to break ties between writes that
        # share a created_at. The SQL ordering above is only the final
        # tiebreak: list.sort is stable, so rows whose instant keys are exactly
        # equal keep it, and `rowid` makes even a full tie resolve to insertion
        # order -- SQLite does not promise a stable sort for one otherwise.
        # Rows that the old DATETIME(updated_at) tied -- it truncates to the
        # second -- but that differ sub-second are now separated, which is the
        # point of this change.
        rows.sort(key=lambda row: (_instant_key(row[5]), _instant_key(row[6])))

        return [
            {
                "id": r[0],
                "memory_id": r[1],
                "old_memory": r[2],
                "new_memory": r[3],
                "event": r[4],
                "created_at": r[5],
                "updated_at": r[6],
                "is_deleted": bool(r[7]),
                "actor_id": r[8],
                "role": r[9],
            }
            for r in rows
        ]

    def save_messages(self, messages: List[Dict[str, Any]], session_scope: str) -> None:
        if not messages:
            return
        with self._lock:
            try:
                self.connection.execute("BEGIN")
                now = datetime.now(timezone.utc).isoformat()
                for message in messages:
                    self.connection.execute(
                        """
                        INSERT INTO messages (id, session_scope, role, content, name, created_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """,
                        (
                            str(uuid.uuid4()),
                            session_scope,
                            message.get("role"),
                            message.get("content"),
                            message.get("name"),
                            now,
                        ),
                    )
                # Evict old messages beyond the most recent 10 for this scope.
                # Wrapped in a derived table to force SQLite to materialize the
                # ORDER BY before the outer NOT IN evaluates it.
                self.connection.execute(
                    """
                    DELETE FROM messages WHERE session_scope = ? AND id NOT IN (
                        SELECT id FROM (
                            SELECT id FROM messages WHERE session_scope = ? ORDER BY created_at DESC LIMIT 10
                        )
                    )
                """,
                    (session_scope, session_scope),
                )
                self.connection.execute("COMMIT")
            except Exception as e:
                self.connection.execute("ROLLBACK")
                logger.error(f"Failed to save messages: {e}")
                raise

    def get_last_messages(self, session_scope: str, limit: int = 10) -> List[Dict[str, Any]]:
        with self._lock:
            # Subquery picks the latest N rows (DESC + LIMIT), outer query
            # re-sorts them chronologically (ASC) for the caller.
            cur = self.connection.execute(
                """
                SELECT role, content, name, created_at FROM (
                    SELECT role, content, name, created_at
                    FROM messages
                    WHERE session_scope = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                ) ORDER BY created_at ASC
            """,
                (session_scope, limit),
            )
            rows = cur.fetchall()

        return [
            {
                "role": r[0],
                "content": r[1],
                "name": r[2],
                "created_at": r[3],
            }
            for r in rows
        ]

    def reset(self) -> None:
        """Drop both tables. Caller is expected to replace this instance."""
        if not self.connection:
            raise RuntimeError("Cannot reset a closed SQLiteManager")
        with self._lock:
            try:
                self.connection.execute("BEGIN")
                self.connection.execute("DROP TABLE IF EXISTS history")
                self.connection.execute("DROP TABLE IF EXISTS messages")
                self.connection.execute("COMMIT")
            except Exception as e:
                self.connection.execute("ROLLBACK")
                logger.error(f"Failed to reset tables: {e}")
                raise

    def close(self) -> None:
        if self.connection:
            self.connection.close()
            self.connection = None

    def __del__(self):
        self.close()
