#!/usr/bin/env python3
"""
Database Manager for Persistent Scan Results
Handles SQLite database operations for incremental scanning
"""
import json
import logging
import os
import sqlite3
import threading
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages SQLite database for persistent scan results.

    Thread-safe: uses a lock for all SQLite access to avoid concurrent
    reads/writes on separate connections racing each other.
    """

    def __init__(self, db_path: str = "scan_cache.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        self.init_database()

    def _connect(self):
        """Create a SQLite connection with a busy timeout."""
        return sqlite3.connect(self.db_path, timeout=30)

    def init_database(self):
        """Initialize database tables."""
        with self._lock, self._connect() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS file_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT UNIQUE NOT NULL,
                    file_size INTEGER NOT NULL,
                    modification_time REAL NOT NULL,
                    hash_md5 TEXT,
                    hash_sha256 TEXT,
                    hash_perceptual TEXT,
                    hash_audio TEXT,
                    hash_video TEXT,
                    text_embedding BLOB,
                    scan_timestamp REAL NOT NULL,
                    algorithm_version TEXT NOT NULL,
                    metadata JSON
                )
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS scan_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE NOT NULL,
                    directory_path TEXT NOT NULL,
                    start_time REAL NOT NULL,
                    end_time REAL,
                    total_files INTEGER,
                    new_files INTEGER,
                    modified_files INTEGER,
                    algorithm_used TEXT NOT NULL,
                    filters_used JSON,
                    status TEXT DEFAULT 'running'
                )
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS duplicate_groups (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    group_hash TEXT NOT NULL,
                    file_paths JSON NOT NULL,
                    algorithm_used TEXT NOT NULL,
                    similarity_score REAL,
                    created_time REAL NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES scan_sessions (session_id)
                )
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    timestamp REAL NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES scan_sessions (session_id)
                )
                """
            )

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_file_path ON file_cache(file_path)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_modification_time ON file_cache(modification_time)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_session_directory ON scan_sessions(directory_path)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_group_hash ON duplicate_groups(group_hash)")
            conn.commit()

    def start_scan_session(self, directory: str, algorithm: str, filters: Dict) -> str:
        """Start a new scan session."""
        session_id = f"scan_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{uuid.uuid4().hex}"

        with self._lock, self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO scan_sessions
                (session_id, directory_path, start_time, algorithm_used, filters_used)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, directory, datetime.now().timestamp(), algorithm, json.dumps(filters)),
            )
            conn.commit()

        return session_id

    def end_scan_session(self, session_id: str, total_files: int, new_files: int, modified_files: int):
        """Mark scan session as completed."""
        with self._lock, self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE scan_sessions
                SET end_time = ?, total_files = ?, new_files = ?, modified_files = ?, status = 'completed'
                WHERE session_id = ?
                """,
                (datetime.now().timestamp(), total_files, new_files, modified_files, session_id),
            )
            conn.commit()

    def get_cached_file_info(self, file_path: str) -> Optional[Dict]:
        """Get cached file information if it exists and is up to date."""
        if not os.path.exists(file_path):
            return None

        current_size = os.path.getsize(file_path)
        current_mtime = os.path.getmtime(file_path)

        with self._lock, self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM file_cache
                WHERE file_path = ? AND file_size = ? AND modification_time = ?
                """,
                (file_path, current_size, current_mtime),
            )

            row = cursor.fetchone()
            if row:
                columns = [desc[0] for desc in cursor.description]
                return dict(zip(columns, row))

        return None

    def cache_file_info(
        self,
        file_path: str,
        hashes: Dict[str, str],
        algorithm_version: str = "1.0",
        metadata: Optional[Dict] = None,
    ):
        """Cache file information."""
        if not os.path.exists(file_path):
            return

        file_size = os.path.getsize(file_path)
        modification_time = os.path.getmtime(file_path)

        with self._lock, self._connect() as conn:
            cursor = conn.cursor()

            text_embedding = None
            if "text_embedding" in hashes:
                # DB-4: Use JSON instead of pickle to avoid deserialization security vulnerability.
                # Numpy arrays are converted to a list of floats before JSON encoding.
                raw_emb = hashes["text_embedding"]
                try:
                    # Support numpy arrays, lists, and anything with a tolist() method
                    emb_list = raw_emb.tolist() if hasattr(raw_emb, "tolist") else list(raw_emb)
                    text_embedding = json.dumps(emb_list).encode("utf-8")
                except Exception as e:
                    logger.warning(f"Could not serialize text_embedding for {file_path}: {e}")

            cursor.execute(
                """
                INSERT OR REPLACE INTO file_cache
                (file_path, file_size, modification_time, hash_md5, hash_sha256,
                 hash_perceptual, hash_audio, hash_video, text_embedding,
                 scan_timestamp, algorithm_version, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    file_path,
                    file_size,
                    modification_time,
                    hashes.get("md5"),
                    hashes.get("sha256"),
                    hashes.get("perceptual"),
                    hashes.get("audio"),
                    hashes.get("video"),
                    text_embedding,
                    datetime.now().timestamp(),
                    algorithm_version,
                    json.dumps(metadata or {}),
                ),
            )
            conn.commit()

    def save_duplicate_groups(self, session_id: str, duplicates: Dict, algorithm: str):
        """Save duplicate groups to database."""
        with self._lock, self._connect() as conn:
            cursor = conn.cursor()

            for group_hash, file_paths in duplicates.items():
                if len(file_paths) > 1:
                    cursor.execute(
                        """
                        INSERT INTO duplicate_groups
                        (session_id, group_hash, file_paths, algorithm_used, created_time)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            session_id,
                            group_hash,
                            json.dumps(file_paths),
                            algorithm,
                            datetime.now().timestamp(),
                        ),
                    )
            conn.commit()

    def get_scan_history(self, directory: Optional[str] = None, limit: int = 10) -> List[Dict]:
        """Get scan history for a directory or all scans."""
        with self._lock, self._connect() as conn:
            cursor = conn.cursor()

            if directory:
                cursor.execute(
                    """
                    SELECT * FROM scan_sessions
                    WHERE directory_path = ?
                    ORDER BY start_time DESC LIMIT ?
                    """,
                    (directory, limit),
                )
            else:
                cursor.execute(
                    """
                    SELECT * FROM scan_sessions
                    ORDER BY start_time DESC LIMIT ?
                    """,
                    (limit,),
                )

            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def get_incremental_files(self, directory: str) -> Tuple[List[str], List[str]]:
        """Get files that need scanning (new or modified).

        DB-2 fix: uses a single batch SQL query instead of one query per file.
        """
        new_files = []
        modified_files = []
        all_files = []

        for root, dirs, files in os.walk(directory):
            for file_name in files:
                all_files.append(os.path.join(root, file_name))

        if not all_files:
            return new_files, modified_files

        # Build a dict of current filesystem info in one pass
        current_info: Dict[str, Tuple[int, float]] = {}
        for file_path in all_files:
            try:
                if os.path.exists(file_path):
                    stat = os.stat(file_path)
                    current_info[file_path] = (stat.st_size, stat.st_mtime)
            except OSError:
                pass

        # Fetch all cached entries in a single query and compare in Python (DB-2 fix)
        with self._lock, self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT file_path, file_size, modification_time FROM file_cache",
            )
            cached: Dict[str, Tuple[int, float]] = {
                row[0]: (row[1], row[2]) for row in cursor.fetchall()
            }

        for file_path, (cur_size, cur_mtime) in current_info.items():
            if file_path not in cached:
                new_files.append(file_path)
            else:
                cached_size, cached_mtime = cached[file_path]
                if cached_size != cur_size or cached_mtime != cur_mtime:
                    modified_files.append(file_path)

        return new_files, modified_files

    def cleanup_old_cache(self, days: int = 30):
        """Clean up cache entries older than specified days."""
        cutoff_time = datetime.now().timestamp() - (days * 24 * 60 * 60)

        with self._lock, self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM file_cache WHERE scan_timestamp < ?", (cutoff_time,))
            cursor.execute("DELETE FROM scan_sessions WHERE start_time < ?", (cutoff_time,))
            cursor.execute(
                """
                DELETE FROM duplicate_groups
                WHERE session_id NOT IN (SELECT session_id FROM scan_sessions)
                """
            )
            conn.commit()

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        with self._lock, self._connect() as conn:
            cursor = conn.cursor()
            stats: Dict[str, Any] = {}

            cursor.execute("SELECT COUNT(*) FROM file_cache")
            stats["cached_files"] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM scan_sessions")
            stats["total_sessions"] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM scan_sessions WHERE status = 'completed'")
            stats["completed_sessions"] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM duplicate_groups")
            stats["duplicate_groups"] = cursor.fetchone()[0]

            stats["database_size"] = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0

            cursor.execute(
                """
                SELECT directory_path, start_time FROM scan_sessions
                ORDER BY start_time DESC LIMIT 1
                """
            )
            latest = cursor.fetchone()
            if latest:
                stats["latest_scan_directory"] = latest[0]
                stats["latest_scan_time"] = datetime.fromtimestamp(latest[1])

        return stats

    def export_cache_data(self, export_path: str) -> bool:
        """Export cache data to JSON file."""
        try:
            stats = self.get_database_stats()
            scan_history = self.get_scan_history(limit=100)
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "database_stats": stats,
                "scan_history": scan_history,
                "database_version": "1.0",
            }

            with open(export_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, default=str)

            return True
        except Exception as e:
            logger.error(f"Export failed: {e}")  # DB-1: was print(), now uses logger
            return False

    def record_performance_metric(self, session_id: str, metric_name: str, metric_value: float):
        """Record a performance metric."""
        with self._lock, self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO performance_metrics
                (session_id, metric_name, metric_value, timestamp)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, metric_name, metric_value, datetime.now().timestamp()),
            )
            conn.commit()
