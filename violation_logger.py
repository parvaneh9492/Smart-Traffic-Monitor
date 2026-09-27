import os
import time
import sqlite3
import cv2
import config


class ViolationLogger:
    """
    Persists speeding violations to a small SQLite database, along with a
    snapshot image of the offending vehicle -- exactly the kind of evidence
    trail a real enforcement system needs.
    """

    def __init__(self):
        self.conn = sqlite3.connect(config.LOG_DB_PATH)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS violations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                track_id INTEGER,
                plate_text TEXT,
                speed_kmh REAL,
                timestamp REAL,
                snapshot_path TEXT
            )
        """)
        self.conn.commit()
        os.makedirs(config.SNAPSHOT_DIR, exist_ok=True)
        self.already_logged = set()   # avoid logging the same vehicle on every frame

    def maybe_log_speeding(self, track_id, plate_text, speed_kmh, frame, box):
        if speed_kmh is None or speed_kmh <= config.SPEED_LIMIT_KMH:
            return
        if track_id in self.already_logged:
            return

        self.already_logged.add(track_id)
        x1, y1, x2, y2 = box
        snapshot_path = os.path.join(config.SNAPSHOT_DIR, f"{track_id}_{int(time.time())}.jpg")
        cv2.imwrite(snapshot_path, frame[max(0, y1):y2, max(0, x1):x2])

        self.conn.execute(
            "INSERT INTO violations (track_id, plate_text, speed_kmh, timestamp, snapshot_path) "
            "VALUES (?, ?, ?, ?, ?)",
            (track_id, plate_text or "UNKNOWN", speed_kmh, time.time(), snapshot_path),
        )
        self.conn.commit()
