"""
logger.py
Logs drowsiness events to a CSV file for review.
"""

import csv
import os
import time


class DrowsinessLogger:
    """
    Records every drowsiness and yawn event to logs/session_YYYYMMDD_HHMMSS.csv
    """

    LOG_DIR = "logs"

    def __init__(self):
        os.makedirs(self.LOG_DIR, exist_ok=True)
        timestamp      = time.strftime("%Y%m%d_%H%M%S")
        self.log_path  = os.path.join(self.LOG_DIR, f"session_{timestamp}.csv")
        self._events   = []
        self._start    = time.time()

        print(f"[INFO] Logging to: {self.log_path}")

    def log_event(self, event_type: str, result: dict):
        """
        Parameters
        ----------
        event_type : "DROWSY" or "YAWN"
        result     : dict returned by DrowsinessDetector.detect()
        """
        entry = {
            "timestamp"    : time.strftime("%H:%M:%S"),
            "elapsed_sec"  : round(time.time() - self._start, 1),
            "event"        : event_type,
            "ear"          : result.get("ear", "N/A"),
            "mar"          : result.get("mar", "N/A"),
            "frame_counter": result.get("frame_counter", 0),
            "total_blinks" : result.get("total_blinks", 0),
            "total_yawns"  : result.get("total_yawns", 0),
        }
        self._events.append(entry)

    def save(self):
        """Write all events to CSV on session end."""
        if not self._events:
            print("[INFO] No events to log.")
            return

        fields = ["timestamp", "elapsed_sec", "event", "ear", "mar",
                  "frame_counter", "total_blinks", "total_yawns"]

        with open(self.log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self._events)

        print(f"[INFO] {len(self._events)} events saved to {self.log_path}")
