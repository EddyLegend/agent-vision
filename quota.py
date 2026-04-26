"""
quota.py — Daily Quota Tracker (persists across restarts)
"""
import json, logging
from datetime import date
from pathlib import Path
from config import QUOTA_LIMITS, QUOTA_FILE

log = logging.getLogger("quota")


class QuotaTracker:
    def __init__(self):
        self._file = Path(QUOTA_FILE)
        self._data = self._load()

    def _load(self) -> dict:
        if self._file.exists():
            try:
                raw = json.loads(self._file.read_text())
                if raw.get("date") == str(date.today()):
                    return raw
            except Exception: pass
        return {"date": str(date.today()), "counts": {}}

    def _save(self):
        try: self._file.write_text(json.dumps(self._data, indent=2))
        except Exception as e: log.error(f"Quota save failed: {e}")

    def can_use(self, service: str) -> bool:
        limit = QUOTA_LIMITS.get(service, 9999)
        used  = self._data["counts"].get(service, 0)
        if used >= limit:
            log.warning(f"Quota exhausted: {service} ({used}/{limit}) — resets at midnight UTC")
            return False
        return True

    def record(self, service: str, amount: int = 1):
        self._data["counts"][service] = self._data["counts"].get(service, 0) + amount
        self._save()

    def remaining(self, service: str) -> int:
        return max(0, QUOTA_LIMITS.get(service, 9999) - self._data["counts"].get(service, 0))

    def status(self) -> dict:
        return {
            s: {
                "used": self._data["counts"].get(s, 0),
                "limit": QUOTA_LIMITS[s],
                "remaining": self.remaining(s),
                "pct_used": round(self._data["counts"].get(s, 0) / QUOTA_LIMITS[s] * 100, 1),
            } for s in QUOTA_LIMITS
        }