"""
cache.py — Smart File Cache
Key = hash(filename + size + mtime + mode) — no file content read needed.
"""
import hashlib, json, logging
from pathlib import Path
from datetime import datetime

log = logging.getLogger("cache")


class VideoCache:
    def __init__(self, cache_dir: Path):
        self.dir = Path(cache_dir)
        self.dir.mkdir(exist_ok=True)

    def _key(self, path: str, mode: str) -> str:
        p = Path(path)
        try:
            s  = p.stat()
            fp = f"{p.name}|{s.st_size}|{s.st_mtime}|{mode}"
        except Exception:
            fp = f"{p.name}|{mode}"
        return hashlib.md5(fp.encode()).hexdigest()

    def get(self, path: str, mode: str) -> dict | None:
        f = self.dir / f"{self._key(path, mode)}.json"
        if not f.exists(): return None
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            log.info(f"Cache HIT [{mode}]: {Path(path).name}")
            return {**data, "from_cache": True}
        except Exception as e:
            log.warning(f"Cache corrupted — rebuilding: {e}")
            f.unlink(missing_ok=True)
            return None

    def set(self, path: str, mode: str, data: dict):
        try:
            (self.dir / f"{self._key(path, mode)}.json").write_text(
                json.dumps({**data, "cached_at": datetime.now().isoformat()},
                           ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as e:
            log.warning(f"Cache write failed: {e}")

    def evict_old(self, max_age_days: int = 7):
        cutoff = datetime.now().timestamp() - max_age_days * 86400
        removed = 0
        for f in self.dir.glob("*.json"):
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink()
                    removed += 1
            except OSError:
                pass
        if removed: log.info(f"Evicted {removed} old cache entries")