"""
agent_interface.py — Browser Agent Integration
FIXED: Replaced unsafe hash() with hashlib.md5 to prevent file collisions.
"""
import logging, hashlib, tempfile, urllib.request
from pathlib import Path
from pipeline import Pipeline

log = logging.getLogger("agent_vision")


class AgentVision:
    def __init__(self, groq_key: str = None):
        import os
        if groq_key: os.environ["GROQ_API_KEY"] = groq_key
        self._pipe = None
        self._tmp  = Path(tempfile.mkdtemp(prefix="agent_vision_"))

    @property
    def pipe(self) -> Pipeline:
        if self._pipe is None:
            self._pipe = Pipeline()
        return self._pipe

    def page_has_video(self, page) -> bool:
        return len(self._videos(page)) > 0

    def list_page_videos(self, page) -> list:
        return self._videos(page)

    def _videos(self, page) -> list:
        return page.evaluate("""
            () => Array.from(document.querySelectorAll('video'))
              .map((v, i) => ({
                index: i,
                src: v.src || v.querySelector('source')?.src || '',
                width: v.videoWidth || v.offsetWidth,
                height: v.videoHeight || v.offsetHeight,
                duration: v.duration || 0,
                paused: v.paused,
              }))
              .filter(v => v.src)
        """)

    def _download(self, url: str, page) -> str:
        # FIX: Safe hash for filenames
        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
        dest = self._tmp / f"video_{url_hash}.mp4"
        
        if dest.exists():
            log.info(f"Reusing cached download: {dest.name}")
            return str(dest)

        log.info(f"Downloading: {url[:70]}...")
        cookies  = page.context.cookies()
        cookie_s = "; ".join(f"{c['name']}={c['value']}" for c in cookies)

        req = urllib.request.Request(url, headers={
            "Cookie":     cookie_s,
            "User-Agent": page.evaluate("navigator.userAgent"),
            "Referer":    page.url,
        })
        with urllib.request.urlopen(req, timeout=60) as r:
            dest.write_bytes(r.read())
            
        log.info(f"Downloaded {dest.stat().st_size / 1e6:.1f}MB → {dest.name}")
        return str(dest)

    def analyze_page_video(self, page, index: int = 0, mode: str = "full") -> dict:
        vids = self._videos(page)
        if not vids: return {"error": "no_video", "page_url": page.url}
        t = vids[min(index, len(vids) - 1)]
        try:
            local  = self._download(t["src"], page)
            result = self.pipe.analyze(local, mode=mode)
            return {"page_url": page.url, "video_src": t["src"],
                    "video_meta": {k: t[k] for k in ["width","height","duration"]},
                    **result.to_dict()}
        except Exception as e:
            log.error(f"analyze_page_video failed: {e}")
            return {"error": str(e), "page_url": page.url}

    def ask_page_video(self, page, question: str, index: int = 0) -> str:
        vids = self._videos(page)
        if not vids: return "No video found."
        t = vids[min(index, len(vids) - 1)]
        try:
            return self.pipe.ask(self._download(t["src"], page), question)
        except Exception as e:
            return f"Error: {e}"

    def analyze_screenshot(self, page, mode: str = "ui") -> dict:
        url_hash = hashlib.md5(page.url.encode()).hexdigest()[:12]
        shot = str(self._tmp / f"shot_{url_hash}.png")
        page.screenshot(path=shot, full_page=False)
        result = self.pipe.analyze(shot, mode=mode)
        return {"page_url": page.url, **result.to_dict()}

    def ask_screenshot(self, page, question: str) -> str:
        url_hash = hashlib.md5((page.url + question).encode()).hexdigest()[:12]
        shot = str(self._tmp / f"shot_{url_hash}.png")
        page.screenshot(path=shot, full_page=False)
        return self.pipe.ask(shot, question)

    # Agent Commands: Direct access for the agent
    def quota_status(self)  -> dict: return self.pipe.quota_status()
    def system_status(self) -> dict: return self.pipe.system_status()

    def cleanup(self):
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)
        if self._pipe: self._pipe.cleanup()