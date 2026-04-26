"""
preprocessor.py — Local Preprocessing (CPU only, zero API calls)
Extracts compressed audio for Groq Whisper.
"""
import subprocess, shutil, logging, tempfile
from pathlib import Path
from config import (SUPPORTED_VIDEO_FORMATS, SUPPORTED_IMAGE_FORMATS,
                    MAX_AUDIO_SIZE_MB, CHUNK_DURATION_SEC)

log = logging.getLogger("preprocessor")


class Preprocessor:
    def __init__(self):
        self.has_ffmpeg  = shutil.which("ffmpeg")  is not None
        self.has_ffprobe = shutil.which("ffprobe") is not None
        self._tmp        = Path(tempfile.mkdtemp(prefix="smolvlm_"))
        if not self.has_ffmpeg:
            log.warning("ffmpeg not found — audio extraction disabled")

    def validate(self, file_path: str) -> dict:
        p   = Path(file_path)
        ext = p.suffix.lower()
        return {
            "valid":    p.exists() and ext in (SUPPORTED_VIDEO_FORMATS | SUPPORTED_IMAGE_FORMATS),
            "exists":   p.exists(),
            "is_video": ext in SUPPORTED_VIDEO_FORMATS,
            "is_image": ext in SUPPORTED_IMAGE_FORMATS,
            "format":   ext,
            "size_mb":  round(p.stat().st_size / 1e6, 1) if p.exists() else 0,
        }

    def extract_audio(self, video_path: str) -> str | None:
        if not self.has_ffmpeg: return None

        p   = Path(video_path)
        out = self._tmp / f"{p.stem}_audio.mp3"

        if out.exists():
            log.info(f"Reusing cached audio: {out.name}")
            return str(out)

        log.info(f"Extracting audio: {p.name}...")
        try:
            r = subprocess.run([
                "ffmpeg", "-i", str(p),
                "-vn", "-ac", "1", "-ar", "16000", "-b:a", "32k",
                "-y", str(out)
            ], capture_output=True, timeout=120)

            if r.returncode != 0:
                log.error(f"ffmpeg error: {r.stderr.decode()[:200]}")
                return None

            size_mb = out.stat().st_size / 1e6
            log.info(f"Audio ready: {size_mb:.1f}MB")

            if size_mb > MAX_AUDIO_SIZE_MB:
                log.warning(f"Audio too large — chunking to first {CHUNK_DURATION_SEC}s")
                return self._chunk(str(out))

            return str(out)

        except subprocess.TimeoutExpired:
            log.error("Audio extraction timed out")
            return None
        except Exception as e:
            log.error(f"Audio extraction failed: {e}")
            return None

    def _chunk(self, audio_path: str) -> str:
        out = self._tmp / "audio_chunk.mp3"
        subprocess.run(["ffmpeg", "-i", audio_path, "-t", str(CHUNK_DURATION_SEC),
                        "-y", str(out)], capture_output=True, timeout=60)
        return str(out) if out.exists() else audio_path

    def get_duration(self, video_path: str) -> float:
        if not self.has_ffprobe: return 0.0
        try:
            r = subprocess.run(["ffprobe", "-v", "quiet", "-show_entries",
                                 "format=duration", "-of", "csv=p=0", str(video_path)],
                                capture_output=True, text=True, timeout=15)
            return float(r.stdout.strip() or "0")
        except Exception: return 0.0

    def cleanup(self):
        import shutil as sh
        sh.rmtree(self._tmp, ignore_errors=True)