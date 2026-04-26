"""
audio_engine.py — Audio Transcription
FIXED: Forces local whisper to CPU to avoid VRAM war with SmolVLM2.
"""
import os, logging
from pathlib import Path

log = logging.getLogger("audio_engine")

EMPTY = {"transcript": "", "segments": [], "language": "unknown", "provider": "none"}


class AudioEngine:
    def __init__(self, caps, quota):
        self.caps        = caps
        self.quota       = quota
        self._local_mdl  = None   # lazy-loaded

    def transcribe(self, audio_path: str | None) -> dict:
        if not audio_path or not Path(audio_path).exists():
            return {**EMPTY, "warning": "No audio file"}

        if self.caps.has_groq and self.quota.can_use("groq_whisper"):
            r = self._groq(audio_path)
            if r:
                self.quota.record("groq_whisper")
                log.info(f"Groq quota left: {self.quota.remaining('groq_whisper')}/day")
                return r

        if self.caps.has_faster_whisper:
            r = self._local(audio_path)
            if r: return r

        log.warning("All transcription failed — returning empty (vision still works)")
        return {**EMPTY, "warning": "Set GROQ_API_KEY or install faster-whisper"}

    def _groq(self, audio_path: str) -> dict | None:
        try:
            from groq import Groq
            client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            p      = Path(audio_path)
            size   = p.stat().st_size / 1e6
            log.info(f"Groq Whisper: {p.name} ({size:.1f}MB)...")

            with open(p, "rb") as f:
                resp = client.audio.transcriptions.create(
                    file=(p.name, f.read()),
                    model="whisper-large-v3",
                    response_format="verbose_json",
                    timestamp_granularities=["segment"],
                )

            segs = []
            if hasattr(resp, "segments") and resp.segments:
                segs = [{"start": round(s.start, 2), "end": round(s.end, 2),
                         "text": s.text.strip()} for s in resp.segments]

            lang = getattr(resp, "language", "unknown")
            log.info(f"Groq done ✓ | lang={lang} | {len(segs)} segments")
            return {"transcript": resp.text, "segments": segs,
                    "language": lang, "provider": "groq_whisper_large_v3"}

        except Exception as e:
            log.error(f"Groq Whisper failed: {e}")
            return None

    def _local(self, audio_path: str) -> dict | None:
        try:
            from faster_whisper import WhisperModel
            from config import WHISPER_LOCAL_MODEL

            if self._local_mdl is None:
                # FIX: Force CPU and int8 to avoid stealing VRAM from SmolVLM2
                device  = "cpu"
                compute = "int8"
                log.info(f"Loading Whisper [{WHISPER_LOCAL_MODEL}] on CPU (int8) to save GPU VRAM...")
                self._local_mdl = WhisperModel(WHISPER_LOCAL_MODEL,
                                               device=device, compute_type=compute)
                log.info("Local Whisper ready ✓")

            segs_gen, info = self._local_mdl.transcribe(audio_path, beam_size=5)
            segs = [{"start": round(s.start, 2), "end": round(s.end, 2),
                     "text": s.text.strip()} for s in segs_gen]
            text = " ".join(s["text"] for s in segs)

            log.info(f"Local Whisper done ✓ | lang={info.language} | {len(segs)} segs")
            return {"transcript": text, "segments": segs,
                    "language": info.language,
                    "provider": f"local_faster_whisper_{WHISPER_LOCAL_MODEL}"}

        except Exception as e:
            log.error(f"Local Whisper failed: {e}")
            return None