"""pipeline.py — Main Orchestration"""
import json, time, logging, concurrent.futures
from pathlib import Path
from dataclasses import dataclass, field, asdict
from config import CACHE_DIR, VALID_MODES
from system_check import check
from cache import VideoCache
from quota import QuotaTracker
from preprocessor import Preprocessor
from vision_engine import VisionEngine
from audio_engine import AudioEngine

log = logging.getLogger("pipeline")

@dataclass
class Result:
    mode: str = ""; from_cache: bool = False; degraded: bool = False; processing_sec: float = 0.0
    providers: dict = field(default_factory=dict); errors: list = field(default_factory=list); warnings: list = field(default_factory=list)
    summary: str = ""; scenes: list = field(default_factory=list); objects: list = field(default_factory=list); text_visible: list = field(default_factory=list)
    actions: list = field(default_factory=list); ui_elements: list = field(default_factory=list); entities: dict = field(default_factory=dict)
    transcript: str = ""; segments: list = field(default_factory=list); language: str = ""; aligned_events: list = field(default_factory=list)
    def to_dict(self): return asdict(self)
    def to_json(self, indent=2): return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

class Pipeline:
    def __init__(self):
        self.caps = check(); self.cache = VideoCache(CACHE_DIR); self.quota = QuotaTracker()
        self.preprocessor = Preprocessor(); self.vision = VisionEngine(self.caps, self.quota); self.audio = AudioEngine(self.caps, self.quota)
        log.info("Pipeline ready ✓")

    def analyze(self, file_path, mode="full", save_to=None):
        t0 = time.time()
        if mode not in VALID_MODES: raise ValueError(f"mode must be one of {VALID_MODES}")
        cached = self.cache.get(file_path, mode)
        if cached: return Result(**{k: cached.get(k, v) for k, v in Result.__dataclass_fields__.items()})
        info = self.preprocessor.validate(file_path)
        if not info["valid"]: r = Result(mode=mode, degraded=True); r.errors.append(f"Invalid file: {info}"); return r
        result = Result(mode=mode); is_video = info["is_video"]
        audio_path = self.preprocessor.extract_audio(file_path) if is_video else None
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            vf = ex.submit(self.vision.analyze_video if is_video else self.vision.analyze_image, file_path, mode)
            af = ex.submit(self.audio.transcribe, audio_path) if is_video else None
            try:
                vd = vf.result(timeout=300); result.providers["vision"] = vd.pop("provider", "unknown")
                vd.pop("input_type", None); vd.pop("parse_error", None); result.summary = vd.get("summary", ""); result.scenes = vd.get("scenes", [])
                result.objects = vd.get("objects", []); result.text_visible = vd.get("text_visible", []); result.actions = vd.get("actions", [])
                result.ui_elements = vd.get("ui_elements", []); result.entities = vd.get("entities", {})
            except Exception as e: result.errors.append(f"vision: {e}"); result.degraded = True
            if af:
                try:
                    ad = af.result(timeout=300); result.providers["audio"] = ad.get("provider", "unknown")
                    result.transcript = ad.get("transcript", ""); result.segments = ad.get("segments", []); result.language = ad.get("language", "")
                except Exception as e: result.errors.append(f"audio: {e}"); result.degraded = True
        if result.scenes and result.segments: result.aligned_events = self._align(result.scenes, result.segments)
        result.processing_sec = round(time.time() - t0, 2)
        self.cache.set(file_path, mode, result.to_dict())
        if save_to: Path(save_to).write_text(result.to_json())
        return result

    def ask(self, file_path, question): return self.vision.ask(file_path, question, is_video=Path(file_path).suffix.lower() in {".mp4", ".mov", ".avi", ".webm", ".mkv"})

    def _align(self, scenes, segments):
        aligned = []
        for i, scene in enumerate(scenes):
            aligned.append({"scene_index": i, "timestamp": scene.get("timestamp", f"~{i}s"), "visual": scene.get("description", ""), "speech": "(no speech)"})
        return aligned

    def quota_status(self): return self.quota.status()
    def system_status(self): return asdict(self.caps)
    def cleanup(self): self.preprocessor.cleanup(); self.cache.evict_old()