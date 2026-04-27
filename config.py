"""config.py — Central Configuration"""
import os
from pathlib import Path

BASE_DIR   = Path(__file__).parent
CACHE_DIR  = BASE_DIR / ".cache"
QUOTA_FILE = BASE_DIR / ".quota.json"
LOG_FILE   = BASE_DIR / "pipeline.log"
CACHE_DIR.mkdir(exist_ok=True)

GROQ_API_KEY        = os.getenv("GROQ_API_KEY", "")
OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY", "")
GEMINI_API_KEY      = os.getenv("GEMINI_API_KEY", "")

SMOLVLM2_LARGE = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
SMOLVLM2_SMALL = "./models/smolvlm-500m"
WHISPER_LOCAL_MODEL = "small"

VRAM_THRESHOLD_LARGE_GB = 99.0
VRAM_THRESHOLD_SMALL_GB = 2.0

QUOTA_LIMITS = {
    "groq_whisper":      2000,
    "groq_whisper_sec":  7200,
    "openrouter":        50,
    "gemini_vision":     1500,
}

MAX_AUDIO_SIZE_MB  = 24
CHUNK_DURATION_SEC = 300
VALID_MODES = ["full", "fast", "ui", "ads", "workflow"]

SUPPORTED_VIDEO_FORMATS = {".mp4", ".mov", ".avi", ".webm", ".mkv", ".m4v"}
SUPPORTED_IMAGE_FORMATS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}