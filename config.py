"""
config.py — Central Configuration
All settings in one place. Edit here, works everywhere.
"""
import os
from pathlib import Path

# ── Directories ───────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
CACHE_DIR  = BASE_DIR / ".cache"
QUOTA_FILE = BASE_DIR / ".quota.json"
LOG_FILE   = BASE_DIR / "pipeline.log"

CACHE_DIR.mkdir(exist_ok=True)

# ── API Keys ──────────────────────────────────────────────────────────────────
GROQ_API_KEY        = os.getenv("GROQ_API_KEY", "")
OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY", "")
GEMINI_API_KEY      = os.getenv("GEMINI_API_KEY", "")  # Added Gemini

# ── SmolVLM2 Models ───────────────────────────────────────────────────────────
SMOLVLM2_LARGE = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
SMOLVLM2_SMALL = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

# Local Whisper fallback model size
WHISPER_LOCAL_MODEL = "small"

# ── VRAM Thresholds ───────────────────────────────────────────────────────────
VRAM_THRESHOLD_LARGE_GB = 5.5
VRAM_THRESHOLD_SMALL_GB = 2.0

# ── Free Tier Quota Limits ────────────────────────────────────────────────────
QUOTA_LIMITS = {
    "groq_whisper":      2000,
    "groq_whisper_sec":  7200,
    "openrouter":        50,
    "gemini_vision":     1500,  # Gemini free tier limit
}

# ── Audio Processing ──────────────────────────────────────────────────────────
MAX_AUDIO_SIZE_MB  = 24
CHUNK_DURATION_SEC = 300

# ── Analysis Modes ────────────────────────────────────────────────────────────
VALID_MODES = ["full", "fast", "ui", "ads", "workflow"]

# ── Supported Formats ─────────────────────────────────────────────────────────
SUPPORTED_VIDEO_FORMATS = {".mp4", ".mov", ".avi", ".webm", ".mkv", ".m4v"}
SUPPORTED_IMAGE_FORMATS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}