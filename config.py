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

# ── SmolVLM2 Models ───────────────────────────────────────────────────────────
# Primary for 6GB VRAM: 500M — Needs ~1.5GB VRAM, leaves room for OS & KV Cache
# Secondary (Images only or high VRAM): 2.2B — Needs ~5.2GB VRAM
SMOLVLM2_LARGE = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
SMOLVLM2_SMALL = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

# Local Whisper fallback model size
# Options: tiny / base / small / medium / large-v3
WHISPER_LOCAL_MODEL = "small"

# ── VRAM Thresholds ───────────────────────────────────────────────────────────
# Adjusted for 6GB GPUs: Must have at least 5.5GB free to risk loading the 2.2B
VRAM_THRESHOLD_LARGE_GB = 5.5   # min free VRAM to run 2.2B
VRAM_THRESHOLD_SMALL_GB = 2.0   # min free VRAM to run 500M

# ── Free Tier Quota Limits ────────────────────────────────────────────────────
QUOTA_LIMITS = {
    "groq_whisper":      2000,   # requests/day
    "groq_whisper_sec":  7200,   # audio seconds/hour
    "openrouter":        50,
}

# ── Audio Processing ──────────────────────────────────────────────────────────
MAX_AUDIO_SIZE_MB  = 24      # Groq free tier: 25MB max per request
CHUNK_DURATION_SEC = 300     # 5 min chunks for long audio

# ── Analysis Modes ────────────────────────────────────────────────────────────
VALID_MODES = ["full", "fast", "ui", "ads", "workflow"]

# ── Supported Formats ─────────────────────────────────────────────────────────
SUPPORTED_VIDEO_FORMATS = {".mp4", ".mov", ".avi", ".webm", ".mkv", ".m4v"}
SUPPORTED_IMAGE_FORMATS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}