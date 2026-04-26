"""
system_check.py — Hardware & Dependency Detection
Runs once at startup. Decides which providers to use.
"""
import os
import shutil
import platform
import logging
from dataclasses import dataclass, field
from config import VRAM_THRESHOLD_LARGE_GB, VRAM_THRESHOLD_SMALL_GB

log = logging.getLogger("system_check")


@dataclass
class SystemCaps:
    # Hardware
    has_cuda:       bool  = False
    gpu_name:       str   = ""
    vram_total_gb:  float = 0.0
    vram_free_gb:   float = 0.0
    is_wsl2:        bool  = False

    # Local tools
    has_ffmpeg:          bool = False
    has_decord:          bool = False    # required for SmolVLM2 video
    has_flash_attn:      bool = False    # optional, speeds up inference
    has_faster_whisper:  bool = False    # local audio fallback

    # API keys
    has_groq:        bool = False
    has_openrouter:  bool = False

    # Decisions
    vision_model:   str = ""  # "large"|"small"|"cpu_large"|"cpu_small"
    audio_provider: str = ""  # "groq"|"local"|"none"

    warnings: list = field(default_factory=list)
    info:     list = field(default_factory=list)


def check() -> SystemCaps:
    caps = SystemCaps()

    # WSL2
    caps.is_wsl2 = "microsoft" in platform.uname().release.lower()
    if caps.is_wsl2:
        caps.info.append("Running in WSL2")

    # CUDA / GPU
    try:
        import torch
        if torch.cuda.is_available():
            caps.has_cuda      = True
            caps.gpu_name      = torch.cuda.get_device_name(0)
            props              = torch.cuda.get_device_properties(0)
            caps.vram_total_gb = round(props.total_memory / 1e9, 1)
            caps.vram_free_gb  = round(
                (props.total_memory - torch.cuda.memory_allocated(0)) / 1e9, 1
            )
            caps.info.append(
                f"GPU: {caps.gpu_name} | "
                f"VRAM: {caps.vram_free_gb}GB free / {caps.vram_total_gb}GB total"
            )
        else:
            caps.warnings.append("No CUDA GPU — SmolVLM2 will run on CPU (slow)")
    except ImportError:
        caps.warnings.append("PyTorch not installed: pip install torch")

    # decord (required for SmolVLM2 video)
    try:
        import decord
        caps.has_decord = True
    except ImportError:
        caps.warnings.append("decord not installed! Fix: pip install decord")

    # flash-attn (optional)
    try:
        import flash_attn
        caps.has_flash_attn = True
        caps.info.append("flash-attn available — faster inference ✓")
    except ImportError:
        caps.info.append("flash-attn not installed (optional)")

    # ffmpeg
    caps.has_ffmpeg = shutil.which("ffmpeg") is not None
    if not caps.has_ffmpeg:
        caps.warnings.append("ffmpeg not found. Fix: sudo apt install ffmpeg")

    # faster-whisper
    try:
        import faster_whisper
        caps.has_faster_whisper = True
    except ImportError:
        caps.info.append("faster-whisper not installed (optional local audio)")

    # API keys
    caps.has_groq       = bool(os.getenv("GROQ_API_KEY"))
    caps.has_openrouter = bool(os.getenv("OPENROUTER_API_KEY"))

    if not caps.has_groq:
        caps.warnings.append(
            "GROQ_API_KEY not set. Free key: https://console.groq.com"
        )

    # Vision model decision (FIXED: Stricter threshold for 6GB cards)
    if caps.has_cuda:
        if caps.vram_free_gb >= VRAM_THRESHOLD_LARGE_GB:
            caps.vision_model = "large"
        elif caps.vram_free_gb >= VRAM_THRESHOLD_SMALL_GB:
            caps.vision_model = "small"
            caps.warnings.append(f"Low VRAM — using SmolVLM2-500M (Recommended for 6GB GPUs)")
        else:
            caps.vision_model = "cpu_small"
    else:
        caps.vision_model = "cpu_large"
        caps.warnings.append("No GPU — SmolVLM2 on CPU (slow)")

    # Audio provider decision
    if caps.has_groq:
        caps.audio_provider = "groq"
    elif caps.has_faster_whisper:
        caps.audio_provider = "local"
    else:
        caps.audio_provider = "none"
        caps.warnings.append("No audio transcription. Set GROQ_API_KEY or install faster-whisper")

    for msg in caps.info:    log.info(f"  ✓ {msg}")
    for msg in caps.warnings: log.warning(f"  ⚠ {msg}")
    log.info(f"Vision: SmolVLM2-{caps.vision_model} | Audio: {caps.audio_provider}")

    return caps