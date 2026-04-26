# Agent Vision Pipeline

A production-grade multimodal AI pipeline for video/image analysis and audio transcription, designed for integration with browser automation agents.

## 🚀 Features

- **Multimodal Understanding**: Analyzes videos and images using SmolVLM2 (2.2B or 500M parameter models)
- **Audio Transcription**: Uses Groq Whisper API (primary) with local faster-whisper fallback (CPU-only to save VRAM)
- **Smart Caching**: Avoids redundant processing with hash-based file caching
- **Quota Management**: Tracks daily API usage to respect free tier limits
- **Hardware Adaptive**: Automatically selects optimal model size based on available VRAM
- **Error Resilient**: Graceful fallbacks and OOM recovery
- **Agent Integration**: Designed for seamless use with browser automation agents (Playwright, etc.)

## 📋 Requirements

```bash
pip install -r requirements.txt
sudo apt install ffmpeg          # required for audio extraction
```

Optional (for faster inference on CUDA):
```bash
pip install flash-attn --no-build-isolation
```

## 🔑 Environment Variables

```bash
export GROQ_API_KEY="..."        # Get free key at: console.groq.com
export OPENROUTER_API_KEY="..."  # Optional fallback for vision tasks
```

## 🏗️ Architecture

### Core Components

1. **Preprocessor** (`preprocessor.py`)
   - Extracts audio from videos using ffmpeg
   - Validates file formats and sizes
   - CPU-only processing to avoid GPU conflicts

2. **Vision Engine** (`vision_engine.py`)
   - Uses SmolVLM2 for image/video understanding
   - Automatic model selection (2.2B vs 500M) based on VRAM
   - OOM recovery with fallback to smaller model
   - OpenRouter vision API fallback when local model unavailable

3. **Audio Engine** (`audio_engine.py`)
   - Primary: Groq Whisper-large-v3 API
   - Fallback: Local faster-whisper (forced to CPU/int8)
   - Quota-aware usage tracking

4. **Pipeline** (`pipeline.py`)
   - Main orchestrator
   - Concurrent vision/audio processing
   - Smart caching system
   - Result alignment (audio timestamps with visual scenes)
   - Agent-friendly commands (quota_status, cleanup)

5. **System Check** (`system_check.py`)
   - Hardware detection (GPU, VRAM, WSL2)
   - Dependency verification
   - Automatic provider selection

6. **Agent Interface** (`agent_interface.py`)
   - Browser agent integration helpers
   - Video detection and downloading from web pages
   - Screenshot analysis
   - Direct access to pipeline commands

## 📖 Usage

### Basic Usage

```python
from pipeline import Pipeline

pipeline = Pipeline()
result = pipeline.analyze("video.mp4", mode="full")
print(result.summary)
```

### Agent Integration

```python
from agent_interface import AgentVision

vision = AgentVision()
if vision.page_has_video(page):
    videos = vision.list_page_videos(page)
    analysis = vision.analyze_page_video(page, index=0, mode="workflow")
```

### Available Modes

- `full`: Complete analysis (scenes, objects, text, actions, UI, entities)
- `fast`: Quick summary with key elements
- `ui`: Focus on UI/software elements only
- `ads`: Advertising and product content analysis
- `workflow`: Step-by-step task extraction

## 📊 Sample Output

```json
{
  "summary": "A person demonstrates how to bake a chocolate cake, showing ingredients and mixing steps",
  "scenes": [
    {"timestamp": "00:00", "description": "Close-up of chocolate cake ingredients on counter"},
    {"timestamp": "00:15", "description": "Person mixing batter in bowl"}
  ],
  "objects": ["chocolate", "flour", "eggs", "mixing bowl", "whisk"],
  "text_visible": ["Chocolate Cake Recipe", "Prep time: 20 mins"],
  "actions": ["pouring", "mixing", "baking"],
  "ui_elements": [],
  "entities": {
    "people": ["baker"],
    "brands": [],
    "locations": ["kitchen"]
  }
}
```

## ⚙️ Configuration

Edit `config.py` to adjust:
- Model selection thresholds
- VRAM limits
- Quota settings
- Supported file formats
- Processing modes

## 🧪 Testing

Run the test script:
```bash
python test_pipeline.py
```

## 📝 Logging

Pipeline logs are saved to `pipeline.log` with INFO level by default.

## 🧹 Cleanup

```python
pipeline.cleanup()  # Clears temporary files and cache
```

## 🤖 Agent Commands

Available via `agent_interface` or direct pipeline access:
- `quota_status()` - Check API usage limits
- `system_status()` - View hardware/software detection results
- `cleanup()` - Clean temporary files

## ⚠️ Limitations

- Requires `ffmpeg` for audio extraction
- Video processing requires `decord` package
- SmolVLM2-2.2B needs ~5.2GB VRAM (use 500M model on 6GB GPUs)
- Groq API key required for audio transcription (free tier: 2000 requests/day)
- Local Whisper runs on CPU only (to avoid VRAM conflicts with vision model)

## 📄 License

Apache 2.0 - See [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- [Hugging Face SmolVLM2](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct)
- [Groq Whisper API](https://console.groq.com)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [Playwright](https://playwright.dev/)