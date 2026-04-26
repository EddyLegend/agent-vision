"""
vision_engine.py — SmolVLM2 Vision Engine
Uses the EXACT API from the HuggingFace model card.
FIXED: Proper OOM recovery (gc.collect) and logic for 6GB GPUs.
"""
import json, logging, os, base64, gc
from pathlib import Path

log = logging.getLogger("vision_engine")

PROMPTS = {
    "full": """Analyze this video/image completely.
Return ONLY raw JSON (no markdown, no explanation):
{
  "summary": "Clear description of what is happening",
  "scenes": [{"timestamp": "00:00", "description": "what you see"}],
  "objects": ["notable objects"],
  "text_visible": ["any text shown on screen"],
  "actions": ["actions being performed"],
  "ui_elements": ["any UI, apps, websites visible"],
  "entities": {"people": [], "brands": [], "locations": []}
}""",
    "fast": """Quick summary of this video/image.
Return ONLY raw JSON:
{
  "summary": "one paragraph summary",
  "key_elements": ["most important things"]
}""",
    "ui": """Analyze ONLY the UI/software visible in this video/image.
Return ONLY raw JSON:
{
  "page_types": ["login page", "dashboard", etc],
  "ui_elements": ["buttons", "menus", "forms"],
  "navigation_flow": ["step 1 observed", "step 2"],
  "forms": [{"name": "form name", "fields": ["field1"]}],
  "clickable_elements": ["element descriptions"],
  "software_detected": ["app or website name"]
}""",
    "ads": """Analyze this for advertising and product content.
Return ONLY raw JSON:
{
  "products": ["product names"],
  "brands": ["brand names"],
  "cta": "call to action text",
  "promo_elements": ["discount codes", "offers"],
  "ad_format": "format description",
  "target_audience": "description"
}""",
    "workflow": """Extract the step-by-step workflow shown in this video.
Focus on: what actions are performed, in what order, on what elements.
Return ONLY raw JSON:
{
  "goal": "what task is being completed",
  "steps": [
    {"order": 1, "action": "click/type/navigate", "target": "element or url", "timestamp": "00:00"}
  ],
  "total_steps": 0,
  "tools_used": ["app or website names"]
}""",
}


class VisionEngine:
    def __init__(self, caps):
        self.caps = caps
        self._model     = None
        self._processor = None
        self._loaded    = False
        self._failed    = False
        self._model_id  = None

    def _load(self, force_small: bool = False):
        if self._loaded or self._failed:
            return

        try:
            import torch
            from transformers import AutoProcessor, AutoModelForImageTextToText
            from config import SMOLVLM2_LARGE, SMOLVLM2_SMALL

            if force_small or self.caps.vision_model in ("small", "cpu_small"):
                model_id = SMOLVLM2_SMALL
            else:
                model_id = SMOLVLM2_LARGE

            self._model_id = model_id
            device = "cuda" if self.caps.has_cuda else "cpu"
            dtype  = torch.bfloat16

            attn_impl = (
                "flash_attention_2"
                if self.caps.has_flash_attn and self.caps.has_cuda
                else "eager"
            )

            log.info(f"Loading {model_id} | device={device} | attn={attn_impl}")

            self._processor = AutoProcessor.from_pretrained(model_id)
            self._model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                torch_dtype=dtype,
                _attn_implementation=attn_impl,
                low_cpu_mem_usage=True,
            ).to(device)

            self._loaded = True
            log.info(f"SmolVLM2 loaded ✓ [{model_id.split('/')[-1]}]")

            if self.caps.has_cuda:
                used_gb = torch.cuda.memory_allocated() / 1e9
                log.info(f"VRAM used: {used_gb:.1f}GB / {self.caps.vram_total_gb}GB")

        except Exception as e:
            if "out of memory" in str(e).lower() and not force_small:
                log.error("VRAM OOM on large model — cleaning up and retrying with 500M")
                
                # FIX: Proper memory cleanup
                del self._model
                del self._processor
                self._model = None
                self._processor = None
                gc.collect()
                
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                
                self._load(force_small=True)
            else:
                log.error(f"Model load failed: {e}")
                self._failed = True

    def _parse(self, text: str) -> dict:
        text = text.strip()
        for fence in ["```json", "```"]:
            if text.startswith(fence): text = text[len(fence):]
        if text.endswith("```"): text = text[:-3]
        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            s, e = text.find("{"), text.rfind("}") + 1
            if s != -1 and e > s:
                try: return json.loads(text[s:e])
                except Exception: pass
        log.warning("JSON parse failed — returning raw")
        return {"raw_response": text, "parse_error": True}

    def _infer(self, messages: list, max_tokens: int = 512) -> str:
        import torch
        inputs = self._processor.apply_chat_template(
            messages, add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors="pt",
        ).to(self._model.device, dtype=torch.bfloat16)

        with torch.no_grad():
            ids = self._model.generate(**inputs, do_sample=False, max_new_tokens=max_tokens)

        n = inputs["input_ids"].shape[1]
        text = self._processor.batch_decode(ids[:, n:], skip_special_tokens=True)[0]

        if self.caps.has_cuda: torch.cuda.empty_cache()
        return text

    def analyze_video(self, video_path: str, mode: str = "full") -> dict:
        self._load()
        if not self._loaded:
            return self._api_fallback_video(video_path, mode)
        if not self.caps.has_decord:
            return {"error": "decord_missing", "summary": "Install decord: pip install decord"}

        try:
            import torch
            from config import SMOLVLM2_LARGE

            messages = [{"role": "user", "content": [
                {"type": "video", "path": str(video_path)},
                {"type": "text",  "text": PROMPTS.get(mode, PROMPTS["full"])},
            ]}]
            
            text   = self._infer(messages)
            result = self._parse(text)
            result["provider"]   = f"smolvlm2_{self._model_id.split('/')[-1]}"
            result["input_type"] = "video"
            log.info(f"Video analysis done [{mode}] ✓")
            return result

        except torch.cuda.OutOfMemoryError:
            log.error("VRAM OOM during video inference")
            import torch
            from config import SMOLVLM2_LARGE
            
            # FIX: Prevent infinite retry loop if already on small model
            if SMOLVLM2_LARGE in self._model_id:
                del self._model
                self._model = None
                gc.collect()
                torch.cuda.empty_cache()
                self._load(force_small=True)
                
                if self._loaded:
                    log.info("Retrying with smaller model...")
                    return self.analyze_video(video_path, mode)

            return self._api_fallback_video(video_path, mode)
            
        except Exception as e:
            log.error(f"Video inference failed: {e}")
            return self._api_fallback_video(video_path, mode)

    def analyze_image(self, image_path: str, mode: str = "full") -> dict:
        self._load()
        if not self._loaded: return self._api_fallback_image(image_path, mode)
        try:
            messages = [{"role": "user", "content": [
                {"type": "image", "path": str(image_path)},
                {"type": "text",  "text": PROMPTS.get(mode, PROMPTS["full"])},
            ]}]
            result = self._parse(self._infer(messages))
            result["provider"] = f"smolvlm2_{self._model_id.split('/')[-1]}"
            log.info(f"Image analysis done [{mode}] ✓")
            return result
        except Exception as e:
            log.error(f"Image inference failed: {e}")
            return self._api_fallback_image(image_path, mode)

    def ask(self, file_path: str, question: str, is_video: bool = True) -> str:
        self._load()
        if not self._loaded: return "Vision model unavailable."
        if is_video and not self.caps.has_decord:
            return "Install decord for video: pip install decord"
        try:
            media = "video" if is_video else "image"
            messages = [{"role": "user", "content": [
                {"type": media, "path": str(file_path)},
                {"type": "text", "text": question},
            ]}]
            return self._infer(messages, max_tokens=256).strip()
        except Exception as e:
            log.error(f"ask() failed: {e}")
            return f"Error: {e}"

    def _api_fallback_video(self, video_path: str, mode: str) -> dict:
        if not self.caps.has_openrouter:
            return {"error": "vision_unavailable", "provider": "none"}
        try:
            import subprocess, tempfile
            tmp = Path(tempfile.mkdtemp())
            subprocess.run(["ffmpeg", "-i", str(video_path), "-vf",
                            "fps=1/30,scale=640:-1", "-frames:v", "3",
                            "-y", str(tmp / "f_%02d.jpg")],
                           capture_output=True, timeout=30)
            frames = sorted(tmp.glob("f_*.jpg"))
            if frames: return self._api_fallback_image(str(frames[-1]), mode) # Send last frame as representative
        except Exception: pass
        return {"error": "all_vision_failed", "provider": "none"}

    def _api_fallback_image(self, image_path: str, mode: str) -> dict:
        try:
            import requests
            with open(image_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
                         "Content-Type": "application/json"},
                json={"model": "meta-llama/llama-3.2-11b-vision-instruct:free",
                      "messages": [{"role": "user", "content": [
                          {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                          {"type": "text", "text": PROMPTS.get(mode, PROMPTS["fast"])},
                      ]}], "max_tokens": 512},
                timeout=60,
            )
            resp.raise_for_status()
            result = self._parse(resp.json()["choices"][0]["message"]["content"])
            result["provider"] = "openrouter_fallback"
            return result
        except Exception as e:
            return {"error": str(e), "provider": "all_failed"}