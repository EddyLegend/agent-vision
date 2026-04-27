"""vision_engine.py — SmolVLM2 Vision Engine + API Fallbacks"""
import json, logging, os, base64, gc
from pathlib import Path

log = logging.getLogger("vision_engine")

PROMPTS = {
    "full": """Analyze this video/image completely. Return ONLY raw JSON (no markdown): {"summary": "...", "scenes": [{"timestamp": "00:00", "description": "..."}], "objects": [], "text_visible": [], "actions": [], "ui_elements": [], "entities": {"people": [], "brands": []}}""",
    "fast": """Quick summary. Return ONLY raw JSON: {"summary": "...", "key_elements": []}""",
    "ui": """Analyze ONLY UI/software. Return ONLY raw JSON: {"page_types": [], "ui_elements": [], "navigation_flow": [], "clickable_elements": [], "software_detected": []}""",
    "ads": """Analyze for ads. Return ONLY raw JSON: {"products": [], "brands": [], "cta": "", "target_audience": ""}""",
    "workflow": """Extract workflow steps. Return ONLY raw JSON: {"goal": "...", "steps": [{"order": 1, "action": "...", "target": "..."}], "tools_used": []}""",
}

class VisionEngine:
    def __init__(self, caps, quota):
        self.caps = caps
        self.quota = quota
        self._model = None
        self._processor = None
        self._loaded = False
        self._failed = False
        self._model_id = None

    def _load(self, force_small=False):
        if self._loaded or self._failed: return
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForImageTextToText
            from config import SMOLVLM2_LARGE, SMOLVLM2_SMALL
            model_id = SMOLVLM2_SMALL if (force_small or self.caps.vision_model in ("small", "cpu_small")) else SMOLVLM2_LARGE
            self._model_id = model_id
            device = "cuda" if self.caps.has_cuda else "cpu"
            dtype = torch.bfloat16
            attn_impl = "flash_attention_2" if (self.caps.has_flash_attn and self.caps.has_cuda) else "eager"
            log.info(f"Loading {model_id} | device={device}")
            self._processor = AutoProcessor.from_pretrained(model_id)
            self._model = AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype=dtype, _attn_implementation=attn_impl, low_cpu_mem_usage=True).to(device)
            self._loaded = True
            if self.caps.has_cuda: torch.cuda.empty_cache()
        except Exception as e:
            if "out of memory" in str(e).lower() and not force_small:
                del self._model; del self._processor; gc.collect()
                try: import torch; torch.cuda.empty_cache()
                except: pass
                self._load(force_small=True)
            else: self._failed = True

    def _parse(self, text):
        text = text.strip()
        for fence in ["```json", "```"]:
            if text.startswith(fence): text = text[len(fence):]
        if text.endswith("```"): text = text[:-3]
        try: return json.loads(text.strip())
        except: pass
        s, e = text.find("{"), text.rfind("}") + 1
        if s != -1 and e > s:
            try: return json.loads(text[s:e])
            except: pass
        return {"raw_response": text, "parse_error": True}

    def _infer(self, messages, max_tokens=512):
        import torch
        inputs = self._processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(self._model.device, dtype=torch.bfloat16)
        with torch.no_grad(): ids = self._model.generate(**inputs, do_sample=False, max_new_tokens=max_tokens)
        text = self._processor.batch_decode(ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        if self.caps.has_cuda: torch.cuda.empty_cache()
        return text

    def analyze_video(self, video_path, mode="full"):
        self._load()
        if not self._loaded: return self._api_fallback_video(video_path, mode)
        try:
            messages = [{"role": "user", "content": [{"type": "video", "path": str(video_path)}, {"type": "text", "text": PROMPTS.get(mode, PROMPTS["full"])}]}]
            result = self._parse(self._infer(messages))
            result["provider"] = f"smolvlm2_{self._model_id.split('/')[-1]}"
            return result
        except Exception as e:
            log.error(f"Video inference failed: {e}")
            return self._api_fallback_video(video_path, mode)

    def analyze_image(self, image_path, mode="full"):
        self._load()
        if not self._loaded: return self._api_fallback_image(image_path, mode)
        try:
            messages = [{"role": "user", "content": [{"type": "image", "path": str(image_path)}, {"type": "text", "text": PROMPTS.get(mode, PROMPTS["full"])}]}]
            result = self._parse(self._infer(messages))
            result["provider"] = f"smolvlm2_{self._model_id.split('/')[-1]}"
            return result
        except Exception as e:
            log.error(f"Image inference failed: {e}")
            return self._api_fallback_image(image_path, mode)

    def ask(self, file_path, question, is_video=True):
        self._load()
        if not self._loaded: return "Vision model unavailable."
        try:
            media = "video" if is_video else "image"
            messages = [{"role": "user", "content": [{"type": media, "path": str(file_path)}, {"type": "text", "text": question}]}]
            return self._infer(messages, max_tokens=256).strip()
        except Exception as e: return f"Error: {e}"

    # --- FALLBACKS ---
    def _api_fallback_video(self, video_path, mode):
        if self.caps.has_gemini and self.quota.can_use("gemini_vision"):
            res = self._gemini_fallback_video(video_path, mode)
            if res and "error" not in res: self.quota.record("gemini_vision"); return res
        if self.caps.has_openrouter and self.quota.can_use("openrouter"):
            res = self._openrouter_fallback_video(video_path, mode)
            if res and "error" not in res: self.quota.record("openrouter"); return res
        return {"error": "all_vision_failed", "provider": "none"}

    def _api_fallback_image(self, image_path, mode):
        if self.caps.has_gemini and self.quota.can_use("gemini_vision"):
            res = self._gemini_fallback_image(image_path, mode)
            if res and "error" not in res: self.quota.record("gemini_vision"); return res
        if self.caps.has_openrouter and self.quota.can_use("openrouter"):
            res = self._openrouter_fallback_image(image_path, mode)
            if res and "error" not in res: self.quota.record("openrouter"); return res
        return {"error": "all_vision_failed", "provider": "none"}

    def _gemini_fallback_image(self, image_path, mode):
        try:
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            from PIL import Image
            img = Image.open(image_path)
            response = genai.GenerativeModel('gemini-1.5-flash').generate_content([PROMPTS.get(mode, PROMPTS["fast"]), img])
            result = self._parse(response.text); result["provider"] = "gemini_1.5_flash"; return result
        except Exception as e: return {"error": str(e)}

    def _gemini_fallback_video(self, video_path, mode):
        try:
            import google.generativeai as genai, time
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            vf = genai.upload_file(path=video_path)
            while vf.state.name == "PROCESSING": time.sleep(2); vf = genai.get_file(vf.name)
            if vf.state.name == "FAILED": raise ValueError("Failed")
            response = genai.GenerativeModel('gemini-1.5-flash').generate_content([PROMPTS.get(mode, PROMPTS["fast"]), vf])
            result = self._parse(response.text); result["provider"] = "gemini_1.5_flash"
            try: genai.delete_file(vf.name)
            except: pass
            return result
        except Exception as e: return {"error": str(e)}

    def _openrouter_fallback_video(self, video_path, mode):
        try:
            import subprocess, tempfile
            tmp = Path(tempfile.mkdtemp())
            subprocess.run(["ffmpeg", "-i", str(video_path), "-vf", "fps=1/30,scale=640:-1", "-frames:v", "3", "-y", str(tmp / "f_%02d.jpg")], capture_output=True, timeout=30)
            frames = sorted(tmp.glob("f_*.jpg"))
            if frames: return self._openrouter_fallback_image(str(frames[-1]), mode)
        except: pass
        return {"error": "openrouter_video_failed"}

    def _openrouter_fallback_image(self, image_path, mode):
        try:
            import requests
            with open(image_path, "rb") as f: b64 = base64.b64encode(f.read()).decode()
            resp = requests.post("https://openrouter.ai/api/v1/chat/completions", headers={"Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}"}, json={"model": "meta-llama/llama-3.2-11b-vision-instruct:free", "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}, {"type": "text", "text": PROMPTS.get(mode, PROMPTS["fast"])}]}]}, timeout=60)
            result = self._parse(resp.json()["choices"][0]["message"]["content"]); result["provider"] = "openrouter_fallback"; return result
        except Exception as e: return {"error": str(e)}