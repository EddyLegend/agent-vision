"""
Download SmolVLM2-500M model weights locally.
Run this script once after cloning the repo.

Requirements:
  pip install huggingface_hub
  hf auth login  (optional but recommended for faster downloads)
"""
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DISABLE_XET"] = "1"

from huggingface_hub import hf_hub_download

repo_id = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
local_dir = "./models/smolvlm-500m"

# Only download essential PyTorch files (ignores massive ONNX files)
files = [
    "model.safetensors",
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
]

os.makedirs(local_dir, exist_ok=True)

print(f"Downloading model to {local_dir}...")
print("If the download times out, just re-run this script. It will resume.\n")

for f in files:
    print(f"⏳ Downloading: {f}")
    try:
        hf_hub_download(
            repo_id=repo_id,
            filename=f,
            local_dir=local_dir,
            resume_download=True,
        )
        print(f"✅ Done: {f}")
    except Exception as e:
        print(f"❌ Error with {f}: {e} — Just re-run the script!")

print("\n🎉 Model download complete!")
