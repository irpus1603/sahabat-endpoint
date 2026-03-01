#!/usr/bin/env python3
"""
Download Qwen3-30B-A3B Q4_K_M GGUF from HuggingFace to a local folder
Model size: ~18.6GB
"""

from huggingface_hub import snapshot_download
import os

# ── Configuration ──────────────────────────────────────────
MODEL_ID    = "Qwen/Qwen3-30B-A3B-GGUF"
LOCAL_DIR   = "./models/Qwen3-30B-A3B-GGUF"   # change to your preferred path
HF_TOKEN    = ""


# ───────────────────────────────────────────────────────────

def download_model():
    os.makedirs(LOCAL_DIR, exist_ok=True)

    print(f"📦 Downloading {MODEL_ID} (Q4_K_M only)")
    print(f"📁 Saving to: {os.path.abspath(LOCAL_DIR)}")
    print(f"💾 Expected size: ~18.6GB\n")

    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=LOCAL_DIR,
        token=HF_TOKEN,
        allow_patterns=[
            "*Q4_K_M*",   # only Q4_K_M quantization
            "*.json",     # config files
            "*.txt",      # tokenizer/readme files
            "*.md",       # model card
        ],
    )

    print(f"\n✅ Download complete! Model saved to: {os.path.abspath(LOCAL_DIR)}")
    print(f"📂 Files downloaded:")
    for f in os.listdir(LOCAL_DIR):
        size_gb = os.path.getsize(os.path.join(LOCAL_DIR, f)) / (1024**3)
        print(f"   {f} ({size_gb:.2f} GB)")

if __name__ == "__main__":
    download_model()