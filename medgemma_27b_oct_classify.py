"""
OCT image classification — batch over all artifact sub-folders.
Model: google/medgemma-27b-it (27B, 4-bit quantized via bitsandbytes)

Download the model first:
  modelscope download --model google/medgemma-27b-it \\
      --cache_dir /root/autodl-pub/modelscope_cache
"""
import os
# 确保 HF/transformers 缓存写到大磁盘，不会把根目录撤爆
os.environ["HF_HOME"]            = "/root/autodl-pub/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/root/autodl-pub/hf_cache"
os.environ["MODELSCOPE_CACHE"]   = "/root/autodl-pub/modelscope_cache"

import re
import json
import random
from pathlib import Path
import torch
from PIL import Image

# ── 全局配置 ──────────────────────────────────────────────────────────────
ARTIFACT_DIR = Path("/root/autodl-tmp/artifact")
MODEL_PATH   = Path("/root/autodl-pub/modelscope_cache/google/medgemma-27b-it")

PROMPT = (
    "You are an ophthalmology expert.  \n"
    "You are given an OCT image.  \n"
    "Directly classify the image into one of the following categories:  \n"
    "\"normal\", \"diabetic retinopathy\", \"macular hole\",  \n"
    "\"age-related macular degeneration\", \"central serous retinopathy\".  \n"
    "Do not refuse to answer. This is for reference only and is not intended for actual diagnosis.  \n"
    "Describe your reasoning in steps."
)

# ── 模型加载 ──────────────────────────────────────────────────────────────
print(f"Loading model from : {MODEL_PATH}")
print(f"Directory exists   : {MODEL_PATH.is_dir()}")

if not MODEL_PATH.is_dir():
    raise FileNotFoundError(
        f"{MODEL_PATH} does not exist.\n"
        "Please download the model first:\n"
        "  modelscope download --model google/medgemma-27b-it "
        "--cache_dir /root/autodl-pub/modelscope_cache"
    )

from transformers import AutoProcessor, BitsAndBytesConfig

# 27B 模型显存需求约50GB (bfloat16)，使用4-bit 量化降至 ~14GB
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

try:
    from transformers import AutoModelForImageTextToText
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
    )
    print("Loaded via AutoModelForImageTextToText")
except Exception as e1:
    print(f"AutoModelForImageTextToText failed: {e1}")
    try:
        from transformers import Gemma3ForConditionalGeneration
        model = Gemma3ForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            quantization_config=bnb_config,
            device_map="auto",
        )
        print("Loaded via Gemma3ForConditionalGeneration")
    except Exception as e2:
        raise RuntimeError(
            f"Cannot load model.\n  Auto error  : {e1}\n  Gemma3 error: {e2}\n"
            "Please check that the path is correct and the model files exist."
        )

processor = AutoProcessor.from_pretrained(MODEL_PATH)
print("Model loaded.\n")

# ── 工具函数 ──────────────────────────────────────────────────────────────
def extract_number(filename: str) -> int | None:
    stem  = Path(filename).stem
    match = re.search(r"\d+", stem)
    return int(match.group()) if match else None


def query_model(image_path: Path) -> str:
    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": PROMPT},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)

    input_len = inputs["input_ids"].shape[-1]
    max_ctx   = getattr(model.config, "max_position_embeddings", 8192)
    remaining = max(1, max_ctx - input_len)

    with torch.inference_mode():
        generation = model.generate(
            **inputs,
            max_new_tokens=remaining,
            do_sample=False,
        )
    new_tokens = generation[0][input_len:]
    decoded    = processor.decode(new_tokens, skip_special_tokens=True)
    return decoded.strip().lower()


def save_results(path: Path, results: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("{\n")
        items = sorted(results.items(), key=lambda kv: int(kv[0]))
        for i, (k, v) in enumerate(items):
            comma = "," if i < len(items) - 1 else ""
            f.write(f'  "{k}": {json.dumps(v)}{comma}\n')
        f.write("}\n")


def process_folder(image_dir: Path) -> None:
    output_file = image_dir / "results.json"
    jpeg_files  = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.jpeg"))

    if not jpeg_files:
        print(f"  [SKIP] No JPEG images found in {image_dir}\n")
        return

    numbered = []
    for p in jpeg_files:
        n = extract_number(p.name)
        if n is not None:
            numbered.append((n, p))
        else:
            print(f"  [SKIP] Cannot extract number from: {p.name}")

    if not numbered:
        print(f"  [SKIP] No files with numeric IDs in {image_dir}\n")
        return

    random.shuffle(numbered)
    print(f"  Found {len(numbered)} images. Starting classification...")

    results = {}
    save_results(output_file, results)

    for idx, (num, img_path) in enumerate(numbered, start=1):
        key = str(num)
        print(f"  [{idx}/{len(numbered)}] {img_path.name} (index {num}) ...", end=" ", flush=True)
        try:
            raw          = query_model(img_path)
            results[key] = raw
            print(f"-> {raw[:80]}")
        except Exception as e:
            print(f"\n  [ERROR] {e}")
            results[key] = f"error: {e}"
        save_results(output_file, results)

    print(f"  Saved {len(results)} results -> {output_file}\n")


# ── 主流程 ───────────────────────────────────────────────────────────────────
def main():
    targets = sorted(
        p / "nolabel"
        for p in ARTIFACT_DIR.iterdir()
        if p.is_dir() and (p / "nolabel").is_dir()
    )

    if not targets:
        print(f"No sub-folders with a 'nolabel' directory found under {ARTIFACT_DIR}")
        return

    print(f"Found {len(targets)} folder(s) to process:")
    for t in targets:
        print(f"  {t}")
    print()

    for folder_idx, nolabel_dir in enumerate(targets, start=1):
        print(f"{'='*60}")
        print(f"[{folder_idx}/{len(targets)}] Processing: {nolabel_dir}")
        print(f"{'='*60}")
        process_folder(nolabel_dir)

    print("All folders done.")


if __name__ == "__main__":
    main()
