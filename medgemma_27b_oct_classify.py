"""
OCT image classification — batch over all artifact sub-folders.
Model: google/medgemma-27b-it (27B, 4-bit quantized via bitsandbytes)

Download the model first:
  modelscope download --model google/medgemma-27b-it \\
      --cache_dir /root/autodl-tmp/modelscope_cache
"""
import os
os.environ["HF_HOME"]            = "/root/autodl-tmp/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/root/autodl-tmp/hf_cache"
os.environ["MODELSCOPE_CACHE"]   = "/root/autodl-tmp/modelscope_cache"

import re
import json
import random
from pathlib import Path
import torch
from PIL import Image

# ── 全局配置 ──────────────────────────────────────────────────────────────
ARTIFACT_DIR = Path("/root/autodl-tmp/artifact")
MODEL_PATH   = Path("/root/autodl-tmp/modelscope_cache/google/medgemma-27b-it")

# Pro 6000 显存 48GB，4-bit 模型占纩14GB，剩余约34GB可用于 batch
# 如果 OOM 可适当减小
BATCH_SIZE = 4

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
        "--cache_dir /root/autodl-tmp/modelscope_cache"
    )

from transformers import AutoProcessor, BitsAndBytesConfig

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
# batch 生成必须用 left padding
processor.tokenizer.padding_side = "left"
print("Model loaded.\n")

# ── 工具函数 ──────────────────────────────────────────────────────────────
def extract_number(filename: str) -> int | None:
    stem  = Path(filename).stem
    match = re.search(r"\d+", stem)
    return int(match.group()) if match else None


def query_model_batch(image_paths: list) -> list:
    """batch_size 张图片一次前向传播，返回同数量的结果字符串列表。"""
    images = [Image.open(p).convert("RGB") for p in image_paths]

    all_messages = [
        [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text",  "text": PROMPT},
        ]}]
        for img in images
    ]

    inputs = processor.apply_chat_template(
        all_messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    ).to(model.device, dtype=torch.bfloat16)

    input_len = inputs["input_ids"].shape[-1]
    max_ctx   = getattr(model.config, "max_position_embeddings", 8192)
    remaining = max(1, max_ctx - input_len)

    with torch.inference_mode():
        generations = model.generate(
            **inputs,
            max_new_tokens=remaining,
            do_sample=False,
        )

    results = []
    for gen in generations:
        new_tokens = gen[input_len:]
        decoded = processor.decode(new_tokens, skip_special_tokens=True)
        results.append(decoded.strip().lower())
    return results


def query_model_single(image_path: Path) -> str:
    """OOM 时的单张图片回退。"""
    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text",  "text": PROMPT},
    ]}]
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
    return processor.decode(new_tokens, skip_special_tokens=True).strip().lower()


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
    print(f"  Found {len(numbered)} images. Batch size = {BATCH_SIZE}. Starting classification...")

    results = {}
    save_results(output_file, results)

    total = len(numbered)
    idx   = 0
    while idx < total:
        batch       = numbered[idx: idx + BATCH_SIZE]
        batch_nums  = [n for n, _ in batch]
        batch_paths = [p for _, p in batch]
        idx        += len(batch)

        print(f"  [{idx}/{total}] Processing {len(batch)} image(s): "
              f"{[p.name for p in batch_paths]} ...", flush=True)

        try:
            raw_list = query_model_batch(batch_paths)
            for num, raw in zip(batch_nums, raw_list):
                results[str(num)] = raw
                print(f"    {num} -> {raw[:80]}")
        except torch.cuda.OutOfMemoryError:
            # OOM 时逑退到单张处理
            print(f"  [OOM] Falling back to single-image inference for this batch.")
            torch.cuda.empty_cache()
            for num, img_path in zip(batch_nums, batch_paths):
                print(f"    {img_path.name} ...", end=" ", flush=True)
                try:
                    raw = query_model_single(img_path)
                    results[str(num)] = raw
                    print(f"-> {raw[:80]}")
                except Exception as e:
                    print(f"\n    [ERROR] {e}")
                    results[str(num)] = f"error: {e}"
        except Exception as e:
            print(f"\n  [ERROR] {e}")
            for num in batch_nums:
                results[str(num)] = f"error: {e}"

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
