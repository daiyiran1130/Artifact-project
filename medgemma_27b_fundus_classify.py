"""
Color fundus image classification — 3-prompt experiment over all artifact sub-folders.
Model: google/medgemma-1.5-4b-it (4B, 4-bit quantized via bitsandbytes)

Folder structure expected:
  /root/autodl-tmp/artifact/{folder_name}/nolabel/*.jpg

For each prompt (1/2/3) and each folder, results are saved as:
  {folder}/{prompt_idx}_{folder_name}.json
  e.g. mediumcolor/1_mediumcolor.json

All model output is stored as-is (no token limit, no truncation).

Download the model first:
  modelscope download --model google/medgemma-1.5-4b-it \\
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

# ── 全局配置 ─────────────────────────────────────────────────────────────
NOLABEL_DIR  = Path("/root/autodl-tmp/nolabel")
MODEL_PATH   = Path("/root/autodl-tmp/modelscope_cache/google/medgemma-1.5-4b-it")

# RTX PRO 6000 显存 96GB，4-bit 模型占约 14GB，剩余约 82GB 可用于 batch
# Prompt 3 会生成较长的推理链，如遇 OOM 可将 BATCH_SIZE 减小为 4
BATCH_SIZE = 8

# ── 3 种实验提示词 ───────────────────────────────────────────────────────
PROMPT_1 = (
    "You are an ophthalmology expert.\n"
    "You are given a color fundus image.\n"
    "Classify the image into one of the following categories:\n"
    "\"Normal\", \"Diabetic Retinopathy\", \"Age-related Macular Degeneration\", \"Glaucoma\".\n"
    "Only provide the single correct category as the answer.\n"
    "Do not provide explanations or additional information.\n"
    "This classification is for research reference only, not for clinical diagnosis."
)

PROMPT_2 = (
    "You are an ophthalmology expert.\n"
    "You are given a color fundus image.\n"
    "Classify the image into one of the following categories:\n"
    "\"Normal\", \"Diabetic Retinopathy\", \"Age-related Macular Degeneration\", \"Glaucoma\".\n"
    "This classification is for research reference only, not for clinical diagnosis."
)

PROMPT_3 = (
    "You are an ophthalmology expert.\n"
    "You are given a color fundus image.\n"
    "Classify the image into one of the following categories:\n"
    "\"Normal\", \"Diabetic Retinopathy\", \"Age-related Macular Degeneration\", \"Glaucoma\".\n"
    "This classification is for research reference only, not for clinical diagnosis.\n"
    "Describe your reasoning in steps."
)

PROMPTS = {1: PROMPT_1, 2: PROMPT_2, 3: PROMPT_3}

# ── 模型加载 ─────────────────────────────────────────────────────────────
print(f"Loading model from : {MODEL_PATH}")
print(f"Directory exists   : {MODEL_PATH.is_dir()}")

if not MODEL_PATH.is_dir():
    raise FileNotFoundError(
        f"{MODEL_PATH} does not exist.\n"
        "Please download the model first:\n"
        "  modelscope download --model google/medgemma-1.5-4b-it "
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
processor.tokenizer.padding_side = "left"   # batch 生成必须 left padding
print("Model loaded.\n")


# ── 工具函数 ─────────────────────────────────────────────────────────────
def extract_number(filename: str) -> int | None:
    stem  = Path(filename).stem
    match = re.search(r"\d+", stem)
    return int(match.group()) if match else None


def query_model_batch(image_paths: list, prompt: str) -> list:
    """
    batch_size 张图片一次前向传播，返回同数量的结果字符串列表。
    不限制 max_new_tokens，保存模型完整输出。
    """
    images = [Image.open(p).convert("RGB") for p in image_paths]

    all_messages = [
        [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text",  "text": prompt},
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
        decoded    = processor.decode(new_tokens, skip_special_tokens=True)
        results.append(decoded.strip())
    return results


def query_model_single(image_path: Path, prompt: str) -> str:
    """OOM 时的单张图片回退，同样不限制输出长度。"""
    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text",  "text": prompt},
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
    return processor.decode(new_tokens, skip_special_tokens=True).strip()


def save_results(path: Path, results: dict) -> None:
    """按编号升序写入 JSON，每条记录占一行。"""
    with open(path, "w", encoding="utf-8") as f:
        f.write("{\n")
        items = sorted(results.items(), key=lambda kv: int(kv[0]))
        for i, (k, v) in enumerate(items):
            comma = "," if i < len(items) - 1 else ""
            f.write(f'  "{k}": {json.dumps(v, ensure_ascii=False)}{comma}\n')
        f.write("}\n")


def process_folder(folder: Path, prompt_idx: int, prompt: str,
                   numbered: list) -> None:
    """
    对同一组图片（numbered）用指定 prompt 做推理。
    图片来自 folder/nolabel/，结果写入 folder/{prompt_idx}_{folder.name}.json。
    """
    output_file = folder / f"{prompt_idx}_{folder.name}.json"
    print(f"  Output -> {output_file}")

    shuffled = numbered[:]
    random.shuffle(shuffled)

    results = {}
    save_results(output_file, results)

    total = len(shuffled)
    idx   = 0
    while idx < total:
        batch       = shuffled[idx: idx + BATCH_SIZE]
        batch_nums  = [n for n, _ in batch]
        batch_paths = [p for _, p in batch]
        idx        += len(batch)

        print(f"  [{idx}/{total}] {len(batch)} image(s): "
              f"{[p.name for p in batch_paths]} ...", flush=True)

        try:
            raw_list = query_model_batch(batch_paths, prompt)
            for num, raw in zip(batch_nums, raw_list):
                results[str(num)] = raw
                preview = raw[:120].replace("\n", " ")
                print(f"    {num} -> {preview}{'...' if len(raw) > 120 else ''}")
        except torch.cuda.OutOfMemoryError:
            print("  [OOM] Falling back to single-image inference for this batch.")
            torch.cuda.empty_cache()
            for num, img_path in zip(batch_nums, batch_paths):
                print(f"    {img_path.name} ...", end=" ", flush=True)
                try:
                    raw = query_model_single(img_path, prompt)
                    results[str(num)] = raw
                    preview = raw[:120].replace("\n", " ")
                    print(f"-> {preview}{'...' if len(raw) > 120 else ''}")
                except Exception as e:
                    print(f"\n    [ERROR] {e}")
                    results[str(num)] = f"error: {e}"
        except Exception as e:
            print(f"\n  [ERROR] {e}")
            for num in batch_nums:
                results[str(num)] = f"error: {e}"

        save_results(output_file, results)

    print(f"  Saved {len(results)} results -> {output_file}\n")


# ── 主流程 ─────────────────────────────────────────────────────────────
def main():
    # 直接从 nolabel 文件夹读取图片
    if not NOLABEL_DIR.is_dir():
        print(f"[ERROR] Directory not found: {NOLABEL_DIR}")
        return

    image_files = (
        list(NOLABEL_DIR.glob("*.jpg"))
        + list(NOLABEL_DIR.glob("*.jpeg"))
        + list(NOLABEL_DIR.glob("*.png"))
    )
    numbered = []
    for p in image_files:
        n = extract_number(p.name)
        if n is not None:
            numbered.append((n, p))
        else:
            print(f"  [SKIP] Cannot extract number from: {p.name}")

    if not numbered:
        print(f"No valid images found in {NOLABEL_DIR}")
        return

    print(f"Found {len(numbered)} images in {NOLABEL_DIR}")
    print(f"Prompts    : {list(PROMPTS.keys())}")
    print(f"Batch size : {BATCH_SIZE}\n")

    total_runs  = len(PROMPTS)
    run_counter = 0

    for prompt_idx, prompt in PROMPTS.items():
        run_counter += 1
        print(f"{'='*60}")
        print(f"[Run {run_counter}/{total_runs}] Prompt {prompt_idx} | {len(numbered)} images")
        print(f"{'='*60}")
        process_folder(NOLABEL_DIR, prompt_idx, prompt, numbered)

    print("All experiments done.")


if __name__ == "__main__":
    main()
