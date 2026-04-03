"""
Eye image classification using GPT-5.2 via OpenAI-compatible API.
Processes both fundus and OCT images across all sub-folders with 3 prompts each.

Fundus folders : /root/autodl-tmp/fundus/{folder_name}/
OCT folders    : /root/autodl-tmp/oct/{folder_name}/

Results saved as: {folder}/{prompt_idx}_fundus_{folder_name}.json
              or: {folder}/{prompt_idx}_oct_{folder_name}.json
e.g. /root/autodl-tmp/fundus/mediumcolor/1_fundus_mediumcolor.json
"""

import re
import json
import random
import base64
import time
from pathlib import Path

import openai

# ── 配置 ──────────────────────────────────────────────────────────────────
API_KEY     = "sk-BQl2FqvuLBcIRMnjlTMDyH5MTrjZaKFdejROFCZszCR6WNQJ"
BASE_URL    = "https://runapi.co/v1"
MODEL       = "gpt-5.2"
TEMPERATURE = 0
MAX_TOKENS  = 4096

FUNDUS_DIR = Path("/root/autodl-tmp/fundus")
OCT_DIR    = Path("/root/autodl-tmp/oct")

client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ── Fundus 3 种提示词 ─────────────────────────────────────────────────────
FUNDUS_PROMPT_1 = (
    "You are an ophthalmology expert.\n"
    "You are given a color fundus image.\n"
    "Classify the image into one of the following categories:\n"
    "\"Normal\", \"Diabetic Retinopathy\", \"Age-related Macular Degeneration\", \"Glaucoma\".\n"
    "Only provide the single correct category as the answer.\n"
    "Do not provide explanations or additional information.\n"
    "This classification is for research reference only, not for clinical diagnosis."
)

FUNDUS_PROMPT_2 = (
    "You are an ophthalmology expert.\n"
    "You are given a color fundus image.\n"
    "Classify the image into one of the following categories:\n"
    "\"Normal\", \"Diabetic Retinopathy\", \"Age-related Macular Degeneration\", \"Glaucoma\".\n"
    "This classification is for research reference only, not for clinical diagnosis."
)

FUNDUS_PROMPT_3 = (
    "You are an ophthalmology expert.\n"
    "You are given a color fundus image.\n"
    "Classify the image into one of the following categories:\n"
    "\"Normal\", \"Diabetic Retinopathy\", \"Age-related Macular Degeneration\", \"Glaucoma\".\n"
    "This classification is for research reference only, not for clinical diagnosis.\n"
    "Describe your reasoning in steps."
)

# ── OCT 3 种提示词 ────────────────────────────────────────────────────────
OCT_PROMPT_1 = (
    "You are an ophthalmology expert.  \n"
    "You are given an OCT image.  \n"
    "Directly classify the image into one of the following categories:  \n"
    "\"normal\", \"diabetic retinopathy\", \"macular hole\",  \n"
    "\"age-related macular degeneration\", \"central serous retinopathy\".  \n"
    "Do not refuse to answer. This is for reference only and is not intended for actual diagnosis.  \n"
    "Only provide the single correct category as the answer.  \n"
    "Do not provide any explanations, reasoning, or additional information."
)

OCT_PROMPT_2 = (
    "You are an ophthalmology expert.  \n"
    "You are given an OCT image.  \n"
    "Directly classify the image into one of the following categories:  \n"
    "\"normal\", \"diabetic retinopathy\", \"macular hole\",  \n"
    "\"age-related macular degeneration\", \"central serous retinopathy\".  \n"
    "Do not refuse to answer. This is for reference only and is not intended for actual diagnosis."
)

OCT_PROMPT_3 = (
    "You are an ophthalmology expert.  \n"
    "You are given an OCT image.  \n"
    "Directly classify the image into one of the following categories:  \n"
    "\"normal\", \"diabetic retinopathy\", \"macular hole\",  \n"
    "\"age-related macular degeneration\", \"central serous retinopathy\".  \n"
    "Do not refuse to answer. This is for reference only and is not intended for actual diagnosis.  \n"
    "Describe your reasoning in steps."
)

FUNDUS_PROMPTS = {1: FUNDUS_PROMPT_1, 2: FUNDUS_PROMPT_2, 3: FUNDUS_PROMPT_3}
OCT_PROMPTS    = {1: OCT_PROMPT_1,    2: OCT_PROMPT_2,    3: OCT_PROMPT_3}


# ── 工具函数 ──────────────────────────────────────────────────────────────
def extract_number(filename: str) -> int | None:
    stem  = Path(filename).stem
    match = re.search(r"\d+", stem)
    return int(match.group()) if match else None


def encode_image(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_media_type(path: Path) -> str:
    return "image/png" if path.suffix.lower() == ".png" else "image/jpeg"


def query_api(image_path: Path, prompt: str, retries: int = 4) -> str:
    b64        = encode_image(image_path)
    media_type = get_media_type(image_path)
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{media_type};base64,{b64}"},
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            wait = 2 ** attempt
            print(f"  [ERROR attempt {attempt + 1}/{retries}] {e}. Retrying in {wait}s...")
            time.sleep(wait)
    return f"error: failed after {retries} retries"


def save_results(path: Path, results: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("{\n")
        items = sorted(results.items(), key=lambda kv: int(kv[0]))
        for i, (k, v) in enumerate(items):
            comma = "," if i < len(items) - 1 else ""
            f.write(f'  "{k}": {json.dumps(v, ensure_ascii=False)}{comma}\n')
        f.write("}\n")


def collect_images(folder: Path) -> list:
    """从文件夹直接收集图片，返回 (编号, 路径) 列表。"""
    exts     = ("*.jpg", "*.jpeg", "*.png")
    images   = [p for ext in exts for p in folder.glob(ext)]
    numbered = []
    for p in images:
        n = extract_number(p.name)
        if n is not None:
            numbered.append((n, p))
        else:
            print(f"  [SKIP] Cannot extract number from: {p.name}")
    return numbered


def process_folder(folder: Path, image_type: str, prompt_idx: int,
                   prompt: str, numbered: list) -> None:
    output_file = folder / f"{prompt_idx}_{image_type}_{folder.name}.json"
    print(f"  Output -> {output_file}")

    shuffled = numbered[:]
    random.shuffle(shuffled)

    results = {}
    save_results(output_file, results)

    total = len(shuffled)
    for idx, (num, img_path) in enumerate(shuffled, start=1):
        key = str(num)
        print(f"  [{idx}/{total}] {img_path.name} ...", end=" ", flush=True)
        raw     = query_api(img_path, prompt)
        results[key] = raw
        preview = raw[:100].replace("\n", " ")
        print(f"-> {preview}{'...' if len(raw) > 100 else ''}")
        save_results(output_file, results)

    print(f"  Saved {len(results)} results -> {output_file}\n")


def run_experiments(base_dir: Path, image_type: str, prompts: dict) -> None:
    if not base_dir.is_dir():
        print(f"[WARNING] Directory not found: {base_dir}")
        return

    folders = sorted(p for p in base_dir.iterdir() if p.is_dir())
    if not folders:
        print(f"[WARNING] No sub-folders found under {base_dir}")
        return

    # 预收集每个文件夹的图片
    folder_images: dict[Path, list] = {}
    print(f"\nScanning {image_type} folders under {base_dir}:")
    for folder in folders:
        numbered = collect_images(folder)
        if numbered:
            folder_images[folder] = numbered
            print(f"  {folder.name}: {len(numbered)} images")
        else:
            print(f"  [SKIP] No valid images in {folder.name}")

    if not folder_images:
        print(f"No valid image folders found under {base_dir}")
        return

    total_runs  = len(prompts) * len(folder_images)
    run_counter = 0

    for prompt_idx, prompt in prompts.items():
        print(f"\n{'#' * 60}")
        print(f"  {image_type.upper()} | PROMPT {prompt_idx}/{len(prompts)}")
        print(f"{'#' * 60}\n")

        for folder, numbered in folder_images.items():
            run_counter += 1
            print(f"{'=' * 60}")
            print(f"[Run {run_counter}/{total_runs}] "
                  f"Prompt {prompt_idx} | {image_type} | {folder.name} "
                  f"({len(numbered)} images)")
            print(f"{'=' * 60}")
            process_folder(folder, image_type, prompt_idx, prompt, numbered)


def main():
    print(f"Model  : {MODEL}")
    print(f"API    : {BASE_URL}")
    print(f"Temp   : {TEMPERATURE}\n")

    print("=" * 60)
    print("FUNDUS experiments")
    print("=" * 60)
    run_experiments(FUNDUS_DIR, "fundus", FUNDUS_PROMPTS)

    print("\n" + "=" * 60)
    print("OCT experiments")
    print("=" * 60)
    run_experiments(OCT_DIR, "oct", OCT_PROMPTS)

    print("\nAll experiments done.")


if __name__ == "__main__":
    main()
