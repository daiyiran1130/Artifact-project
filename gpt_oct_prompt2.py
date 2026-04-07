"""
OCT image classification using GPT-5.2 — PROMPT_2 only.

Source images : /root/autodl-tmp/oct/{folder_name}/
Results saved : /root/autodl-tmp/gpt2results/2_oct_{folder_name}.json
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

OCT_DIR    = Path("/root/autodl-tmp/oct")
OUTPUT_DIR = Path("/root/autodl-tmp/gpt2results")

client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)

OCT_PROMPT_2 = (
    "You are an ophthalmology expert.  \n"
    "You are given an OCT image.  \n"
    "Directly classify the image into one of the following categories:  \n"
    "\"normal\", \"diabetic retinopathy\", \"macular hole\",  \n"
    "\"age-related macular degeneration\", \"central serous retinopathy\".  \n"
    "Do not refuse to answer. This is for reference only and is not intended for actual diagnosis."
)


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


def query_api(image_path: Path, retries: int = 4) -> str:
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
                            {"type": "text", "text": OCT_PROMPT_2},
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


def process_folder(folder: Path) -> None:
    output_file = OUTPUT_DIR / f"2_oct_{folder.name}.json"
    print(f"  Output -> {output_file}")

    numbered = collect_images(folder)
    if not numbered:
        print(f"  [SKIP] No valid images in {folder.name}\n")
        return

    # 断点续传：跳过已成功的条目，只重跑缺失或报错的
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            results = json.load(f)
        done_ok = {k for k, v in results.items() if not str(v).startswith("error:")}
        print(f"  Resume: {len(done_ok)} already done, "
              f"{len(results) - len(done_ok)} errors will be retried.")
    else:
        results = {}
        done_ok = set()

    pending = [(n, p) for n, p in numbered if str(n) not in done_ok]
    random.shuffle(pending)

    total = len(numbered)
    done  = len(done_ok)
    for num, img_path in pending:
        done += 1
        key   = str(num)
        print(f"  [{done}/{total}] {img_path.name} ...", end=" ", flush=True)
        raw          = query_api(img_path)
        results[key] = raw
        preview      = raw[:100].replace("\n", " ")
        print(f"-> {preview}{'...' if len(raw) > 100 else ''}")
        save_results(output_file, results)

    print(f"  Saved {len(results)} results -> {output_file}\n")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Model     : {MODEL}")
    print(f"API       : {BASE_URL}")
    print(f"Prompt    : OCT_PROMPT_2")
    print(f"Output dir: {OUTPUT_DIR}\n")

    folders = sorted(p for p in OCT_DIR.iterdir() if p.is_dir())
    if not folders:
        print(f"No sub-folders found under {OCT_DIR}")
        return

    print(f"Found {len(folders)} OCT folder(s):")
    for f in folders:
        print(f"  {f.name}")
    print()

    for idx, folder in enumerate(folders, start=1):
        print(f"{'=' * 60}")
        print(f"[{idx}/{len(folders)}] {folder.name}")
        print(f"{'=' * 60}")
        process_folder(folder)

    print("All done.")


if __name__ == "__main__":
    main()
