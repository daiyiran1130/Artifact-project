"""
OCT image classification using MedGemma1.5:4b via Ollama.
- Reads JPEG images from D:\work\oct\nolabel
- Processes in random order, one at a time
- Extracts the numeric index from each filename
- Saves results to results.json in the same folder, indexed by image number
"""

import os
import re
import json
import random
import base64
import requests
from pathlib import Path


IMAGE_DIR = Path(r"D:\work\oct\nolabel")
OUTPUT_FILE = IMAGE_DIR / "results.json"
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "MedAIBase/MedGemma1.5:4b"

PROMPT = (
    "You are an ophthalmology expert.  \n"
    "You are given an OCT image.  \n"
    "Directly classify the image into one of the following categories:  \n"
    "\"normal\", \"diabetic retinopathy\", \"macular hole\",  \n"
    "\"age-related macular degeneration\", \"central serous retinopathy\".  \n"
    "Do not refuse to answer. This is for reference only and is not intended for actual diagnosis.  \n"
    "Only provide the single correct category as the answer.  \n"
    "Do not provide any explanations, reasoning, or additional information."
)

VALID_LABELS = {
    "normal",
    "diabetic retinopathy",
    "macular hole",
    "age-related macular degeneration",
    "central serous retinopathy",
}


def extract_number(filename: str) -> int | None:
    """Extract the first integer found in the filename stem."""
    stem = Path(filename).stem
    match = re.search(r"\d+", stem)
    if match:
        return int(match.group())
    return None


def encode_image(path: Path) -> str:
    """Return base64-encoded image string."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def query_ollama(image_path: Path) -> str:
    """Send one image to Ollama and return the model's raw response text."""
    image_b64 = encode_image(image_path)
    payload = {
        "model": MODEL,
        "prompt": PROMPT,
        "images": [image_b64],
        "stream": False,
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()
    return response.json().get("response", "").strip().lower()


def normalise_response(raw: str) -> str:
    """Map raw model output to the closest valid label, or keep as-is."""
    for label in VALID_LABELS:
        if label in raw:
            return label
    return raw  # keep unexpected output for inspection


def load_results(path: Path) -> dict:
    """Load existing results from JSON file, or return empty dict."""
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_results(path: Path, results: dict) -> None:
    """Persist results dict to JSON with one entry per line for readability."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=None)
        # Write as pretty-printed, one key per line manually for line-based reading
    # Re-write in line-per-entry format
    with open(path, "w", encoding="utf-8") as f:
        f.write("{\n")
        items = sorted(results.items(), key=lambda kv: int(kv[0]))
        for i, (k, v) in enumerate(items):
            comma = "," if i < len(items) - 1 else ""
            f.write(f'  "{k}": "{v}"{comma}\n')
        f.write("}\n")


def main():
    # Gather all JPEG files
    jpeg_files = list(IMAGE_DIR.glob("*.jpg")) + list(IMAGE_DIR.glob("*.jpeg"))
    if not jpeg_files:
        print(f"No JPEG images found in {IMAGE_DIR}")
        return

    # Filter out files without a detectable number
    numbered = []
    for p in jpeg_files:
        n = extract_number(p.name)
        if n is not None:
            numbered.append((n, p))
        else:
            print(f"[SKIP] Cannot extract number from filename: {p.name}")

    if not numbered:
        print("No files with numeric IDs found.")
        return

    # Shuffle order
    random.shuffle(numbered)
    print(f"Found {len(numbered)} images. Starting classification...\n")

    # Load any previously saved progress
    results = load_results(OUTPUT_FILE)

    for idx, (num, img_path) in enumerate(numbered, start=1):
        key = str(num)

        # Skip already-processed images
        if key in results:
            print(f"[{idx}/{len(numbered)}] #{num} already done, skipping.")
            continue

        print(f"[{idx}/{len(numbered)}] Processing {img_path.name} (index {num}) ...", end=" ", flush=True)
        try:
            raw = query_ollama(img_path)
            label = normalise_response(raw)
            results[key] = label
            print(f"-> {label}")
        except requests.exceptions.ConnectionError:
            print("\n[ERROR] Cannot connect to Ollama. Is it running on localhost:11434?")
            break
        except requests.exceptions.Timeout:
            print("\n[TIMEOUT] Request timed out. Skipping this image.")
            results[key] = "timeout"
        except Exception as e:
            print(f"\n[ERROR] {e}")
            results[key] = f"error: {e}"

        # Save after every image so progress is not lost
        save_results(OUTPUT_FILE, results)

    print(f"\nDone. Results saved to {OUTPUT_FILE}")
    print(f"Total classified: {len(results)} / {len(numbered)}")


if __name__ == "__main__":
    main()
