"""
Calculate overall accuracy and per-class accuracy for MedGemma classification results.

Input:
  - JSON files in RESULTS_DIR (e.g., /root/autodl-tmp/medgemma27b/)
    Format: {"image_index": "model_output_text", ...}
    Files are grouped by numeric prefix: 1_*.json, 2_*.json, 3_*.json

  - LABELS_CSV: CSV file where row N (1-indexed) is the label for image N,
    and the label is in the second column (index 1).

Output:
  - 3 JSON files (one per prefix group) saved to OUTPUT_DIR.
    Each contains overall_accuracy and per_class_accuracy for every experiment.

Classes (case-insensitive keyword matching used to parse model output):
  normal, diabetic retinopathy, macular hole,
  age-related macular degeneration, central serous retinopathy
"""

import csv
import json
import re
from pathlib import Path
from collections import defaultdict

# ── Configuration ─────────────────────────────────────────────────────────────
RESULTS_DIR = Path("/root/autodl-tmp/medgemma27b")
LABELS_CSV  = Path("/root/autodl-tmp/medgemma27b/labels.csv")
OUTPUT_DIR  = Path("/root/autodl-tmp/medgemma27b")

# Canonical class names and their keyword aliases (longest match tried first)
CLASSES = [
    "age-related macular degeneration",
    "diabetic retinopathy",
    "glaucoma",
    "normal",
]

# Short aliases that also count as a class match
ALIASES = {
    "amd":                              "age-related macular degeneration",
    "age related macular degeneration": "age-related macular degeneration",
    "armd":                             "age-related macular degeneration",
    "dr":                               "diabetic retinopathy",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_labels(csv_path: Path) -> dict[int, str]:
    """
    Return {image_index: label} from the CSV.
    Row 1 → image 1, row 2 → image 2, etc.  Label is the second column.
    """
    labels = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row_num, row in enumerate(reader, start=1):
            if len(row) < 2:
                continue
            label = row[1].strip().lower()
            labels[row_num] = label
    return labels


def parse_prediction(text: str) -> str | None:
    """
    Extract the predicted class from free-form model output text.
    Returns the canonical class name, or None if no class is found.
    """
    t = text.lower()

    # Check aliases first (they may contain multi-word strings)
    for alias, canonical in ALIASES.items():
        if alias in t:
            return canonical

    # Then check canonical class names (longest first to avoid partial matches)
    for cls in CLASSES:
        if cls in t:
            return cls

    return None


def compute_accuracy(results: dict[str, str], labels: dict[int, str]) -> dict:
    """
    Given a prediction dict {str(index): raw_text} and a labels dict {int: label},
    return overall_accuracy and per_class_accuracy.
    """
    total = 0
    correct = 0
    class_total:   dict[str, int] = defaultdict(int)
    class_correct: dict[str, int] = defaultdict(int)

    for key, raw in results.items():
        try:
            idx = int(key)
        except ValueError:
            continue

        true_label = labels.get(idx)
        if true_label is None:
            continue

        pred_label = parse_prediction(raw)

        total += 1
        class_total[true_label] += 1

        if pred_label == true_label:
            correct += 1
            class_correct[true_label] += 1

    overall = correct / total if total > 0 else 0.0

    per_class = {}
    for cls in sorted(class_total):
        n = class_total[cls]
        c = class_correct.get(cls, 0)
        per_class[cls] = round(c / n, 6) if n > 0 else 0.0

    return {
        "overall_accuracy": round(overall, 6),
        "total_images":     total,
        "correct":          correct,
        "per_class_accuracy": per_class,
        "per_class_counts": {
            cls: {"correct": class_correct.get(cls, 0), "total": class_total[cls]}
            for cls in sorted(class_total)
        },
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Load ground-truth labels
    if not LABELS_CSV.exists():
        raise FileNotFoundError(f"Labels CSV not found: {LABELS_CSV}")
    labels = load_labels(LABELS_CSV)
    print(f"Loaded {len(labels)} labels from {LABELS_CSV}")

    # Collect all result JSON files (exclude output files we produce)
    output_names = {"accuracy_group1.json", "accuracy_group2.json", "accuracy_group3.json"}
    json_files = [
        p for p in RESULTS_DIR.glob("*.json")
        if p.name not in output_names
    ]
    if not json_files:
        raise FileNotFoundError(f"No JSON result files found in {RESULTS_DIR}")

    # Group files by numeric prefix (1, 2, 3)
    groups: dict[str, dict] = defaultdict(dict)
    for jf in sorted(json_files):
        m = re.match(r"^(\d+)_", jf.stem)
        if not m:
            print(f"[SKIP] Cannot determine group for: {jf.name}")
            continue
        prefix = m.group(1)
        experiment_name = jf.stem          # e.g. "1_mediumcolor"

        with open(jf, encoding="utf-8") as f:
            results = json.load(f)

        acc = compute_accuracy(results, labels)
        groups[prefix][experiment_name] = acc
        print(f"  [{prefix}] {experiment_name}: "
              f"overall={acc['overall_accuracy']:.4f}  "
              f"({acc['correct']}/{acc['total_images']})")

    # Save one output JSON per group
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for prefix, experiments in sorted(groups.items()):
        out_path = OUTPUT_DIR / f"accuracy_group{prefix}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(experiments, f, indent=2, ensure_ascii=False)
        print(f"\nSaved group {prefix} results → {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
