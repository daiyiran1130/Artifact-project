"""
Evaluate classification accuracy for OCT disease classification experiments.

Reads ground-truth labels from /root/labels-cul.xlsx and predictions from
JSON result files in /root/autodl-tmp/3prompt/.

Label mapping (xlsx column B → full disease name):
  AMRD  → age-related macular degeneration
  CSR   → central serous retinopathy
  MH    → macular hole
  DR    → diabetic retinopathy
  NORMAL→ normal

Prediction extraction: looks for $\boxed{...}$ in the model output text,
then falls back to keyword matching.
"""

import json
import re
from pathlib import Path

import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
LABEL_FILE  = Path("/root/labels-cul.xlsx")
RESULTS_DIR = Path("/root/autodl-tmp/3prompt")

# ── Label mapping ─────────────────────────────────────────────────────────────
# Maps the xlsx label code → canonical keyword used to match model output
LABEL_TO_KEYWORDS = {
    "AMRD":   ["age-related macular degeneration", "amd", "amrd"],
    "CSR":    ["central serous retinopathy", "csr"],
    "MH":     ["macular hole", "mh"],
    "DR":     ["diabetic retinopathy", "dr"],
    "NORMAL": ["normal"],
}

# All disease keywords ordered longest-first so longer phrases match before short ones
ALL_DISEASE_KEYWORDS = [
    ("age-related macular degeneration", "AMRD"),
    ("central serous retinopathy",       "CSR"),
    ("diabetic retinopathy",             "DR"),
    ("macular hole",                     "MH"),
    ("normal",                           "NORMAL"),
    ("amd",                              "AMRD"),
    ("amrd",                             "AMRD"),
    ("csr",                              "CSR"),
    ("mh",                               "MH"),
    # "dr" is ambiguous, skip standalone abbreviation
]


def extract_prediction(text: str) -> str | None:
    """
    Extract the predicted disease label from a model output string.

    Strategy (in order):
    1. Look for \\boxed{...} (LaTeX boxed answer)
    2. Look for 'final answer is/the answer is' followed by disease name
    3. Look for 'most likely classification is' followed by disease name
    4. Keyword scan on the last 300 characters (conclusion region)
    5. Full-text keyword scan
    """
    text_lower = text.lower()

    # 1. LaTeX boxed answer: \boxed{central serous retinopathy}
    boxed = re.search(r'\\boxed\{([^}]+)\}', text_lower)
    if boxed:
        candidate = boxed.group(1).strip()
        for keyword, label in ALL_DISEASE_KEYWORDS:
            if keyword in candidate:
                return label

    # 2. "final answer is X" or "the answer is X"
    final_ans = re.search(
        r'(?:final answer is|the answer is)\s+([^\n.]+)',
        text_lower
    )
    if final_ans:
        candidate = final_ans.group(1).strip()
        for keyword, label in ALL_DISEASE_KEYWORDS:
            if keyword in candidate:
                return label

    # 3. "most likely classification is X" / "classified as X"
    classif = re.search(
        r'(?:most likely (?:classification|diagnosis) is|classified as|classification is)\s+([^\n.]+)',
        text_lower
    )
    if classif:
        candidate = classif.group(1).strip()
        for keyword, label in ALL_DISEASE_KEYWORDS:
            if keyword in candidate:
                return label

    # 4. Keyword scan over the last 300 chars (conclusion region)
    tail = text_lower[-300:]
    for keyword, label in ALL_DISEASE_KEYWORDS:
        if keyword in tail:
            return label

    # 5. Full-text keyword scan
    for keyword, label in ALL_DISEASE_KEYWORDS:
        if keyword in text_lower:
            return label

    return None  # could not determine


def load_labels(xlsx_path: Path) -> dict[int, str]:
    """
    Load ground-truth labels from xlsx.
    Returns {image_index (1-based): label_code}.
    """
    df = pd.read_excel(xlsx_path, header=None)
    labels = {}
    for row_idx, row in df.iterrows():
        image_num = row_idx + 1          # 1-based
        label_code = str(row[1]).strip().upper()
        labels[image_num] = label_code
    return labels


def evaluate_file(json_path: Path, labels: dict[int, str]) -> dict:
    """
    Evaluate a single result JSON file against ground-truth labels.
    Returns a dict with overall and per-class accuracy stats.
    """
    with open(json_path, encoding="utf-8") as f:
        results = json.load(f)

    # Per-class counters: {label: [correct, total]}
    class_stats: dict[str, list[int]] = {lbl: [0, 0] for lbl in LABEL_TO_KEYWORDS}
    overall_correct = 0
    overall_total   = 0
    unparseable     = 0

    for key, text in results.items():
        image_num = int(key)
        gt_label  = labels.get(image_num)
        if gt_label is None:
            continue  # image number not in label file

        pred_label = extract_prediction(text)

        overall_total += 1
        if gt_label in class_stats:
            class_stats[gt_label][1] += 1

        if pred_label is None:
            unparseable += 1
            continue

        if pred_label == gt_label:
            overall_correct += 1
            if gt_label in class_stats:
                class_stats[gt_label][0] += 1
        # If wrong, total was already incremented above

    overall_acc = overall_correct / overall_total if overall_total > 0 else 0.0

    per_class = {}
    for lbl, (correct, total) in class_stats.items():
        per_class[lbl] = {
            "correct": correct,
            "total":   total,
            "accuracy": correct / total if total > 0 else 0.0,
        }

    return {
        "file":            json_path.name,
        "overall_correct": overall_correct,
        "overall_total":   overall_total,
        "overall_accuracy": overall_acc,
        "unparseable":     unparseable,
        "per_class":       per_class,
    }


def print_report(stats: dict) -> None:
    """Pretty-print evaluation results for one JSON file."""
    print(f"\n{'='*60}")
    print(f"  File: {stats['file']}")
    print(f"{'='*60}")
    print(f"  Overall accuracy: {stats['overall_accuracy']:.4f}  "
          f"({stats['overall_correct']}/{stats['overall_total']})")
    if stats['unparseable'] > 0:
        print(f"  Unparseable predictions: {stats['unparseable']}")
    print(f"\n  Per-class accuracy:")
    print(f"  {'Label':<10} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print(f"  {'-'*40}")
    for lbl, info in stats['per_class'].items():
        print(f"  {lbl:<10} {info['correct']:>8} {info['total']:>8} {info['accuracy']:>10.4f}")


OUTPUT_RESULTS = RESULTS_DIR / "culresults.json"


def build_output_record(stats: dict) -> dict:
    """将单个文件的统计结果整理为要写入 culresults.json 的格式。"""
    return {
        "overall_accuracy": round(stats["overall_accuracy"], 6),
        "overall_correct":  stats["overall_correct"],
        "overall_total":    stats["overall_total"],
        "unparseable":      stats["unparseable"],
        "per_class": {
            lbl: {
                "accuracy": round(info["accuracy"], 6),
                "correct":  info["correct"],
                "total":    info["total"],
            }
            for lbl, info in stats["per_class"].items()
        },
    }


def main():
    # Load ground-truth labels
    print(f"Loading labels from {LABEL_FILE} ...")
    labels = load_labels(LABEL_FILE)
    print(f"Loaded {len(labels)} labels.")

    # Find all JSON result files, skip culresults.json itself
    json_files = sorted(
        p for p in RESULTS_DIR.glob("*.json")
        if p.name != OUTPUT_RESULTS.name
    )
    if not json_files:
        print(f"No JSON files found in {RESULTS_DIR}")
        return

    print(f"Found {len(json_files)} result files in {RESULTS_DIR}")

    all_stats = []
    output = {}  # {filename: accuracy_record}

    for json_path in json_files:
        stats = evaluate_file(json_path, labels)
        print_report(stats)
        all_stats.append(stats)
        output[json_path.name] = build_output_record(stats)

    # ── 保存到 culresults.json ───────────────────────────────────────────────
    with open(OUTPUT_RESULTS, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {OUTPUT_RESULTS}")

    # ── Summary across all files ─────────────────────────────────────────────
    if len(all_stats) > 1:
        print(f"\n{'='*60}")
        print("  SUMMARY ACROSS ALL FILES")
        print(f"{'='*60}")

        agg_class: dict[str, list[int]] = {lbl: [0, 0] for lbl in LABEL_TO_KEYWORDS}
        agg_correct = 0
        agg_total   = 0

        for s in all_stats:
            agg_correct += s['overall_correct']
            agg_total   += s['overall_total']
            for lbl, info in s['per_class'].items():
                agg_class[lbl][0] += info['correct']
                agg_class[lbl][1] += info['total']

        print(f"  Files evaluated: {len(all_stats)}")
        print(f"  Avg overall accuracy: "
              f"{sum(s['overall_accuracy'] for s in all_stats)/len(all_stats):.4f}")
        print(f"  Pooled overall accuracy: "
              f"{agg_correct/agg_total if agg_total else 0:.4f}  ({agg_correct}/{agg_total})")

        print(f"\n  Pooled per-class accuracy:")
        print(f"  {'Label':<10} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
        print(f"  {'-'*40}")
        for lbl, (correct, total) in agg_class.items():
            acc = correct / total if total > 0 else 0.0
            print(f"  {lbl:<10} {correct:>8} {total:>8} {acc:>10.4f}")

        print(f"\n  Per-file overall accuracy:")
        print(f"  {'File':<40} {'Accuracy':>10}")
        print(f"  {'-'*52}")
        for s in all_stats:
            print(f"  {s['file']:<40} {s['overall_accuracy']:>10.4f}")


if __name__ == "__main__":
    main()
