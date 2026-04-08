"""
Calculate classification accuracy for GPT / Claude API experiment results.

Label files:
  fundus : /root/labels.csv          2nd column, row N = image N (no header)
  OCT    : /root/labels-cul.xlsx     2nd column, row N = image N (no header)

Result JSON files are all in one folder /root/autodl-tmp/GPTRESULTS.
Image type (fundus/oct) is determined from the filename:
  {prompt_idx}_{image_type}_{folder_name}.json
  e.g. 1_fundus_mediumcolor.json  /  2_oct_original.json

Output:
  accuracy_results.json   (in current working directory)
"""

import json
import csv
import re
from pathlib import Path

try:
    import openpyxl
except ImportError:
    raise ImportError("Please run: pip install openpyxl")

# ── 配置 ──────────────────────────────────────────────────────────────────
RESULTS_DIR   = Path("/root/autodl-tmp/GPTRESULTS")
FUNDUS_LABELS = Path("/root/labels.csv")
OCT_LABELS    = Path("/root/octlabel.csv")
OUTPUT_FILE   = Path("accuracy_results.json")

# 合法标签集合（小写）
FUNDUS_VALID = {
    "normal",
    "diabetic retinopathy",
    "age-related macular degeneration",
    "glaucoma",
}
OCT_VALID = {
    "normal",
    "diabetic retinopathy",
    "macular hole",
    "age-related macular degeneration",
    "central serous retinopathy",
}


# ── 标签加载 ──────────────────────────────────────────────────────────────
def load_fundus_labels(csv_path: Path) -> dict:
    """读取 CSV 第二列，行号即图片编号（1-based，无表头）。"""
    labels = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row_idx, row in enumerate(reader, start=1):
            if len(row) >= 2 and row[1].strip():
                labels[row_idx] = row[1].strip().lower()
    return labels


def load_oct_labels(xlsx_path: Path) -> dict:
    """读取 XLSX 第二列，行号即图片编号（1-based，无表头）。"""
    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    ws = wb.active
    labels = {}
    for row_idx, row in enumerate(ws.iter_rows(values_only=True), start=1):
        if len(row) >= 2 and row[1] is not None and str(row[1]).strip():
            labels[row_idx] = str(row[1]).strip().lower()
    wb.close()
    return labels


# ── 预测值规范化 ───────────────────────────────────────────────────────────
def normalize_prediction(raw: str, valid_labels: set) -> str | None:
    """
    从模型输出中提取分类标签。
    - PROMPT_1/2（短答案）：直接匹配
    - PROMPT_3（推理链）：取文本中最后出现的合法标签作为最终答案
    """
    text  = raw.strip().lower()
    found = [label for label in valid_labels if label in text]
    if not found:
        return None
    if len(found) == 1:
        return found[0]
    return max(found, key=lambda label: text.rfind(label))


# ── 准确率计算 ─────────────────────────────────────────────────────────────
def calculate_accuracy(results: dict, labels: dict, valid_labels: set) -> dict:
    correct   = 0
    incorrect = 0
    errors    = 0
    no_label  = 0
    no_match  = 0

    per_class: dict[str, dict] = {lb: {"correct": 0, "total": 0} for lb in valid_labels}

    for key, raw in results.items():
        num = int(key)

        if str(raw).startswith("error:"):
            errors += 1
            continue

        if num not in labels:
            no_label += 1
            continue

        gt   = labels[num]
        pred = normalize_prediction(raw, valid_labels)

        if pred is None:
            no_match += 1
            if gt in per_class:
                per_class[gt]["total"] += 1
            continue

        if gt in per_class:
            per_class[gt]["total"] += 1
            if pred == gt:
                per_class[gt]["correct"] += 1

        if pred == gt:
            correct += 1
        else:
            incorrect += 1

    total_valid = correct + incorrect
    accuracy    = round(correct / total_valid, 4) if total_valid > 0 else None

    per_class_accuracy = {}
    for lb, stat in per_class.items():
        if stat["total"] > 0:
            per_class_accuracy[lb] = {
                "accuracy": round(stat["correct"] / stat["total"], 4),
                "correct":  stat["correct"],
                "total":    stat["total"],
            }

    return {
        "accuracy":           accuracy,
        "correct":            correct,
        "incorrect":          incorrect,
        "no_match":           no_match,
        "errors":             errors,
        "no_label":           no_label,
        "total_entries":      len(results),
        "per_class_accuracy": per_class_accuracy,
    }


# ── 主流程 ────────────────────────────────────────────────────────────────
def main():
    print(f"Loading fundus labels from {FUNDUS_LABELS} ...")
    fundus_labels = load_fundus_labels(FUNDUS_LABELS)
    print(f"  {len(fundus_labels)} labels loaded.")

    print(f"Loading OCT labels from {OCT_LABELS} ...")
    oct_labels = load_fundus_labels(OCT_LABELS)
    print(f"  {len(oct_labels)} labels loaded.\n")

    # 文件名格式：{prompt_idx}_{image_type}_{folder_name}.json
    pattern = re.compile(r"^(\d+)_(fundus|oct)_(.+)\.json$")

    accuracy: dict = {"fundus": {}, "oct": {}}

    label_map      = {"fundus": fundus_labels, "oct": oct_labels}
    valid_label_map = {"fundus": FUNDUS_VALID,  "oct": OCT_VALID}

    json_files = sorted(RESULTS_DIR.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {RESULTS_DIR}")
        return

    print(f"Found {len(json_files)} JSON file(s) in {RESULTS_DIR}\n")
    print("=" * 65)

    for json_file in json_files:
        m = pattern.match(json_file.name)
        if not m:
            print(f"[SKIP] Unrecognised filename: {json_file.name}")
            continue

        prompt_idx  = int(m.group(1))
        image_type  = m.group(2)          # "fundus" or "oct"
        folder_name = m.group(3)
        prompt_key  = f"prompt_{prompt_idx}"

        labels       = label_map[image_type]
        valid_labels = valid_label_map[image_type]

        with open(json_file, "r", encoding="utf-8") as f:
            results = json.load(f)

        stats       = calculate_accuracy(results, labels, valid_labels)
        valid_total = stats["correct"] + stats["incorrect"]

        print(f"{json_file.name}")
        print(f"  type={image_type}  folder={folder_name}  prompt={prompt_idx}")
        print(f"  acc={stats['accuracy']}  "
              f"({stats['correct']}/{valid_total} valid, "
              f"no_match={stats['no_match']}, errors={stats['errors']})")
        print()

        accuracy[image_type].setdefault(folder_name, {})[prompt_key] = stats

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(accuracy, f, ensure_ascii=False, indent=2)
    print(f"Done. Results saved -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
