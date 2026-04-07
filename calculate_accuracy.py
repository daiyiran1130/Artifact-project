"""
Calculate classification accuracy for GPT / Claude API experiment results.

Label files:
  fundus : /root/autodl-tmp/fundus/labels.csv       2nd column, row N = image N (no header)
  OCT    : /root/autodl-tmp/oct/labels-cul.xlsx     2nd column, row N = image N (no header)

Result JSON files found automatically:
  {folder}/{prompt_idx}_{image_type}_{folder_name}.json

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
FUNDUS_DIR    = Path("/root/autodl-tmp/fundus")
OCT_DIR       = Path("/root/autodl-tmp/oct")
FUNDUS_LABELS = FUNDUS_DIR / "labels.csv"
OCT_LABELS    = OCT_DIR / "labels-cul.xlsx"
OUTPUT_FILE   = Path("accuracy_results.json")

# 合法标签集合（小写），用于从模型输出中提取分类结果
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
    """
    读取 CSV 第二列，行号即图片编号（1-based，无表头）。
    返回 {image_num: label_lowercase}
    """
    labels = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row_idx, row in enumerate(reader, start=1):
            if len(row) >= 2 and row[1].strip():
                labels[row_idx] = row[1].strip().lower()
    return labels


def load_oct_labels(xlsx_path: Path) -> dict:
    """
    读取 XLSX 第二列，行号即图片编号（1-based，无表头）。
    返回 {image_num: label_lowercase}
    """
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
    - PROMPT_3（推理链）：找文本中最后出现的合法标签作为最终答案
    返回小写标签，无法匹配则返回 None。
    """
    text = raw.strip().lower()
    found = [label for label in valid_labels if label in text]
    if not found:
        return None
    if len(found) == 1:
        return found[0]
    # 多个匹配时取文本中最后出现的（推理链通常以结论收尾）
    return max(found, key=lambda label: text.rfind(label))


# ── 准确率计算 ─────────────────────────────────────────────────────────────
def calculate_accuracy(results: dict, labels: dict, valid_labels: set) -> dict:
    correct   = 0
    incorrect = 0
    errors    = 0   # API 调用失败的条目
    no_label  = 0   # 在标签文件中找不到对应编号
    no_match  = 0   # 模型输出无法匹配任何合法标签

    wrong_examples   = []   # 记录前5个错误样本（便于调试）
    nomatch_examples = []

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
            if len(nomatch_examples) < 5:
                nomatch_examples.append({"image": num, "raw": raw[:120]})
            continue

        if pred == gt:
            correct += 1
        else:
            incorrect += 1
            if len(wrong_examples) < 5:
                wrong_examples.append({"image": num, "gt": gt, "pred": pred})

    total_valid = correct + incorrect
    accuracy = round(correct / total_valid, 4) if total_valid > 0 else None

    return {
        "accuracy":         accuracy,
        "correct":          correct,
        "incorrect":        incorrect,
        "no_match":         no_match,
        "errors":           errors,
        "no_label":         no_label,
        "total_entries":    len(results),
        "wrong_examples":   wrong_examples,
        "nomatch_examples": nomatch_examples,
    }


# ── 扫描文件夹 ─────────────────────────────────────────────────────────────
def scan_folder(folder: Path, image_type: str,
                labels: dict, valid_labels: set) -> dict:
    """
    找出该文件夹下所有 {prompt_idx}_{image_type}_{folder_name}.json，
    返回 {"prompt_1": {...}, "prompt_2": {...}, ...}
    """
    pattern = re.compile(
        rf"^(\d+)_{re.escape(image_type)}_{re.escape(folder.name)}\.json$"
    )
    folder_results = {}
    for json_file in sorted(folder.glob("*.json")):
        m = pattern.match(json_file.name)
        if not m:
            continue
        prompt_idx = int(m.group(1))
        with open(json_file, "r", encoding="utf-8") as f:
            results = json.load(f)
        stats = calculate_accuracy(results, labels, valid_labels)
        key   = f"prompt_{prompt_idx}"
        folder_results[key] = stats
        valid_total = stats["correct"] + stats["incorrect"]
        print(f"    {json_file.name:<45}  "
              f"acc={stats['accuracy']}  "
              f"({stats['correct']}/{valid_total} valid, "
              f"no_match={stats['no_match']}, errors={stats['errors']})")
    return folder_results


def run_type(base_dir: Path, image_type: str,
             labels: dict, valid_labels: set) -> dict:
    if not base_dir.is_dir():
        print(f"[WARNING] {base_dir} not found, skipping.")
        return {}
    folders = sorted(p for p in base_dir.iterdir() if p.is_dir())
    type_results = {}
    for folder in folders:
        stats = scan_folder(folder, image_type, labels, valid_labels)
        if stats:
            print(f"  [{folder.name}]")
            type_results[folder.name] = stats
    return type_results


# ── 主流程 ────────────────────────────────────────────────────────────────
def main():
    print(f"Loading fundus labels from {FUNDUS_LABELS} ...")
    fundus_labels = load_fundus_labels(FUNDUS_LABELS)
    print(f"  {len(fundus_labels)} labels loaded.\n")

    print(f"Loading OCT labels from {OCT_LABELS} ...")
    oct_labels = load_oct_labels(OCT_LABELS)
    print(f"  {len(oct_labels)} labels loaded.\n")

    accuracy = {}

    print("=" * 60)
    print("FUNDUS accuracy")
    print("=" * 60)
    accuracy["fundus"] = run_type(FUNDUS_DIR, "fundus", fundus_labels, FUNDUS_VALID)

    print()
    print("=" * 60)
    print("OCT accuracy")
    print("=" * 60)
    accuracy["oct"] = run_type(OCT_DIR, "oct", oct_labels, OCT_VALID)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(accuracy, f, ensure_ascii=False, indent=2)
    print(f"\nDone. Results saved -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
