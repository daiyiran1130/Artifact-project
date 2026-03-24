import json
import sys
import argparse
import pandas as pd

LABEL_MAP = {
    "NORMAL": "normal",
    "DR": "diabetic retinopathy",
    "MH": "macular hole",
    "AMRD": "age-related macular degeneration",
    "CSR": "central serous retinopathy",
}


def load_ground_truth(xlsx_path):
    """Load ground truth labels from xlsx file (abbreviations in first column)."""
    df = pd.read_excel(xlsx_path, header=None)
    labels = []
    for val in df.iloc[:, 0]:
        abbr = str(val).strip().upper()
        if abbr not in LABEL_MAP:
            raise ValueError(f"Unknown label abbreviation: '{abbr}'")
        labels.append(LABEL_MAP[abbr])
    return labels


def load_predictions(json_path):
    """Load model predictions from json file (1-indexed keys)."""
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # Sort by numeric key and return ordered list of predictions
    predictions = [raw[str(k)] for k in sorted(raw.keys(), key=lambda x: int(x))]
    return predictions


def calculate_accuracy(ground_truth, predictions):
    if len(ground_truth) != len(predictions):
        raise ValueError(
            f"Length mismatch: ground truth has {len(ground_truth)} samples, "
            f"predictions has {len(predictions)} samples."
        )
    correct = sum(gt == pred for gt, pred in zip(ground_truth, predictions))
    total = len(ground_truth)
    accuracy = correct / total
    return correct, total, accuracy


def per_class_accuracy(ground_truth, predictions):
    classes = sorted(set(ground_truth))
    results = {}
    for cls in classes:
        indices = [i for i, gt in enumerate(ground_truth) if gt == cls]
        correct = sum(predictions[i] == cls for i in indices)
        results[cls] = (correct, len(indices), correct / len(indices))
    return results


def main():
    parser = argparse.ArgumentParser(description="Calculate disease classification accuracy.")
    parser.add_argument("xlsx", help="Path to ground truth xlsx file")
    parser.add_argument("json", help="Path to model predictions json file")
    parser.add_argument("--per-class", action="store_true", help="Show per-class accuracy")
    args = parser.parse_args()

    ground_truth = load_ground_truth(args.xlsx)
    predictions = load_predictions(args.json)

    correct, total, accuracy = calculate_accuracy(ground_truth, predictions)
    print(f"Overall Accuracy: {correct}/{total} = {accuracy:.4f} ({accuracy*100:.2f}%)")

    if args.per_class:
        print("\nPer-class Accuracy:")
        per_class = per_class_accuracy(ground_truth, predictions)
        for cls, (c, t, acc) in per_class.items():
            print(f"  {cls}: {c}/{t} = {acc:.4f} ({acc*100:.2f}%)")


if __name__ == "__main__":
    main()
