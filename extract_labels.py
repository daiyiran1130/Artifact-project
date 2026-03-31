import os
import re
import pandas as pd

LABEL_DIR = "/root/autodl-tmp/label"
OUTPUT_FILE = "labels.csv"

LABEL_MAP = {
    "A": "Age-related Macular Degeneration",
    "D": "Diabetic Retinopathy",
    "G": "Glaucoma",
    "N": "Normal",
}

def extract_labels(label_dir):
    pattern = re.compile(r"^(\d+)_([ADGN])\.png$", re.IGNORECASE)
    records = {}

    for filename in os.listdir(label_dir):
        m = pattern.match(filename)
        if m:
            idx = int(m.group(1))
            suffix = m.group(2).upper()
            label = LABEL_MAP[suffix]
            records[idx] = label

    if not records:
        print("No matching images found.")
        return

    max_idx = max(records.keys())
    rows = []
    for i in range(1, max_idx + 1):
        label = records.get(i, "")
        rows.append({"label": label})

    df = pd.DataFrame(rows)
    df.index = range(1, len(df) + 1)
    df.to_csv(OUTPUT_FILE, index=True, index_label="image_index")
    print(f"Saved {len(records)} labels to {OUTPUT_FILE}")
    print(df.head(10))

if __name__ == "__main__":
    extract_labels(LABEL_DIR)
