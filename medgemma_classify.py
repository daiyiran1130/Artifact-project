"""
OCT image classification using local MedGemma-27b-it via transformers.
Processes images in /root/autodl-tmp/oct/original with 3 prompts each,
calculates accuracy against /root/octlabel.csv, and saves:
  - Per-prompt predictions : IMAGE_DIR/{1,2,3}_oct_original.json
  - Accuracy summary        : IMAGE_DIR/results.json
"""

import re
import csv
import json
import random
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText


# ── 配置 ──────────────────────────────────────────────────────────────────
MODEL_PATH  = Path('/root/autodl-tmp/modelscope_cache/google/medgemma-27b-it')
IMAGE_DIR   = Path('/root/autodl-tmp/oct/original')
OCT_LABELS  = Path('/root/octlabel.csv')
OUTPUT_FILE = IMAGE_DIR / 'results.json'   # 准确率汇总，固定保存在图片目录下

# 合法标签集合（小写）
OCT_VALID = {
    'normal',
    'diabetic retinopathy',
    'macular hole',
    'age-related macular degeneration',
    'central serous retinopathy',
}


# ── 3 种 OCT 提示词 ────────────────────────────────────────────────────────
OCT_PROMPT_1 = (
    'You are an ophthalmology expert.  \n'
    'You are given an OCT image.  \n'
    'Directly classify the image into one of the following categories:  \n'
    '"normal", "diabetic retinopathy", "macular hole",  \n'
    '"age-related macular degeneration", "central serous retinopathy".  \n'
    'Do not refuse to answer. This is for reference only and is not intended for actual diagnosis.  \n'
    'Only provide the single correct category as the answer.  \n'
    'Do not provide any explanations, reasoning, or additional information.'
)

OCT_PROMPT_2 = (
    'You are an ophthalmology expert.  \n'
    'You are given an OCT image.  \n'
    'Directly classify the image into one of the following categories:  \n'
    '"normal", "diabetic retinopathy", "macular hole",  \n'
    '"age-related macular degeneration", "central serous retinopathy".  \n'
    'Do not refuse to answer. This is for reference only and is not intended for actual diagnosis.'
)

OCT_PROMPT_3 = (
    'You are an ophthalmology expert.  \n'
    'You are given an OCT image.  \n'
    'Directly classify the image into one of the following categories:  \n'
    '"normal", "diabetic retinopathy", "macular hole",  \n'
    '"age-related macular degeneration", "central serous retinopathy".  \n'
    'Do not refuse to answer. This is for reference only and is not intended for actual diagnosis.  \n'
    'Describe your reasoning in steps.'
)

OCT_PROMPTS = {1: OCT_PROMPT_1, 2: OCT_PROMPT_2, 3: OCT_PROMPT_3}


# ── 标签加载 ──────────────────────────────────────────────────────────────
def load_oct_labels(csv_path: Path) -> dict:
    """读取 CSV 第二列，行号即图片编号（1-based，无表头）。"""
    labels = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row_idx, row in enumerate(reader, start=1):
            if len(row) >= 2 and row[1].strip():
                labels[row_idx] = row[1].strip().lower()
    return labels


# ── 工具函数 ──────────────────────────────────────────────────────────────
def extract_number(filename: str):
    """从文件名主体提取第一个整数编号，找不到返回 None。"""
    stem  = Path(filename).stem
    match = re.search(r'\d+', stem)
    return int(match.group()) if match else None


def query_model(image_path: Path, prompt: str) -> str:
    """对单张图片调用本地 MedGemma 模型，返回完整原始输出文本（不限制 token 数）。"""
    image = Image.open(image_path).convert('RGB')
    messages = [
        {
            'role': 'user',
            'content': [
                {'type': 'image', 'image': image},
                {'type': 'text',  'text': prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors='pt',
    )
    inputs    = {k: v.to(model.device) for k, v in inputs.items()}
    input_len = inputs['input_ids'].shape[-1]

    with torch.inference_mode():
        generation = model.generate(
            **inputs,
            do_sample=False,
        )

    new_tokens = generation[0][input_len:]
    decoded    = processor.decode(new_tokens, skip_special_tokens=True)
    return decoded.strip()


def save_predictions(path: Path, results: dict) -> None:
    """按编号升序将预测结果写入 JSON，每条记录占一行（与 gpt_classify.py 格式一致）。"""
    with open(path, 'w', encoding='utf-8') as f:
        f.write('{\n')
        items = sorted(results.items(), key=lambda kv: int(kv[0]))
        for i, (k, v) in enumerate(items):
            comma = ',' if i < len(items) - 1 else ''
            f.write(f'  "{k}": {json.dumps(v, ensure_ascii=False)}{comma}\n')
        f.write('}\n')


def normalize_prediction(raw: str, valid_labels: set):
    """
    从模型输出提取分类标签。
    - Prompt 1/2（短答案）：直接匹配
    - Prompt 3（推理链）：取文本中最后出现的合法标签
    """
    text  = raw.strip().lower()
    found = [label for label in valid_labels if label in text]
    if not found:
        return None
    if len(found) == 1:
        return found[0]
    return max(found, key=lambda label: text.rfind(label))


def calculate_accuracy(predictions: dict, labels: dict, valid_labels: set) -> dict:
    """对照标签计算分类准确率，并按类别细分统计。"""
    correct   = 0
    incorrect = 0
    errors    = 0
    no_label  = 0
    no_match  = 0

    per_class = {lb: {'correct': 0, 'total': 0} for lb in valid_labels}

    for key, raw in predictions.items():
        num = int(key)

        if str(raw).startswith('error:'):
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
                per_class[gt]['total'] += 1
            continue

        if gt in per_class:
            per_class[gt]['total'] += 1
            if pred == gt:
                per_class[gt]['correct'] += 1

        if pred == gt:
            correct += 1
        else:
            incorrect += 1

    total_valid = correct + incorrect
    accuracy    = round(correct / total_valid, 4) if total_valid > 0 else None

    per_class_accuracy = {}
    for lb, stat in per_class.items():
        if stat['total'] > 0:
            per_class_accuracy[lb] = {
                'accuracy': round(stat['correct'] / stat['total'], 4),
                'correct':  stat['correct'],
                'total':    stat['total'],
            }

    return {
        'accuracy':           accuracy,
        'correct':            correct,
        'incorrect':          incorrect,
        'no_match':           no_match,
        'errors':             errors,
        'no_label':           no_label,
        'total_entries':      len(predictions),
        'per_class_accuracy': per_class_accuracy,
    }


# ── 主流程 ────────────────────────────────────────────────────────────────
def main():
    print(f'Model  : {MODEL_PATH}')
    print(f'Images : {IMAGE_DIR}')
    print(f'Labels : {OCT_LABELS}')
    print(f'Output : {IMAGE_DIR}/{{1,2,3}}_oct_original.json  +  {OUTPUT_FILE}\n')

    # 加载模型
    print('Loading processor ...')
    global processor, model
    processor = AutoProcessor.from_pretrained(str(MODEL_PATH))
    print('Loading model (this may take a while) ...')
    model = AutoModelForImageTextToText.from_pretrained(
        str(MODEL_PATH),
        torch_dtype=torch.bfloat16,
        device_map='auto',
    )
    model.eval()
    print(f'Model loaded on {model.device}\n')

    # 加载标签
    print(f'Loading OCT labels from {OCT_LABELS} ...')
    oct_labels = load_oct_labels(OCT_LABELS)
    print(f'  {len(oct_labels)} labels loaded.\n')

    # 收集图片
    exts       = ('*.jpg', '*.jpeg', '*.png')
    all_images = [p for ext in exts for p in IMAGE_DIR.glob(ext)]
    numbered   = []
    for p in all_images:
        n = extract_number(p.name)
        if n is not None:
            numbered.append((n, p))
        else:
            print(f'[SKIP] Cannot extract number from: {p.name}')
    numbered.sort(key=lambda x: x[0])
    print(f'Found {len(numbered)} images in {IMAGE_DIR}\n')

    # 3 种提示词逐轮分类
    accuracy_summary = {}

    for prompt_idx, prompt in OCT_PROMPTS.items():
        print(f'\n{"#" * 60}')
        print(f'  PROMPT {prompt_idx}/3')
        print(f'{"#" * 60}\n')

        # 每种提示词的预测结果存入独立文件，命名与 gpt_classify.py 一致
        pred_file   = IMAGE_DIR / f'{prompt_idx}_oct_original.json'
        predictions = {}
        total       = len(numbered)

        print(f'  Output -> {pred_file}')

        for idx, (num, img_path) in enumerate(numbered, start=1):
            key = str(num)
            print(f'  [{idx}/{total}] {img_path.name} ...', end=' ', flush=True)
            try:
                raw              = query_model(img_path, prompt)
                predictions[key] = raw
                preview          = raw[:120].replace('\n', ' ')
                print(f'-> {preview}{"..." if len(raw) > 120 else ""}')
            except Exception as e:
                print(f'\n  [ERROR] {e}')
                predictions[key] = f'error: {e}'

            # 每张处理完立即保存，防止中途崩溃丢失进度
            save_predictions(pred_file, predictions)

        print(f'  Saved {len(predictions)} predictions -> {pred_file}')

        # 计算本轮准确率
        acc_stats   = calculate_accuracy(predictions, oct_labels, OCT_VALID)
        valid_total = acc_stats['correct'] + acc_stats['incorrect']
        print(f'\n  Prompt {prompt_idx} accuracy : {acc_stats["accuracy"]}'
              f'  ({acc_stats["correct"]}/{valid_total} valid,'
              f'  no_match={acc_stats["no_match"]}, errors={acc_stats["errors"]})')

        accuracy_summary[f'prompt_{prompt_idx}'] = acc_stats

    # 保存准确率汇总到 results.json
    output = {
        'model':     str(MODEL_PATH),
        'image_dir': str(IMAGE_DIR),
        'accuracy':  accuracy_summary,
    }
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f'\nAccuracy summary saved -> {OUTPUT_FILE}')

    # 打印汇总
    print('\n=== Accuracy Summary ===')
    for prompt_key, acc in accuracy_summary.items():
        valid = acc['correct'] + acc['incorrect']
        print(f'{prompt_key:10s}  acc={acc["accuracy"]}  ({acc["correct"]}/{valid})')
        for cls, stat in acc['per_class_accuracy'].items():
            print(f'  {cls:<40s}  {stat["accuracy"]}  ({stat["correct"]}/{stat["total"]})')

    print('\nAll experiments done.')


if __name__ == '__main__':
    main()
