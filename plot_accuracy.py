#!/usr/bin/env python3
"""
Medical Image Classification Bar Chart Generator
=================================================
Draws accuracy bar charts with 95% bootstrap CI and pairwise McNemar tests.

Two chart modes:
  Mode A  X-axis = models      (4 bars, 1 image type per chart)
  Mode B  X-axis = image types (groups of 4 model bars, up to N image types)

Usage:
    python plot_accuracy.py
    -> Follow the interactive prompts

JSON filename format expected in RESULTS_DIR:
    {prompt}_{dataset}_{image_type}_{model}.json
    e.g.  1_fundus_weakblur_4b.json
"""

import os
import re
import json
import subprocess
import sys
import numpy as np
import pandas as pd
import matplotlib
# matplotlib.use('Agg')  # ← 取消注释此行可关闭弹窗，只保存文件（服务器/无界面环境使用）
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from itertools import combinations
from scipy import stats


# ================================================================
# ★  CONFIGURATION — Edit this section to customise the chart  ★
# ================================================================

# ---- Paths ----
RESULTS_DIR = r"D:\work\artifact photo\moderesults"

CSV_PATHS = {
    "fundus": r"D:\work\artifact photo\fundus.csv",
    "oct":    r"D:\work\artifact photo\oct.csv",
}

# ---- Model display names ----
# Key   = filename token (must match the JSON filename)
# Value = display name shown on the chart  ← rename here
MODEL_NAMES = {
    "4b":     "Gemma 4B",
    "27b":    "Gemma 27B",
    "gpt":    "GPT-4o",
    "claude": "Claude",
}

# ---- Bar colours (hex or matplotlib colour names) ----
MODEL_COLORS = {
    "4b":     "#8B5CF6",   # purple
    "27b":    "#3B82F6",   # blue
    "gpt":    "#EF4444",   # red
    "claude": "#F59E0B",   # yellow / amber
}

# ---- Bootstrap CI ----
N_BOOTSTRAP = 10000
CI_LEVEL    = 0.95
RANDOM_SEED = 42

# ---- Figure output ----
FIGURE_DPI  = 600
FIGURE_SIZE = (2000 / FIGURE_DPI, 1200 / FIGURE_DPI)   # → (3.333, 2.00) inches @ 600 dpi = 2000×1200 px
SHOW_PLOT   = True   # True = 保存后用系统图片查看器自动打开；False = 只保存
# ---- 图像保存目录（留空则保存到脚本运行时的当前目录）----
OUTPUT_DIR  = ""     # 例如改成 r"D:\work\artifact photo\charts" 则固定存到该文件夹

# ----------------------------------------------------------------
# ★  默认输入值 — 每次运行时的预填项，直接回车即可使用默认值  ★
# ----------------------------------------------------------------
# 如果某项留空字符串 ""，运行时就必须手动输入，不能跳过。
# Mode A：横轴是模型（4 根柱子，对应 1 种图像类型）
# Mode B：横轴是图像类型（每种图像类型下画 4 个模型的柱子）
DEFAULT_MODE       = "A"              # 图表模式：填 "A" 或 "B"
DEFAULT_DATASET    = "fundus"         # 数据集类型：填 "fundus" 或 "oct"
DEFAULT_PROMPT     = "1"             # 提示词编号：填 "1"、"2"、"3" 等
DEFAULT_IMAGE_TYPE = "weakblur"       # 【仅 Mode A 用】图像类型，例如 "weakblur"、"strongblur"
DEFAULT_IMAGE_LIST = "weakblur, strongblur, mediumcolor, original"
#                                     # 【仅 Mode B 用】多个图像类型，逗号分隔
DEFAULT_ORIG_TYPE  = "original"       # 【仅 Mode C 用】基准图像类型（变化率的分母）
DEFAULT_COMP_LIST  = "weakblur, strongblur, mediumcolor"
#                                     # 【仅 Mode C 用】要与 original 比较的图像类型列表
DEFAULT_OUTPUT     = ""              # 输出文件路径，留空则自动命名
# ----------------------------------------------------------------


# ================================================================
# LABEL EXTRACTION  (handles all three JSON formats automatically)
# ================================================================

# Cache ground-truth label vocabularies to avoid re-reading CSV
_gt_label_cache: dict = {}


def _load_gt_vocab(dataset_type: str) -> set:
    csv_path = CSV_PATHS[dataset_type]
    df = pd.read_csv(csv_path, header=None)
    return set(df.iloc[:, 1].astype(str).str.strip().unique())


def _gt_vocab(dataset_type: str) -> set:
    if dataset_type not in _gt_label_cache:
        _gt_label_cache[dataset_type] = _load_gt_vocab(dataset_type)
    return _gt_label_cache[dataset_type]


def _fuzzy_match(candidate: str, vocab: set):
    """Try to match a candidate string against the known label vocabulary."""
    cand_low = candidate.lower()

    # 1. Exact (case-insensitive)
    for label in vocab:
        if cand_low == label.lower():
            return label

    # 2. Candidate starts with a known label (or vice versa)
    for label in sorted(vocab, key=len, reverse=True):
        if cand_low.startswith(label.lower()) or label.lower().startswith(cand_low):
            return label

    # 3. Substring containment
    for label in sorted(vocab, key=len, reverse=True):
        if label.lower() in cand_low or cand_low in label.lower():
            return label

    return None


def extract_label(raw_text: str, dataset_type: str):
    """
    Extract the classification label from a JSON value.

    Handles:
      Format 1  — direct label string, e.g. "Normal"
      Format 2  — **Classification: Label**\\n\\n<explanation>
      Format 3  — long free-text with the label embedded anywhere
    Returns the matched label string, or None if parsing fails.
    """
    text = str(raw_text).strip()
    vocab = _gt_vocab(dataset_type)

    # --- Format 1: direct or very short value ---
    if len(text) < 120:
        matched = _fuzzy_match(text, vocab)
        if matched:
            return matched

    # --- Format 2 & 3: regex extraction ---
    # 先去掉双引号包裹（""word"" → word），方便后面匹配
    text_clean = text.replace('""', '"')

    patterns = [
        # **Classification: Label** 或 **Classification (qualifier): Label**
        r'\*{1,2}\s*Classification(?:[^:*\n]*):\s*([^*\n]{2,80?}?)\s*\*{1,2}',
        # ## Classification: Label
        r'#{1,3}\s*Classification(?:[^:\n]*):\s*([^\n]{2,80})',
        # "Classification: Label" 普通形式
        r'\bClassification(?:[^:\n]*):\s*([^\n\*#,]{2,80})',
        # "the (most likely) classification is X"  ← Format 3 常见结论句
        r'(?:most likely\s+)?classification\s+is\s+"?([^".\n,]{2,60})"?',
        # "classified as X"
        r'classified\s+as\s+"?([^".\n,]{2,60})"?',
        # "most consistent with X"
        r'most\s+consistent\s+with\s+"?([^".\n,]{2,60})"?',
        # "the image is most likely X" / "most likely X"
        r'(?:image\s+is\s+)?most\s+likely\s+"?([^".\n,]{2,60})"?',
        # "diagnosis/assessment: X" 或 "primary classification: X"
        r'(?:diagnosis|assessment|primary\s+classification)\s*[:\-]\s*"?([^".\n,]{2,60})"?',
    ]
    for pattern in patterns:
        m = re.search(pattern, text_clean, re.IGNORECASE)
        if m:
            candidate = m.group(1).strip().strip('"').strip("'").strip('*').strip()
            matched = _fuzzy_match(candidate, vocab)
            if matched:
                return matched

    # --- 兜底扫描：在结论区域优先搜索，避免在否定句中误匹配 ---
    # 策略：先在最后 30% 的文本里找（结论通常在末尾），再搜全文
    # 同时跳过紧跟否定词之后的标签
    _NEG = re.compile(
        r'\b(?:no|not|without|absence\s+of|rule\s+out|negative\s+for|'
        r'unlikely|does\s+not|cannot|can\'t|nor)\b',
        re.IGNORECASE
    )
    text_low = text_clean.lower()
    tail = text_clean[int(len(text_clean) * 0.6):]   # 后 40% 文本

    for search_zone in (tail, text_clean):
        zone_low = search_zone.lower()
        for label in sorted(vocab, key=len, reverse=True):
            lab_low = label.lower()
            pos = zone_low.find(lab_low)
            if pos == -1:
                continue
            # 检查该位置之前 60 个字符是否有否定词
            context_before = search_zone[max(0, pos - 60): pos]
            if _NEG.search(context_before):
                continue   # 否定语境，跳过
            return label

    return None   # 无法解析


# ================================================================
# DATA LOADING
# ================================================================

def load_ground_truth(dataset_type: str) -> dict:
    """
    Load ground-truth CSV.
    Second column = label; row number (1-based) = image ID.
    Returns {image_id: label}.
    """
    df = pd.read_csv(CSV_PATHS[dataset_type], header=None)
    return {i + 1: str(df.iloc[i, 1]).strip() for i in range(len(df))}


def _regex_extract_pairs(content: str) -> dict:
    """
    逐行提取 "数字": "值" 格式的键值对，用于解析格式混乱的 JSON 文件。
    处理：
      - 键值对之间夹杂的裸文本行（直接跳过）
      - 字符串内用 ""word"" 代替标准 \"word\" 的双引号写法
    """
    result = {}
    for line in content.splitlines():
        # 匹配形如  "123": "...任意内容...",  的行（末尾逗号可选）
        m = re.match(r'\s*"(\d+)"\s*:\s*"(.+)"[,]?\s*$', line.strip())
        if m:
            key   = m.group(1)
            value = m.group(2).replace('""', '"')   # ""word"" → "word"
            result[key] = value
    return result


def load_json_results(filepath: str, dataset_type: str) -> dict:
    """
    Load a model results JSON file.
    Returns {image_id (int): predicted_label (str)}.

    解析策略（依次尝试）：
      1. 标准 json.loads()
      2. raw_decode()  — 处理末尾有多余内容的情况
      3. 正则逐行提取 — 处理键值对之间夹杂裸文本的情况
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    fname = os.path.basename(filepath)
    data  = None

    # ── 策略 1：标准解析 ────────────────────────────────────────────
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        pass

    # ── 策略 2：raw_decode（忽略末尾多余内容）──────────────────────
    if data is None:
        try:
            data, _ = json.JSONDecoder().raw_decode(content.strip())
        except json.JSONDecodeError:
            pass

    # ── 策略 3：正则逐行提取（文件格式严重混乱时）──────────────────
    if data is None:
        data = _regex_extract_pairs(content)
        if data:
            print(f"    [Info] {fname}: 使用正则回退解析器，提取到 {len(data)} 条记录")
        else:
            print(f"    [Error] {fname}: 三种解析方式均失败，跳过此文件")
            return {}

    parsed = {}
    failed = []
    for key, value in data.items():
        img_id = int(key)
        label  = extract_label(str(value), dataset_type)
        if label is not None:
            parsed[img_id] = label
        else:
            failed.append(img_id)

    if failed:
        print(f"    [Warning] {fname}: could not parse {len(failed)} entries "
              f"(IDs: {failed[:5]}{'…' if len(failed) > 5 else ''})")
    return parsed


def find_json_files(dataset_type: str, image_type: str, prompt_num: str) -> dict:
    """
    Find JSON files matching {prompt}_{dataset}_{image_type}_{model}.json.
    Returns {model_key: full_path}.
    """
    found = {}
    for model in MODEL_NAMES:
        fname    = f"{prompt_num}_{dataset_type}_{image_type}_{model}.json"
        fpath    = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(fpath):
            found[model] = fpath
        else:
            print(f"  [Missing] {fname}")
    return found


def per_sample_correctness(predictions: dict, ground_truth: dict) -> dict:
    """
    Returns {image_id: 1 (correct) | 0 (wrong)} for all common keys.
    """
    common = sorted(set(predictions.keys()) & set(ground_truth.keys()))
    return {k: int(predictions[k] == ground_truth[k]) for k in common}


# ================================================================
# STATISTICS
# ================================================================

def bootstrap_ci(correct_dict: dict):
    """
    Compute accuracy and percentile bootstrap 95 % CI.
    Returns (accuracy, ci_low, ci_high).
    """
    values = np.array(list(correct_dict.values()), dtype=float)
    accuracy = float(values.mean())

    rng = np.random.default_rng(RANDOM_SEED)
    boot = np.array([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(N_BOOTSTRAP)
    ])

    alpha   = 1.0 - CI_LEVEL
    ci_low  = float(np.percentile(boot, 100 * alpha / 2))
    ci_high = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return accuracy, ci_low, ci_high


def mcnemar_test(c1: np.ndarray, c2: np.ndarray) -> float:
    """
    McNemar's test on two aligned binary arrays.
    Uses mid-p correction when b+c < 25.
    Returns p-value.
    """
    b = int(np.sum((c1 == 1) & (c2 == 0)))   # model1 right, model2 wrong
    c = int(np.sum((c1 == 0) & (c2 == 1)))   # model1 wrong, model2 right
    n = b + c
    if n == 0:
        return 1.0
    # Exact binomial mid-p (recommended for small n)
    if n < 25:
        p = 2 * stats.binom.cdf(min(b, c), n, 0.5)
        # mid-p: subtract half the point probability
        p -= stats.binom.pmf(min(b, c), n, 0.5)
        return float(min(p, 1.0))
    # Large sample: chi-squared with continuity correction
    stat = (abs(b - c) - 1.0) ** 2 / n
    return float(1.0 - stats.chi2.cdf(stat, df=1))


# Significance level thresholds → annotation label
_SIG_TABLE = [(0.001, "***"), (0.01, "**"), (0.05, "*"), (2.0, "ns")]


def sig_label(p_raw: float, n_tests: int = 1) -> tuple:
    """Return (label, p_adjusted) with Bonferroni correction."""
    p_adj = min(p_raw * n_tests, 1.0)
    for thresh, lab in _SIG_TABLE:
        if p_adj < thresh:
            return lab, p_adj
    return "ns", p_adj


def _save_and_open(fig, output_path: str):
    """保存图像；若 SHOW_PLOT=True 则用系统默认程序自动打开（不弹 matplotlib 窗口）。"""
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    abs_path = os.path.abspath(output_path)
    print(f"\n  ✓ Saved → {abs_path}")
    if SHOW_PLOT:
        if sys.platform == 'win32':
            os.startfile(abs_path)                          # Windows：用照片/画图打开
        elif sys.platform == 'darwin':
            subprocess.run(['open', abs_path])              # macOS
        else:
            subprocess.run(['xdg-open', abs_path])         # Linux


# ================================================================
# PLOTTING — shared helpers
# ================================================================

def _no_overlap_levels(bracket_list):
    """
    bracket_list: [(x1, x2, label), ...] 按跨度从小到大排列
    返回每条括号应放置的层号，保证同层内无水平重叠。
    """
    levels = []
    occupied = []   # [(level, x1, x2), ...]
    for (x1, x2, _) in bracket_list:
        lvl = 0
        while any(ol == lvl and ox1 < x2 and x1 < ox2
                  for ol, ox1, ox2 in occupied):
            lvl += 1
        levels.append(lvl)
        occupied.append((lvl, x1, x2))
    return levels


def _draw_bracket(ax, x1, x2, y_base, bar_h, text, fontsize=9):
    """Draw an L-shaped significance bracket between two x positions."""
    ax.plot(
        [x1, x1, x2, x2],
        [y_base, y_base + bar_h, y_base + bar_h, y_base],
        lw=0.35, color='#333333', clip_on=False
    )
    ax.text(
        (x1 + x2) / 2, y_base + bar_h + 0.004,
        text, ha='center', va='bottom',
        fontsize=fontsize, color='#333333'
    )


def _sig_legend_lines():
    """Return the standard significance legend entries."""
    return [
        Line2D([], [], color='none', label=''),
        Line2D([], [], color='none', label='Significance (McNemar,'),
        Line2D([], [], color='none', label='Bonferroni corrected):'),
        Line2D([], [], color='none', label='  ***  p < 0.001'),
        Line2D([], [], color='none', label='  **   p < 0.01'),
        Line2D([], [], color='none', label='  *    p < 0.05'),
        Line2D([], [], color='none', label='  ns   p ≥ 0.05'),
        Line2D([], [], color='none', label=''),
        Line2D([], [], color='none',
               label=f'Error bars: {int(CI_LEVEL*100)}% CI (bootstrap)'),
    ]


def _apply_common_style(ax):
    """Y 轴固定刻度 0/0.25/0.50/0.75/1.00，隐藏上/右边框。ylim 由各绘图函数单独设置。"""
    ax.set_yticks([0, 0.25, 0.50, 0.75, 1.00])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.2f}'))
    ax.tick_params(axis='y', labelsize=8)
    ax.tick_params(axis='x', labelsize=8, length=0)  # length=0 隐藏 x 轴刻度线
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_alpha(0.5)
    ax.spines['bottom'].set_alpha(0.5)


# ================================================================
# MODE A — X-axis = models  (4 bars, one image type)
# ================================================================

def plot_mode_a(dataset_type: str, image_type: str,
                prompt_num: str, output_path: str = None):
    """
    Mode A: bar per model, single image type.
    Significance brackets shown for ALL model pairs.
    """
    print(f"\n{'='*58}")
    print(f"  [Mode A]  dataset={dataset_type}  image={image_type}  prompt={prompt_num}")
    print(f"{'='*58}")

    gt = load_ground_truth(dataset_type)
    print(f"  Ground truth: {len(gt)} samples | {set(gt.values())}")

    json_files = find_json_files(dataset_type, image_type, prompt_num)
    if not json_files:
        print("  ERROR: no JSON files found."); return

    # ── Per-sample correctness + CI ──────────────────────────────
    results = {}
    for model, fpath in json_files.items():
        print(f"  Loading {os.path.basename(fpath)}")
        preds   = load_json_results(fpath, dataset_type)
        correct = per_sample_correctness(preds, gt)
        acc, ci_lo, ci_hi = bootstrap_ci(correct)
        results[model] = dict(correct=correct, acc=acc, ci_lo=ci_lo, ci_hi=ci_hi)
        print(f"    {MODEL_NAMES[model]:12s}  acc={acc:.4f}  "
              f"CI=[{ci_lo:.4f},{ci_hi:.4f}]  n={len(correct)}")

    models   = list(results.keys())
    n_models = len(models)
    pairs    = list(combinations(range(n_models), 2))
    n_pairs  = len(pairs)

    # ── Pairwise McNemar ─────────────────────────────────────────
    print("\n  Pairwise McNemar tests (Bonferroni):")
    pstats = {}
    for i, j in pairs:
        m1, m2 = models[i], models[j]
        common = sorted(set(results[m1]['correct']) & set(results[m2]['correct']))
        c1 = np.array([results[m1]['correct'][k] for k in common])
        c2 = np.array([results[m2]['correct'][k] for k in common])
        p_raw = mcnemar_test(c1, c2)
        lab, p_adj = sig_label(p_raw, n_tests=n_pairs)
        pstats[(i, j)] = dict(sig=lab, p_raw=p_raw, p_adj=p_adj)
        print(f"    {MODEL_NAMES[m1]:12s} vs {MODEL_NAMES[m2]:12s}  "
              f"p={p_raw:.4f}  p_adj={p_adj:.4f}  {lab}")

    # ── Figure ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#FAFAFA')

    x         = np.arange(n_models, dtype=float)
    bar_width = 0.58

    for idx, model in enumerate(models):
        r      = results[model]
        acc    = r['acc']
        err_dn = acc - r['ci_lo']
        err_up = r['ci_hi'] - acc

        ax.bar(x[idx], acc, width=bar_width,
               color=MODEL_COLORS.get(model, '#888888'),
               alpha=0.90, edgecolor='none', linewidth=0.8, zorder=3)
        ax.errorbar(x[idx], acc, yerr=[[err_dn], [err_up]],
                    fmt='none', ecolor='#111111',
                    elinewidth=0.35, capsize=4, capthick=0.35, zorder=4)

    # ── 显著性括号（只画 * / ** / ***，ns 不画）────────────────────
    max_ci_hi = max(results[m]['ci_hi'] for m in models)
    BRACKET_H   = 0.016
    BRACKET_GAP = 0.072   # 层间距（加大避免遮挡）
    y0 = max_ci_hi + 0.04

    # 按跨度从小到大排列，只画非 *** 的对
    to_draw = sorted(
        [(x[i], x[j], pstats[(i,j)]['sig']) for (i,j) in pairs
         if pstats[(i,j)]['sig'] != '***'],
        key=lambda t: t[1] - t[0]
    )
    bracket_levels = _no_overlap_levels(to_draw)
    max_level = max(bracket_levels, default=0)

    for (bx1, bx2, sig), lvl in zip(to_draw, bracket_levels):
        _draw_bracket(ax, bx1, bx2,
                      y0 + lvl * BRACKET_GAP,
                      BRACKET_H, sig, fontsize=5)

    y_max = y0 + (max_level + 1) * BRACKET_GAP + 0.04

    # ── Axes ─────────────────────────────────────────────────────
    ax.set_xticks(x)
    ax.set_xticklabels([])          # Mode A：不显示横轴模型名，靠颜色区分
    ax.set_xlim(-0.6, n_models - 0.4)
    ax.set_ylim(0, max(1.00, y_max))
    _apply_common_style(ax)

    plt.tight_layout()

    if output_path is None:
        fname = f"chart_A_{dataset_type}_{image_type}_prompt{prompt_num}.png"
        output_path = os.path.join(OUTPUT_DIR, fname) if OUTPUT_DIR else fname
    _save_and_open(fig, output_path)


# ================================================================
# MODE B — X-axis = image types  (N groups × 4 model bars)
# ================================================================

def plot_mode_b(dataset_type: str, image_types: list, prompt_num: str,
                output_path: str = None):
    """
    Mode B: groups of bars by image type, bars within each group = models.
    Only statistically significant within-group brackets are drawn to avoid clutter.
    Bonferroni correction is applied across ALL within-group pairs combined.
    """
    print(f"\n{'='*58}")
    print(f"  [Mode B]  dataset={dataset_type}  prompt={prompt_num}")
    print(f"  Image types: {image_types}")
    print(f"{'='*58}")

    gt = load_ground_truth(dataset_type)
    print(f"  Ground truth: {len(gt)} samples | {set(gt.values())}")

    # Ordered model list (consistent left-to-right order within every group)
    all_models = list(MODEL_NAMES.keys())
    n_models   = len(all_models)
    pairs      = list(combinations(range(n_models), 2))

    # Bonferroni denominator = total pairs across all groups
    n_total_tests = len(image_types) * len(pairs)

    # ── Load all results ─────────────────────────────────────────
    # all_res[img_type][model] = {correct, acc, ci_lo, ci_hi}
    all_res: dict = {}
    for img_type in image_types:
        print(f"\n  Image type: {img_type}")
        json_files = find_json_files(dataset_type, img_type, prompt_num)
        group_res  = {}
        for model in all_models:
            if model not in json_files:
                continue
            print(f"    Loading {os.path.basename(json_files[model])}")
            preds   = load_json_results(json_files[model], dataset_type)
            correct = per_sample_correctness(preds, gt)
            acc, ci_lo, ci_hi = bootstrap_ci(correct)
            group_res[model] = dict(correct=correct, acc=acc,
                                    ci_lo=ci_lo, ci_hi=ci_hi)
            print(f"      {MODEL_NAMES[model]:12s}  acc={acc:.4f}  "
                  f"CI=[{ci_lo:.4f},{ci_hi:.4f}]  n={len(correct)}")
        all_res[img_type] = group_res

    # ── Pairwise McNemar within each group ───────────────────────
    # pstats[(img_type, i, j)] = {sig, p_raw, p_adj}
    pstats: dict = {}
    print("\n  Pairwise McNemar tests (Bonferroni across all groups):")
    for img_type in image_types:
        for i, j in pairs:
            m1, m2 = all_models[i], all_models[j]
            if m1 not in all_res[img_type] or m2 not in all_res[img_type]:
                continue
            c1d = all_res[img_type][m1]['correct']
            c2d = all_res[img_type][m2]['correct']
            common = sorted(set(c1d) & set(c2d))
            c1 = np.array([c1d[k] for k in common])
            c2 = np.array([c2d[k] for k in common])
            p_raw = mcnemar_test(c1, c2)
            lab, p_adj = sig_label(p_raw, n_tests=n_total_tests)
            pstats[(img_type, i, j)] = dict(sig=lab, p_raw=p_raw, p_adj=p_adj)
            if lab != 'ns':
                print(f"    [{img_type}] {MODEL_NAMES[m1]:12s} vs "
                      f"{MODEL_NAMES[m2]:12s}  "
                      f"p={p_raw:.4f}  p_adj={p_adj:.4f}  {lab}")

    # ── Figure layout ─────────────────────────────────────────────
    n_groups    = len(image_types)
    bar_width   = 0.12          # each individual bar (narrower)
    group_gap   = 0.22          # gap between groups (tighter)
    inner_span  = n_models * bar_width   # total width of bars in one group
    group_step  = inner_span + group_gap # center-to-center distance between groups

    # Bar offsets within a group (centered at 0)
    offsets = np.array([(k - (n_models - 1) / 2) * bar_width
                        for k in range(n_models)])
    group_centers = np.arange(n_groups) * group_step

    fig_width = max(FIGURE_SIZE[0], n_groups * 1.8)  # 每组占 1.8 寸，不会过扁长
    fig, ax = plt.subplots(figsize=(fig_width, FIGURE_SIZE[1]), dpi=FIGURE_DPI)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#FAFAFA')

    max_ci_hi = 0.0

    for g_idx, img_type in enumerate(image_types):
        for m_idx, model in enumerate(all_models):
            if model not in all_res.get(img_type, {}):
                continue
            r   = all_res[img_type][model]
            xc  = group_centers[g_idx] + offsets[m_idx]
            acc = r['acc']
            max_ci_hi = max(max_ci_hi, r['ci_hi'])

            ax.bar(xc, acc, width=bar_width * 0.88,
                   color=MODEL_COLORS.get(model, '#888888'),
                   alpha=0.90, edgecolor='none', linewidth=0.5, zorder=3)
            ax.errorbar(xc, acc,
                        yerr=[[acc - r['ci_lo']], [r['ci_hi'] - acc]],
                        fmt='none', ecolor='#111111',
                        elinewidth=0.35, capsize=2, capthick=0.35, zorder=4)

    # ── 显著性括号（组内，只画非 *** 的对）─────────────────────────
    BRACKET_H   = 0.013
    BRACKET_GAP = 0.065   # 层间距（加大避免遮挡）
    y0_bracket  = max_ci_hi + 0.04

    max_level = 0
    for g_idx, img_type in enumerate(image_types):
        to_draw = sorted(
            [(group_centers[g_idx] + offsets[i],
              group_centers[g_idx] + offsets[j],
              pstats[(img_type, i, j)]['sig'])
             for (i, j) in pairs
             if (img_type, i, j) in pstats
             and pstats[(img_type, i, j)]['sig'] != '***'],
            key=lambda t: t[1] - t[0]
        )
        bracket_levels = _no_overlap_levels(to_draw)
        for (bx1, bx2, sig), lvl in zip(to_draw, bracket_levels):
            _draw_bracket(ax, bx1, bx2,
                          y0_bracket + lvl * BRACKET_GAP,
                          BRACKET_H, sig, fontsize=4.5)
        if bracket_levels:
            max_level = max(max_level, max(bracket_levels))

    y_max = y0_bracket + (max_level + 1) * BRACKET_GAP + 0.04

    # ── Axes ─────────────────────────────────────────────────────
    ax.set_xticks(group_centers)
    # 去掉 weak/medium/strong/original 前缀，只显示类型词（blur/color/light 等）
    _prefixes = ('weak', 'medium', 'strong', 'original')
    short_labels = []
    for t in image_types:
        label = t
        for p in _prefixes:
            if t.lower().startswith(p):
                label = t[len(p):] or t   # 若去掉后为空则保留原名
                break
        short_labels.append(label)
    ax.set_xticklabels(short_labels)
    ax.set_xlim(group_centers[0] - group_step * 0.55,
                group_centers[-1] + group_step * 0.55)
    ax.set_ylim(0, max(1.00, y_max))
    _apply_common_style(ax)

    plt.tight_layout()

    if output_path is None:
        img_str = '_'.join(image_types)
        fname = f"chart_B_{dataset_type}_{img_str}_prompt{prompt_num}.png"
        output_path = os.path.join(OUTPUT_DIR, fname) if OUTPUT_DIR else fname
    _save_and_open(fig, output_path)


# ================================================================
# MODE C — 变化率柱状图  (acc(type) - acc(original))
# ================================================================

def paired_bootstrap_ci(correct_type: dict, correct_orig: dict):
    """
    配对 bootstrap CI：在相同图像 ID 上同时重采样，计算
    acc(type) - acc(orig) 的 95% CI。
    Returns (change, ci_low, ci_high).
    """
    common   = sorted(set(correct_type) & set(correct_orig))
    arr_type = np.array([correct_type[k] for k in common], dtype=float)
    arr_orig = np.array([correct_orig[k] for k in common], dtype=float)

    change = arr_type.mean() - arr_orig.mean()

    rng  = np.random.default_rng(RANDOM_SEED)
    boot = np.empty(N_BOOTSTRAP)
    for i in range(N_BOOTSTRAP):
        idx      = rng.integers(0, len(common), size=len(common))
        boot[i]  = arr_type[idx].mean() - arr_orig[idx].mean()

    alpha   = 1.0 - CI_LEVEL
    ci_low  = float(np.percentile(boot, 100 * alpha / 2))
    ci_high = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return change, ci_low, ci_high


def plot_mode_c(dataset_type: str, orig_type: str, comp_types: list,
                prompt_num: str, output_path: str = None):
    """
    Mode C: 变化率柱状图。
    X 轴 = 比较图像类型（非 original），每组 4 个模型柱子，
    高度 = acc(type, model) - acc(original, model)。
    Y 轴以 0 为中心，刻度 -0.4/-0.2/0/0.2/0.4。
    """
    print(f"\n{'='*58}")
    print(f"  [Mode C]  dataset={dataset_type}  prompt={prompt_num}")
    print(f"  Baseline: {orig_type}  vs  {comp_types}")
    print(f"{'='*58}")

    gt         = load_ground_truth(dataset_type)
    all_models = list(MODEL_NAMES.keys())
    n_models   = len(all_models)
    pairs      = list(combinations(range(n_models), 2))

    # ── 加载 original 基准 ────────────────────────────────────────
    print(f"\n  Loading baseline ({orig_type})…")
    orig_files = find_json_files(dataset_type, orig_type, prompt_num)
    orig_correct: dict = {}          # {model: {img_id: 0/1}}
    for model, fpath in orig_files.items():
        preds = load_json_results(fpath, dataset_type)
        orig_correct[model] = per_sample_correctness(preds, gt)

    # ── 加载对比图像类型 + 计算变化率 ────────────────────────────
    # all_res[img_type][model] = {change, ci_lo, ci_hi, correct_type}
    all_res: dict = {}
    for img_type in comp_types:
        print(f"\n  Image type: {img_type}")
        json_files = find_json_files(dataset_type, img_type, prompt_num)
        group: dict = {}
        for model in all_models:
            if model not in json_files or model not in orig_correct:
                continue
            preds   = load_json_results(json_files[model], dataset_type)
            cor_t   = per_sample_correctness(preds, gt)
            change, ci_lo, ci_hi = paired_bootstrap_ci(cor_t, orig_correct[model])
            group[model] = dict(correct=cor_t, change=change,
                                ci_lo=ci_lo, ci_hi=ci_hi)
            print(f"    {MODEL_NAMES[model]:12s}  Δacc={change:+.4f}  "
                  f"CI=[{ci_lo:+.4f},{ci_hi:+.4f}]")
        all_res[img_type] = group

    # ── 两两 McNemar（Bonferroni，基于 image_type 准确率）────────
    n_total_tests = len(comp_types) * len(pairs)
    pstats: dict  = {}
    print("\n  Pairwise McNemar tests (Bonferroni across all groups):")
    for img_type in comp_types:
        for i, j in pairs:
            m1, m2 = all_models[i], all_models[j]
            if m1 not in all_res[img_type] or m2 not in all_res[img_type]:
                continue
            c1d    = all_res[img_type][m1]['correct']
            c2d    = all_res[img_type][m2]['correct']
            common = sorted(set(c1d) & set(c2d))
            c1     = np.array([c1d[k] for k in common])
            c2     = np.array([c2d[k] for k in common])
            p_raw  = mcnemar_test(c1, c2)
            lab, p_adj = sig_label(p_raw, n_tests=n_total_tests)
            pstats[(img_type, i, j)] = dict(sig=lab, p_raw=p_raw, p_adj=p_adj)
            if lab != 'ns':
                print(f"    [{img_type}] {MODEL_NAMES[m1]:12s} vs "
                      f"{MODEL_NAMES[m2]:12s}  "
                      f"p={p_raw:.4f}  p_adj={p_adj:.4f}  {lab}")

    # ── Figure layout（与 Mode B 相同参数）──────────────────────
    n_groups   = len(comp_types)
    bar_width  = 0.12
    group_gap  = 0.22
    inner_span = n_models * bar_width
    group_step = inner_span + group_gap
    offsets    = np.array([(k - (n_models - 1) / 2) * bar_width
                           for k in range(n_models)])
    group_centers = np.arange(n_groups) * group_step

    fig_width = max(FIGURE_SIZE[0], n_groups * 1.8)
    fig, ax   = plt.subplots(figsize=(fig_width, FIGURE_SIZE[1]), dpi=FIGURE_DPI)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#FAFAFA')

    # ── 画柱子 ───────────────────────────────────────────────────
    max_abs_ci = 0.0
    for g_idx, img_type in enumerate(comp_types):
        for m_idx, model in enumerate(all_models):
            if model not in all_res.get(img_type, {}):
                continue
            r      = all_res[img_type][model]
            xc     = group_centers[g_idx] + offsets[m_idx]
            change = r['change']
            err_dn = change - r['ci_lo']
            err_up = r['ci_hi'] - change
            max_abs_ci = max(max_abs_ci, abs(r['ci_lo']), abs(r['ci_hi']))

            ax.bar(xc, change, width=bar_width * 0.88,
                   color=MODEL_COLORS.get(model, '#888888'),
                   alpha=0.90, edgecolor='none', zorder=3)
            ax.errorbar(xc, change,
                        yerr=[[err_dn], [err_up]],
                        fmt='none', ecolor='#111111',
                        elinewidth=0.7, capsize=2, capthick=0.7, zorder=4)

    # ── 显著性括号（只画非 ***，同 Mode B）──────────────────────
    BRACKET_H   = 0.013
    BRACKET_GAP = 0.065
    # 括号画在正值方向上方（或最高 CI 上方）
    y_bracket_base = max(max_abs_ci + 0.04, 0.04)

    max_level = 0
    for g_idx, img_type in enumerate(comp_types):
        to_draw = sorted(
            [(group_centers[g_idx] + offsets[i],
              group_centers[g_idx] + offsets[j],
              pstats[(img_type, i, j)]['sig'])
             for (i, j) in pairs
             if (img_type, i, j) in pstats
             and pstats[(img_type, i, j)]['sig'] != '***'],
            key=lambda t: t[1] - t[0]
        )
        lvls = _no_overlap_levels(to_draw)
        for (bx1, bx2, sig), lvl in zip(to_draw, lvls):
            _draw_bracket(ax, bx1, bx2,
                          y_bracket_base + lvl * BRACKET_GAP,
                          BRACKET_H, sig, fontsize=4.5)
        if lvls:
            max_level = max(max_level, max(lvls))

    # ── 0 基准线 ─────────────────────────────────────────────────
    ax.axhline(0, color='#555555', linewidth=0.8, zorder=2)

    # ── Axes ─────────────────────────────────────────────────────
    ax.set_xticks(group_centers)
    _prefixes = ('weak', 'medium', 'strong', 'original')
    short_labels = []
    for t in comp_types:
        label = t
        for p in _prefixes:
            if t.lower().startswith(p):
                label = t[len(p):] or t
                break
        short_labels.append(label)
    ax.set_xticklabels(short_labels)
    ax.set_xlim(group_centers[0] - group_step * 0.55,
                group_centers[-1] + group_step * 0.55)

    # Y 轴：以 0 为中心，固定刻度，上下各留空间给括号和负 CI
    y_top = y_bracket_base + (max_level + 1) * BRACKET_GAP + 0.04
    y_top = max(y_top, 0.25)
    y_bot = min(-max_abs_ci - 0.04, -0.25)
    ax.set_ylim(y_bot, y_top)
    ax.set_yticks([-0.2, -0.1, 0, 0.1, 0.2])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:+.2f}'))

    ax.tick_params(axis='y', labelsize=8)
    ax.tick_params(axis='x', labelsize=8, length=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_alpha(0.5)
    ax.spines['bottom'].set_visible(False)   # 用 axhline 代替底边框

    plt.tight_layout()

    if output_path is None:
        comp_str = '_'.join(comp_types)
        fname = f"chart_C_{dataset_type}_{comp_str}_vs_{orig_type}_prompt{prompt_num}.png"
        output_path = os.path.join(OUTPUT_DIR, fname) if OUTPUT_DIR else fname
    _save_and_open(fig, output_path)


# ================================================================
# MAIN — interactive CLI
# ================================================================

def _ask(msg: str, default: str = "") -> str:
    """打印提示并读取输入；直接回车则使用 default 默认值。"""
    hint = f" [默认: {default}]" if default else ""
    raw = input(f"  {msg}{hint}: ").strip()
    return raw if raw else default


def main():
    print("=" * 58)
    print("  Medical Image Classification Chart Generator")
    print("=" * 58)
    print(f"  Results dir : {RESULTS_DIR}")
    print(f"  Models      : {', '.join(MODEL_NAMES.values())}")
    print()
    print("  Chart modes:")
    print("    A — 横轴是模型      (4 根柱子，1 种图像类型)")
    print("    B — 横轴是图像类型  (N 组 × 4 模型柱子，绝对准确率)")
    print("    C — 横轴是图像类型  (N 组 × 4 模型柱子，相对 original 的变化率)")
    print()
    print("  直接回车使用方括号里的默认值；")
    print("  默认值在脚本顶部 DEFAULT_* 变量处修改。")
    print()

    mode = _ask("Mode (A / B / C)", DEFAULT_MODE).upper()
    if mode not in ('A', 'B', 'C'):
        print("  ERROR: 只能填 A、B 或 C"); return

    dataset = _ask("Dataset type (fundus / oct)", DEFAULT_DATASET).lower()
    if dataset not in CSV_PATHS:
        print(f"  ERROR: 未知数据集 '{dataset}'，可选: {list(CSV_PATHS.keys())}"); return

    prompt_n = _ask("Prompt number (提示词编号)", DEFAULT_PROMPT)
    out_file = _ask("Output path   (留空=自动命名)", DEFAULT_OUTPUT) or None

    if mode == 'A':
        img_type = _ask("Image type (图像类型)", DEFAULT_IMAGE_TYPE)
        plot_mode_a(dataset, img_type, prompt_n, out_file)

    elif mode == 'B':
        print("  多个图像类型用逗号分隔，例如: weakblur, strongblur, mediumcolor")
        raw = _ask("Image types (图像类型列表)", DEFAULT_IMAGE_LIST)
        img_types = [t.strip() for t in raw.split(',') if t.strip()]
        if not img_types:
            print("  ERROR: 未提供任何图像类型"); return
        plot_mode_b(dataset, img_types, prompt_n, out_file)

    else:  # Mode C
        orig = _ask("Baseline image type (基准类型，如 original)", DEFAULT_ORIG_TYPE)
        print("  比较图像类型用逗号分隔，例如: weakblur, strongblur, mediumcolor")
        raw  = _ask("Comparison image types (对比类型列表)", DEFAULT_COMP_LIST)
        comp_types = [t.strip() for t in raw.split(',') if t.strip()]
        if not comp_types:
            print("  ERROR: 未提供任何对比图像类型"); return
        plot_mode_c(dataset, orig, comp_types, prompt_n, out_file)


if __name__ == "__main__":
    main()
