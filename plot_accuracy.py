#!/usr/bin/env python3
"""
Medical Image Classification Bar Chart Generator
=================================================
Draws accuracy bar charts with 95% bootstrap CI and pairwise McNemar tests
for comparing model performance across different experimental conditions.

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
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # Remove this line if you want an interactive popup window
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
FIGURE_DPI  = 300
FIGURE_SIZE = (11, 7)   # inches (width, height)


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
    patterns = [
        # **Classification: Label** or **Classification (qualifier): Label**
        r'\*{1,2}\s*Classification(?:[^:*\n]*):\s*([^*\n]{2,80?}?)\s*\*{1,2}',
        # ## Classification: Label
        r'#{1,3}\s*Classification(?:[^:\n]*):\s*([^\n]{2,80})',
        # Plain "Classification: Label" (first occurrence)
        r'\bClassification(?:[^:\n]*):\s*([^\n\*#,]{2,80})',
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            candidate = m.group(1).strip().strip('*').strip('#').strip()
            matched = _fuzzy_match(candidate, vocab)
            if matched:
                return matched

    # --- Last resort: scan full text for known labels (longest match wins) ---
    text_low = text.lower()
    for label in sorted(vocab, key=len, reverse=True):
        if label.lower() in text_low:
            return label

    return None   # Could not parse


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


def load_json_results(filepath: str, dataset_type: str) -> dict:
    """
    Load a model results JSON file.
    Returns {image_id (int): predicted_label (str)}.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

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
        fname = os.path.basename(filepath)
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


# ================================================================
# PLOTTING
# ================================================================

def _draw_bracket(ax, x1, x2, y_base, bar_h, text, fontsize=9):
    """Draw an L-shaped significance bracket between two x positions."""
    ax.plot(
        [x1, x1, x2, x2],
        [y_base, y_base + bar_h, y_base + bar_h, y_base],
        lw=1.3, color='#333333', clip_on=False
    )
    ax.text(
        (x1 + x2) / 2, y_base + bar_h + 0.004,
        text, ha='center', va='bottom',
        fontsize=fontsize, color='#333333'
    )


def plot_bar_chart(dataset_type: str, image_type: str,
                   prompt_num: str, output_path: str = None):
    """Load data, run statistics, and render the bar chart."""

    print(f"\n{'='*58}")
    print(f"  dataset={dataset_type}  image_type={image_type}  prompt={prompt_num}")
    print(f"{'='*58}")

    # ── 1. Ground truth ──────────────────────────────────────────
    print("\n[1/4] Loading ground truth…")
    gt = load_ground_truth(dataset_type)
    print(f"  {len(gt)} samples  |  labels: {set(gt.values())}")

    # ── 2. Model predictions ─────────────────────────────────────
    print("\n[2/4] Locating JSON files…")
    json_files = find_json_files(dataset_type, image_type, prompt_num)
    if not json_files:
        print("  ERROR: no JSON files found — check inputs.")
        return

    # ── 3. Per-sample correctness + bootstrap CI ─────────────────
    print("\n[3/4] Computing accuracy & bootstrap CI…")
    results = {}
    for model, fpath in json_files.items():
        print(f"  {os.path.basename(fpath)}")
        preds   = load_json_results(fpath, dataset_type)
        correct = per_sample_correctness(preds, gt)
        acc, ci_lo, ci_hi = bootstrap_ci(correct)
        results[model] = dict(correct=correct, acc=acc, ci_lo=ci_lo, ci_hi=ci_hi)
        print(f"    {MODEL_NAMES[model]:12s}  acc={acc:.4f}  "
              f"95%CI=[{ci_lo:.4f},{ci_hi:.4f}]  n={len(correct)}")

    models   = list(results.keys())
    n_models = len(models)

    # ── 4. Pairwise McNemar (Bonferroni) ─────────────────────────
    print("\n[4/4] Pairwise McNemar tests (Bonferroni)…")
    pairs   = list(combinations(range(n_models), 2))
    n_pairs = len(pairs)
    pstats  = {}    # (i,j) -> {sig, p_raw, p_adj}

    for i, j in pairs:
        m1, m2   = models[i], models[j]
        common   = sorted(set(results[m1]['correct']) & set(results[m2]['correct']))
        c1       = np.array([results[m1]['correct'][k] for k in common])
        c2       = np.array([results[m2]['correct'][k] for k in common])
        p_raw    = mcnemar_test(c1, c2)
        lab, p_adj = sig_label(p_raw, n_tests=n_pairs)
        pstats[(i, j)] = dict(sig=lab, p_raw=p_raw, p_adj=p_adj)
        print(f"  {MODEL_NAMES[m1]:12s} vs {MODEL_NAMES[m2]:12s}  "
              f"p={p_raw:.4f}  p_adj={p_adj:.4f}  {lab}")

    # ── Figure ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#FAFAFA')

    x         = np.arange(n_models)
    bar_width = 0.52
    max_ci_hi = 0.0

    for idx, model in enumerate(models):
        r   = results[model]
        acc = r['acc']
        err_dn = acc - r['ci_lo']
        err_up = r['ci_hi'] - acc
        max_ci_hi = max(max_ci_hi, r['ci_hi'])

        # Bar
        ax.bar(
            x[idx], acc,
            width=bar_width,
            color=MODEL_COLORS.get(model, '#888888'),
            alpha=0.90,
            edgecolor='white',
            linewidth=0.8,
            zorder=3,
        )
        # Error bar (CI)
        ax.errorbar(
            x[idx], acc,
            yerr=[[err_dn], [err_up]],
            fmt='none',
            ecolor='#111111',
            elinewidth=2.2,
            capsize=8,
            capthick=2.2,
            zorder=4,
        )
        # Accuracy label on top
        ax.text(
            x[idx], r['ci_hi'] + 0.013,
            f'{acc:.3f}',
            ha='center', va='bottom',
            fontsize=11, fontweight='bold', color='#111111',
            zorder=5,
        )

    # Significance brackets — stack by span distance
    BRACKET_H   = 0.018   # height of the bracket's horizontal bar
    BRACKET_GAP = 0.050   # vertical gap between stacking levels
    y0 = max_ci_hi + 0.060

    # Group pairs by bar-span distance; draw closest pairs at the bottom
    dist_groups: dict = {}
    for (i, j) in pairs:
        dist_groups.setdefault(j - i, []).append((i, j))

    level = 0
    for dist in sorted(dist_groups):
        for (i, j) in dist_groups[dist]:
            y = y0 + level * BRACKET_GAP
            _draw_bracket(ax, x[i], x[j], y, BRACKET_H,
                          pstats[(i, j)]['sig'], fontsize=9)
            level += 1

    # ── Axis formatting ──────────────────────────────────────────
    ax.set_xticks(x)
    ax.set_xticklabels(
        [MODEL_NAMES[m] for m in models],
        fontsize=13, fontweight='bold'
    )
    ax.tick_params(axis='y', labelsize=11)
    ax.set_ylabel('Accuracy', fontsize=13, labelpad=8)
    ax.set_xlim(-0.55, n_models - 0.45)
    ax.set_ylim(0, y0 + level * BRACKET_GAP + 0.08)

    ax.set_title(
        f"Model Accuracy Comparison\n"
        f"Dataset: {dataset_type.upper()}   |   Image type: {image_type}   |   Prompt: {prompt_num}",
        fontsize=14, fontweight='bold', pad=14
    )
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.2f}'))
    ax.grid(axis='y', linestyle='--', alpha=0.35, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_alpha(0.5)
    ax.spines['bottom'].set_alpha(0.5)

    # ── Legend ───────────────────────────────────────────────────
    model_patches = [
        mpatches.Patch(
            color=MODEL_COLORS.get(m, '#888888'), alpha=0.90,
            label=MODEL_NAMES[m]
        )
        for m in models
    ]
    note_lines = [
        Line2D([], [], color='none', label=''),
        Line2D([], [], color='none', label='Significance (McNemar test,'),
        Line2D([], [], color='none', label='Bonferroni corrected):'),
        Line2D([], [], color='none', label='  ***  p < 0.001'),
        Line2D([], [], color='none', label='  **   p < 0.01'),
        Line2D([], [], color='none', label='  *    p < 0.05'),
        Line2D([], [], color='none', label='  ns   p ≥ 0.05'),
        Line2D([], [], color='none', label=''),
        Line2D([], [], color='none',
               label=f'Error bars: {int(CI_LEVEL*100)}% CI (bootstrap)'),
    ]
    ax.legend(
        handles=model_patches + note_lines,
        loc='upper right',
        fontsize=9,
        framealpha=0.90,
        edgecolor='#cccccc',
        handlelength=1.2,
    )

    plt.tight_layout()

    # ── Save ─────────────────────────────────────────────────────
    if output_path is None:
        output_path = f"chart_{dataset_type}_{image_type}_prompt{prompt_num}.png"

    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight', facecolor='white')
    print(f"\n  ✓ Saved → {output_path}")
    plt.close(fig)


# ================================================================
# MAIN — interactive CLI
# ================================================================

def _prompt(msg: str, default: str = None) -> str:
    suffix = f" [{default}]" if default else ""
    raw = input(f"  {msg}{suffix}: ").strip()
    return raw if raw else (default or raw)


def main():
    print("=" * 58)
    print("  Medical Image Classification Chart Generator")
    print("=" * 58)
    print(f"  Results dir : {RESULTS_DIR}")
    print(f"  Models      : {', '.join(MODEL_NAMES.values())}\n")

    dataset   = _prompt("Dataset type  (fundus / oct)").lower()
    if dataset not in CSV_PATHS:
        print(f"  ERROR: unknown dataset '{dataset}'. "
              f"Must be one of {list(CSV_PATHS.keys())}"); return

    img_type  = _prompt("Image type    (e.g. weakblur, strongblur, mediumcolor)")
    prompt_n  = _prompt("Prompt number (e.g. 1, 2, 3)")
    out_file  = _prompt("Output path   (Enter = auto-name)", default="") or None

    plot_bar_chart(dataset, img_type, prompt_n, out_file)


if __name__ == "__main__":
    main()
