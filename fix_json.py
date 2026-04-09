#!/usr/bin/env python3
"""
fix_json.py — 修复格式混乱的 JSON 结果文件
==========================================
处理以下两类问题：
  1. 键值对之间夹杂裸露文本行（the image shows...、here's why: 等）
  2. 单条答案内部有换行符，导致逐行解析失误

算法：
  - 以 "数字": 作为分隔标记，将整个文件内容按标记切分
  - 两个标记之间的全部内容（包括换行）均属于前一个键的值
  - 提取出值字符串后折叠空白、去掉引号残迹，写成规范 JSON

用法：
  python fix_json.py                        # 处理 RESULTS_DIR 下所有 .json
  python fix_json.py path/to/file.json      # 只处理指定文件
  python fix_json.py --dir path/to/folder   # 处理指定文件夹

默认：原地覆盖（先备份到 .json.bak）
"""

import os
import re
import sys
import json
import shutil
import argparse

# ── 改成你的结果文件夹路径 ──────────────────────────────────────────
RESULTS_DIR = r"D:\work\artifact photo\moderesults"
# ──────────────────────────────────────────────────────────────────


def extract_pairs(content: str) -> dict:
    """
    按 "数字": 标记分割全文，提取所有键值对。
    值内部的换行、多余空白会被折叠成单个空格。
    """
    # 找到所有 "digit": 的位置
    key_pattern = re.compile(r'"(\d+)"\s*:')
    matches = list(key_pattern.finditer(content))

    if not matches:
        return {}

    result = {}
    for idx, match in enumerate(matches):
        key = match.group(1)

        # 值：从本标记结束到下一标记开始之间的全部文本
        val_start = match.end()
        val_end   = matches[idx + 1].start() if idx + 1 < len(matches) else len(content)
        raw_value = content[val_start:val_end]

        # 清理：去掉首尾空白、逗号、引号
        raw_value = raw_value.strip().rstrip(',').strip()

        if raw_value.startswith('"'):
            raw_value = raw_value[1:]
        if raw_value.endswith('"'):
            raw_value = raw_value[:-1]

        # 处理 ""word"" 双引号写法 → 单引号
        raw_value = raw_value.replace('""', '"')

        # 把所有内部换行、连续空白折叠成单个空格
        raw_value = ' '.join(raw_value.split())

        result[key] = raw_value

    return result


def fix_file(filepath: str, backup: bool = True) -> bool:
    """
    修复单个文件。
    返回 True 表示成功（已修复或本来就是合法 JSON），False 表示失败。
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 先试标准解析
    try:
        data = json.loads(content)
        print(f"  [OK]      {os.path.basename(filepath)} — 格式正常，无需修复")
        return True
    except json.JSONDecodeError:
        pass

    # 尝试 raw_decode（末尾有多余内容）
    try:
        data, _ = json.JSONDecoder().raw_decode(content.strip())
        print(f"  [OK]      {os.path.basename(filepath)} — raw_decode 可读，无需修复")
        return True
    except json.JSONDecodeError:
        pass

    # 用标记分割法提取键值对
    data = extract_pairs(content)
    if not data:
        print(f"  [FAILED]  {os.path.basename(filepath)} — 无法提取任何键值对，跳过")
        return False

    # 备份原文件
    if backup:
        shutil.copy2(filepath, filepath + '.bak')

    # 写回规范 JSON
    clean_json = json.dumps(data, ensure_ascii=False, indent=2)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(clean_json)

    print(f"  [FIXED]   {os.path.basename(filepath)} — 提取 {len(data)} 条，已覆盖（备份: .bak）")
    return True


def fix_directory(dirpath: str):
    files = [f for f in os.listdir(dirpath) if f.endswith('.json')]
    if not files:
        print(f"  目录 {dirpath} 下没有 .json 文件")
        return

    print(f"\n扫描目录: {dirpath}  ({len(files)} 个 .json 文件)\n")
    ok = fail = 0
    for fname in sorted(files):
        success = fix_file(os.path.join(dirpath, fname))
        if success:
            ok += 1
        else:
            fail += 1

    print(f"\n完成：成功 {ok} 个，失败 {fail} 个")


def main():
    parser = argparse.ArgumentParser(description='Fix malformed JSON result files')
    parser.add_argument('files', nargs='*', help='指定要修复的 .json 文件')
    parser.add_argument('--dir', default=None, help='处理指定文件夹（默认用 RESULTS_DIR）')
    parser.add_argument('--no-backup', action='store_true', help='不备份原文件')
    args = parser.parse_args()

    backup = not args.no_backup

    if args.files:
        for f in args.files:
            fix_file(f, backup=backup)
    else:
        target_dir = args.dir or RESULTS_DIR
        fix_directory(target_dir)


if __name__ == '__main__':
    main()
