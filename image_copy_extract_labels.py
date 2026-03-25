"""
功能：
1. 将 D:\work\color\Original\label 中的图片复制到 D:\work\color\Original\nolabel，
   文件名只保留编号（去掉英文标签部分）
2. 将 label 文件夹中的图片标签按顺序提取并保存到 D:\work\color\labels.xlsx
"""

import os
import re
import shutil
import openpyxl
from pathlib import Path


# ── 路径配置 ──────────────────────────────────────────────────────────────────
SRC_DIR    = Path(r"D:\work\color\Original\label")
DST_DIR    = Path(r"D:\work\color\Original\nolabel")
EXCEL_PATH = Path(r"D:\work\color\labels.xlsx")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def extract_number_and_label(stem: str):
    """
    从文件名主干中分离编号与标签。

    假设命名规则为：编号在前，英文标签（由字母/空格/下划线/连字符组成）在后，
    例如：
      "001_normal"  -> number="001", label="normal"
      "042 artifact" -> number="042", label="artifact"
      "123-blur-motion" -> number="123", label="blur-motion"
      "007"         -> number="007", label=""

    编号部分：连续数字（可含前导零）
    标签部分：其余非数字内容（去掉分隔符后的英文字符串）
    """
    # 匹配：开头的数字串，后面可选地跟着分隔符+其余内容
    m = re.match(r'^(\d+)([\s_\-]*)(.*)$', stem)
    if m:
        number = m.group(1)
        label  = m.group(3).strip()
    else:
        # 找不到数字就整体当做标签，编号留空
        number = stem
        label  = ""
    return number, label


def main():
    if not SRC_DIR.exists():
        print(f"[错误] 源目录不存在：{SRC_DIR}")
        return

    DST_DIR.mkdir(parents=True, exist_ok=True)

    # 按文件名排序，保证顺序
    files = sorted(
        [f for f in SRC_DIR.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
    )

    if not files:
        print("[警告] 源目录中没有找到图片文件。")
        return

    records = []  # [(编号, 标签, 原文件名, 新文件名)]

    for src_file in files:
        stem = src_file.stem
        ext  = src_file.suffix

        number, label = extract_number_and_label(stem)

        new_name = number + ext
        dst_file = DST_DIR / new_name

        # 若目标已存在同名文件则追加后缀避免覆盖
        counter = 1
        while dst_file.exists():
            new_name = f"{number}_{counter}{ext}"
            dst_file = DST_DIR / new_name
            counter += 1

        shutil.copy2(src_file, dst_file)
        records.append((number, label, src_file.name, new_name))
        print(f"  复制: {src_file.name}  ->  {new_name}  (标签: {label!r})")

    # ── 写入 Excel ────────────────────────────────────────────────────────────
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Labels"

    headers = ["序号", "编号", "标签", "原文件名", "新文件名"]
    ws.append(headers)

    # 加粗表头
    for cell in ws[1]:
        cell.font = openpyxl.styles.Font(bold=True)

    for idx, (number, label, orig, new) in enumerate(records, start=1):
        ws.append([idx, number, label, orig, new])

    # 自动调整列宽
    for col in ws.columns:
        max_len = max(len(str(cell.value)) if cell.value else 0 for cell in col)
        ws.column_dimensions[col[0].column_letter].width = max_len + 4

    EXCEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    wb.save(EXCEL_PATH)

    print(f"\n完成！共处理 {len(records)} 张图片。")
    print(f"  无标签图片保存至：{DST_DIR}")
    print(f"  标签表格保存至：{EXCEL_PATH}")


if __name__ == "__main__":
    main()
