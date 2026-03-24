"""
calculate_accuracy.py
---------------------
用途：计算眼科疾病图像分类模型的准确度。

输入：
  1. xlsx 文件 —— 人工标注的正确分类，第一列为疾病缩写：
       NORMAL  → normal（正常）
       DR      → diabetic retinopathy（糖尿病视网膜病变）
       MH      → macular hole（黄斑裂孔）
       AMRD    → age-related macular degeneration（年龄相关性黄斑变性）
       CSR     → central serous retinopathy（中心性浆液性脉络膜视网膜病变）

  2. json 文件 —— 模型输出的分类结果，格式为以图像编号（从 1 开始）为键、
     疾病全称（或含疾病名称的长文本）为值的字典，例如：
       {
         "1": "normal",
         "2": "Based on the OCT image, the patient has diabetic retinopathy ...",
         ...
       }
     程序会自动从长文本中提取疾病名称。若文本中包含多个疾病名称，取最先出现的一个。

输出：
  - 总体准确度（Overall Accuracy）
  - 可选：每类疾病的单独准确度（--per-class 标志）

用法示例：
  python calculate_accuracy.py ground_truth.xlsx predictions.json
  python calculate_accuracy.py ground_truth.xlsx predictions.json --per-class
"""

import json
import re
import argparse
import pandas as pd

# 疾病缩写到全称的映射表
# xlsx 文件中使用大写缩写，模型输出及比较时统一使用小写全称
LABEL_MAP = {
    "NORMAL": "normal",                          # 正常
    "DR":     "diabetic retinopathy",            # 糖尿病视网膜病变
    "MH":     "macular hole",                    # 黄斑裂孔
    "AMRD":   "age-related macular degeneration", # 年龄相关性黄斑变性
    "CSR":    "central serous retinopathy",      # 中心性浆液性脉络膜视网膜病变
}

# 所有合法的疾病全称，按长度从长到短排列。
# 优先匹配长字符串，防止 "normal" 误匹配 "age-related macular degeneration" 等
# 包含 "normal" 子串的情况（实际上不存在，但保持此顺序是良好实践）。
DISEASE_NAMES = sorted(LABEL_MAP.values(), key=len, reverse=True)


def extract_disease(text):
    """
    从任意长度的文本中提取疾病名称。

    匹配规则：
      - 忽略大小写
      - 按疾病名称从长到短依次搜索，取文本中最先出现的匹配项
      - 若文本本身就是合法疾病名（精确匹配），直接返回，不做搜索

    参数：
        text (str): 模型输出的原始字符串，可以是疾病名或含疾病名的长句。

    返回：
        str: 匹配到的疾病全称（小写），如 "normal"、"macular hole" 等。

    异常：
        ValueError: 文本中找不到任何已知疾病名称时抛出。
    """
    text_lower = text.strip().lower()

    # 快速路径：文本本身就是合法疾病名，直接返回
    if text_lower in DISEASE_NAMES:
        return text_lower

    # 在文本中搜索每个疾病名称出现的位置，记录 (位置, 疾病名) 对
    matches = []
    for disease in DISEASE_NAMES:
        pos = text_lower.find(disease)
        if pos != -1:
            matches.append((pos, disease))

    if not matches:
        raise ValueError(
            f"无法从以下文本中提取疾病名称，请检查模型输出：\n  '{text}'"
        )

    # 取最先出现（位置最小）的匹配结果
    matches.sort(key=lambda x: x[0])
    return matches[0][1]


def _parse_json_with_fallback(json_path):
    """
    读取 JSON 文件，若标准解析失败则启用正则表达式兜底解析。

    兜底场景：模型输出的长文本中包含未转义的双引号、换行符等特殊字符，
    导致 JSON 格式损坏，标准 json.load() 抛出 JSONDecodeError。

    兜底策略：
      用正则从文件原始文本中逐条提取 "编号": "任意内容" 的键值对。
      匹配规则为：找到数字键后，将其后的内容一直读到下一个数字键或文件结尾，
      再从这段内容中直接提取疾病名称，绕过 JSON 字符串边界问题。

    参数：
        json_path (str): JSON 文件路径。

    返回：
        dict: {编号字符串: 原始文本字符串} 的字典。

    异常：
        ValueError: 兜底解析也未能提取到任何条目时抛出。
    """
    with open(json_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # 第一步：尝试标准 JSON 解析
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError as e:
        print(f"[警告] JSON 格式有误（{e}），启用正则兜底解析……")

    # 第二步：正则兜底
    # 匹配 "数字" : " ... " 结构，值部分跨行贪婪匹配至下一个 "数字" 键或文件末尾
    # 直接用 [\s\S]*? 非贪婪匹配两个键之间的所有内容（包含换行）
    pattern = re.compile(
        r'"(\d+)"\s*:\s*"([\s\S]*?)"'   # 匹配 "key": "value"
        r'(?=\s*(?:,\s*"\d+"|\s*\}))',   # 向前断言：后面跟着下一个键或 }
    )
    matches = pattern.findall(raw_text)

    if not matches:
        # 更宽松的备用正则：直接在数字键之间切割，不依赖闭合引号
        # 每段切割后再用 extract_disease 从原始文本中提取疾病名
        chunk_pattern = re.compile(r'"(\d+)"\s*:\s*"([\s\S]*?)"', re.MULTILINE)
        matches = chunk_pattern.findall(raw_text)

    if not matches:
        raise ValueError(
            "标准 JSON 解析与正则兜底均失败，请手动检查 JSON 文件格式。"
        )

    print(f"[信息] 正则兜底解析成功，共提取到 {len(matches)} 条记录。")
    return {k: v for k, v in matches}


def load_ground_truth(xlsx_path):
    """
    从 xlsx 文件读取人工标注的正确标签。

    参数：
        xlsx_path (str): xlsx 文件路径，第一列为疾病缩写（如 NORMAL、DR 等）。

    返回：
        list[str]: 按行顺序排列的疾病全称列表，例如 ["normal", "macular hole", ...]。

    异常：
        ValueError: 若遇到不在 LABEL_MAP 中的未知缩写则抛出。
    """
    # header=None 表示第一行也是数据，不作为列名
    df = pd.read_excel(xlsx_path, header=None)

    labels = []
    for val in df.iloc[:, 0]:           # 只取第一列
        abbr = str(val).strip().upper() # 统一转大写并去除首尾空格
        if abbr not in LABEL_MAP:
            raise ValueError(f"未知的疾病缩写：'{abbr}'，请检查 xlsx 文件内容。")
        labels.append(LABEL_MAP[abbr])  # 转换为全称后加入列表

    return labels


def load_predictions(json_path):
    """
    从 json 文件读取模型的预测结果，并从每条文本中提取疾病名称。

    json 文件格式要求：键为图像编号字符串（从 "1" 开始），值为模型输出的文本
    （可以是疾病全称，也可以是含疾病名称的长文本）。
    例如：
      {"1": "normal", "2": "Based on the image, the diagnosis is macular hole ..."}

    参数：
        json_path (str): json 文件路径。

    返回：
        list[str]: 按图像编号从小到大排列的预测标签列表（均为标准疾病全称）。

    异常：
        ValueError: 某条文本中无法提取到已知疾病名称时抛出，并提示对应编号。
    """
    # 使用兜底解析：JSON 格式损坏时自动切换为正则提取
    raw = _parse_json_with_fallback(json_path)

    # json 中的键为字符串，需转为整数后排序，确保顺序与 xlsx 行顺序一致
    predictions = []
    for k in sorted(raw.keys(), key=lambda x: int(x)):
        text = raw[str(k)]
        try:
            disease = extract_disease(text)
        except ValueError as e:
            raise ValueError(f"图像编号 {k}：{e}") from e
        predictions.append(disease)

    return predictions


def calculate_accuracy(ground_truth, predictions):
    """
    计算整体分类准确度。

    准确度 = 预测正确的样本数 / 总样本数

    参数：
        ground_truth (list[str]): 正确标签列表。
        predictions  (list[str]): 模型预测标签列表。

    返回：
        tuple: (correct, total, accuracy)
            - correct  (int):   预测正确的样本数
            - total    (int):   总样本数
            - accuracy (float): 准确度，范围 [0, 1]

    异常：
        ValueError: 若两个列表长度不一致则抛出。
    """
    if len(ground_truth) != len(predictions):
        raise ValueError(
            f"样本数量不匹配：xlsx 共 {len(ground_truth)} 条，"
            f"json 共 {len(predictions)} 条，请检查两个文件是否对应同一批图像。"
        )

    # 逐一比较每个样本的预测结果与正确标签
    correct = sum(gt == pred for gt, pred in zip(ground_truth, predictions))
    total = len(ground_truth)
    accuracy = correct / total
    return correct, total, accuracy


def per_class_accuracy(ground_truth, predictions):
    """
    计算每类疾病的分类准确度。

    参数：
        ground_truth (list[str]): 正确标签列表。
        predictions  (list[str]): 模型预测标签列表。

    返回：
        dict: 键为疾病全称，值为 (correct, total, accuracy) 三元组。
              例如：{"normal": (9, 10, 0.9), "macular hole": (8, 10, 0.8), ...}
    """
    # 从正确标签中提取所有出现过的疾病类别，并排序以便输出稳定
    classes = sorted(set(ground_truth))

    results = {}
    for cls in classes:
        # 找出该类别在 ground_truth 中的所有样本位置
        indices = [i for i, gt in enumerate(ground_truth) if gt == cls]
        # 统计这些位置中预测正确的数量
        correct = sum(predictions[i] == cls for i in indices)
        results[cls] = (correct, len(indices), correct / len(indices))

    return results


def main():
    """
    命令行入口函数。

    解析命令行参数，依次调用数据加载、准确度计算和结果输出。
    """
    parser = argparse.ArgumentParser(
        description="计算眼科疾病图像分类模型的准确度。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python calculate_accuracy.py ground_truth.xlsx predictions.json\n"
            "  python calculate_accuracy.py ground_truth.xlsx predictions.json --per-class"
        ),
    )
    parser.add_argument("xlsx", help="人工标注的正确分类文件路径（.xlsx），第一列为疾病缩写")
    parser.add_argument("json", help="模型预测结果文件路径（.json），键为图像编号，值为疾病全称")
    parser.add_argument(
        "--per-class",
        action="store_true",
        help="同时输出每类疾病的单独准确度",
    )
    args = parser.parse_args()

    # 加载数据
    ground_truth = load_ground_truth(args.xlsx)
    predictions  = load_predictions(args.json)

    # 计算并输出总体准确度
    correct, total, accuracy = calculate_accuracy(ground_truth, predictions)
    print(f"Overall Accuracy: {correct}/{total} = {accuracy:.4f} ({accuracy * 100:.2f}%)")

    # 若指定 --per-class，则额外输出每类准确度
    if args.per_class:
        print("\nPer-class Accuracy:")
        per_class = per_class_accuracy(ground_truth, predictions)
        for cls, (c, t, acc) in per_class.items():
            print(f"  {cls}: {c}/{t} = {acc:.4f} ({acc * 100:.2f}%)")


if __name__ == "__main__":
    main()
