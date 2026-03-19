"""
OCT image classification using MedGemma1.5:4b via Ollama.
- Reads JPEG images from D:\work\oct\nolabel
- Processes in random order, one at a time
- Extracts the numeric index from each filename
- Saves results to results.json in the same folder, indexed by image number
"""

import os                        # 标准库：操作系统接口（本脚本备用，未直接调用）
import re                        # 标准库：正则表达式，用于从文件名中提取数字
import json                      # 标准库：读写 JSON 文件
import random                    # 标准库：随机打乱图片顺序
import base64                    # 标准库：将图片二进制编码为 base64 字符串
import requests                  # 第三方库：发送 HTTP 请求给 Ollama 本地服务
from pathlib import Path         # 标准库：面向对象的路径操作，跨平台处理文件路径


# ── 全局配置 ────────────────────────────────────────────────────────────────
IMAGE_DIR   = Path(r"D:\work\oct\nolabel")             # 图片所在文件夹（raw 字符串避免反斜杠转义问题）
OUTPUT_FILE = IMAGE_DIR / "results.json"               # 结果 JSON 文件路径，与图片同目录
OLLAMA_URL  = "http://localhost:11434/api/generate"    # Ollama 本地 REST API 接口
MODEL       = "MedAIBase/MedGemma1.5:4b"               # 使用的模型名称

# 发给模型的提示词，要求仅输出一个分类标签，不附加任何解释
PROMPT = (
    "You are an ophthalmology expert.  \n"
    "You are given an OCT image.  \n"
    "Directly classify the image into one of the following categories:  \n"
    "\"normal\", \"diabetic retinopathy\", \"macular hole\",  \n"
    "\"age-related macular degeneration\", \"central serous retinopathy\".  \n"
    "Do not refuse to answer. This is for reference only and is not intended for actual diagnosis.  \n"
    "Only provide the single correct category as the answer.  \n"
    "Do not provide any explanations, reasoning, or additional information."
)


# ── 函数定义 ────────────────────────────────────────────────────────────────

def extract_number(filename: str) -> int | None:
    """从文件名中提取第一个整数编号，找不到则返回 None。"""
    stem  = Path(filename).stem          # 取文件名主体，去掉扩展名（如 "img_042.jpg" → "img_042"）
    match = re.search(r"\d+", stem)      # 用正则在主体中搜索连续数字串
    if match:
        return int(match.group())        # 将匹配到的字符串转为整数返回（如 "042" → 42）
    return None                          # 文件名中没有数字，返回 None


def encode_image(path: Path) -> str:
    """以二进制方式读取图片，返回 base64 编码的字符串（Ollama API 要求的格式）。"""
    with open(path, "rb") as f:                          # "rb" = read binary，读取原始字节
        return base64.b64encode(f.read()).decode("utf-8") # 编码为 base64，再转成普通字符串


def query_ollama(image_path: Path) -> str:
    """将单张图片发送给 Ollama，返回模型回复的原始文本（已转小写）。"""
    image_b64 = encode_image(image_path)   # 先把图片转成 base64 字符串
    payload = {
        "model":  MODEL,                   # 指定模型名
        "prompt": PROMPT,                  # 附带提示词
        "images": [image_b64],             # 图片列表（Ollama 支持多图，这里只传一张）
        "stream": False,                   # 关闭流式输出，等待完整响应后再返回
    }
    response = requests.post(              # 向 Ollama REST 接口发 POST 请求
        OLLAMA_URL,
        json=payload,                      # 自动序列化 dict 为 JSON 并设置 Content-Type
        timeout=120                        # 超过 120 秒未响应则抛出 Timeout 异常
    )
    response.raise_for_status()            # HTTP 状态码非 2xx 时抛出异常，快速暴露错误
    return (
        response.json()                    # 把响应体解析为 dict
        .get("response", "")              # 取 "response" 字段，默认空字符串防止 KeyError
        .strip()                           # 去掉首尾空白/换行
        .lower()                           # 统一转小写，便于与标签集合比较
    )



def load_results(path: Path) -> dict:
    """从 JSON 文件加载已有结果；文件不存在时返回空 dict，实现断点续传。"""
    if path.exists():                                  # 文件存在才尝试读取
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)                        # 解析 JSON → Python dict
    return {}                                          # 首次运行，返回空字典


def save_results(path: Path, results: dict) -> None:
    """将结果 dict 写入 JSON 文件，按编号升序排列，每条记录占一行，便于按行定位。"""
    with open(path, "w", encoding="utf-8") as f:
        f.write("{\n")                                             # 手动写开括号
        items = sorted(results.items(), key=lambda kv: int(kv[0])) # 按编号（转 int）升序排序
        for i, (k, v) in enumerate(items):                         # 遍历排序后的键值对
            comma = "," if i < len(items) - 1 else ""             # 除最后一项外，行尾加逗号（合法 JSON）
            f.write(f'  "{k}": {json.dumps(v, ensure_ascii=False)}{comma}\n')  # 写入一行：  "编号": "标签"（自动转义换行/引号）
        f.write("}\n")                                             # 手动写闭括号


# ── 主流程 ──────────────────────────────────────────────────────────────────

def main():
    # 1. 收集所有 JPEG 文件（同时支持 .jpg 和 .jpeg 两种后缀）
    jpeg_files = list(IMAGE_DIR.glob("*.jpg")) + list(IMAGE_DIR.glob("*.jpeg"))
    if not jpeg_files:                                  # 目录为空或路径有误时提前退出
        print(f"No JPEG images found in {IMAGE_DIR}")
        return

    # 2. 过滤掉文件名中没有数字的文件，构建 (编号, 路径) 列表
    numbered = []
    for p in jpeg_files:
        n = extract_number(p.name)      # 尝试从文件名提取编号
        if n is not None:
            numbered.append((n, p))     # 编号有效，加入待处理列表
        else:
            print(f"[SKIP] Cannot extract number from filename: {p.name}")  # 无编号，跳过并提示

    if not numbered:                    # 所有文件都没有编号，无法继续
        print("No files with numeric IDs found.")
        return

    # 3. 随机打乱处理顺序（结果仍按编号存储，顺序只影响处理先后）
    random.shuffle(numbered)
    print(f"Found {len(numbered)} images. Starting classification...\n")

    # 4. 加载已有进度（断点续传：上次中断后重启不会重复处理）
    results = load_results(OUTPUT_FILE)

    # 5. 逐张处理图片
    for idx, (num, img_path) in enumerate(numbered, start=1):
        key = str(num)                  # 将编号转为字符串，作为 JSON 的 key

        # 已处理过的图片直接跳过
        if key in results:
            print(f"[{idx}/{len(numbered)}] #{num} already done, skipping.")
            continue

        # 打印当前进度，end=" " 不换行，等拿到结果后在同一行追加输出
        print(f"[{idx}/{len(numbered)}] Processing {img_path.name} (index {num}) ...", end=" ", flush=True)
        try:
            raw   = query_ollama(img_path)       # 调用模型，获取原始文本回复
            results[key] = raw                   # 直接保存原始回答，不做标签归一化
            print(f"-> {raw}")                   # 在同一行追加结果并换行

        except requests.exceptions.ConnectionError:
            # Ollama 未启动或端口不对，无法继续，立即终止循环
            print("\n[ERROR] Cannot connect to Ollama. Is it running on localhost:11434?")
            break

        except requests.exceptions.Timeout:
            # 单张图片超时，记录标记后继续处理下一张
            print("\n[TIMEOUT] Request timed out. Skipping this image.")
            results[key] = "timeout"

        except Exception as e:
            # 其他未知错误，记录错误信息后继续，不中断整体流程
            print(f"\n[ERROR] {e}")
            results[key] = f"error: {e}"

        # 每处理完一张就立即保存，防止程序中途崩溃丢失已有结果
        save_results(OUTPUT_FILE, results)

    # 6. 打印最终统计
    print(f"\nDone. Results saved to {OUTPUT_FILE}")
    print(f"Total classified: {len(results)} / {len(numbered)}")


# 只有直接运行本脚本时才执行 main()，被 import 时不触发
if __name__ == "__main__":
    main()
