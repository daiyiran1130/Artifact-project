"""
OCT image classification using MedGemma1.5-4b-it loaded directly via modelscope/transformers.
No Ollama required. The model runs in the current Python process on local GPU/CPU.
Prerequisites:
    pip install modelscope transformers accelerate torch pillow requests
"""
import os
os.environ["MODELSCOPE_CACHE"] = "/root/autodl-tmp/modelscope_cache"
os.environ["HF_HOME"] = "/root/autodl-tmp/hf_cache"
import re
os.environ["TRANSFORMERS_CACHE"] = "/root/autodl-tmp/hf_cache"
import json                        # 读写 JSON 结果文件
import random                      # 随机打乱图片处理顺序
from pathlib import Path           # 跨平台路径操作
import torch                       # PyTorch，提供张量运算和推理模式
from PIL import Image              # Pillow，直接读取图片为 PIL Image 对象（模型可直接接受）
#from modelscope import (           # modelscope 替代 huggingface_hub 在国内下载模型
#    AutoProcessor,                 # 负责文本+图片的预处理（tokenize、resize 等）
#    AutoModelForImageTextToText,   # 图文多模态生成模型的通用加载类
#)
from modelscope import AutoModelForImageTextToText
from transformers import AutoProcessor
# ── 全局配置 ────────────────────────────────────────────────────────────────
IMAGE_DIR   = Path("/root/autodl-tmp/artifact/weakintensity/nolabel")
OUTPUT_FILE = IMAGE_DIR / "results.json"     # 结果输出路径，与图片同目录
MODEL_ID    = "google/medgemma-1.5-4b-it"   # modelscope 上的模型标识符
# 分类提示词：要求模型只输出单个类别标签
PROMPT = (
    "You are an ophthalmology expert.  \n"
    "You are given an OCT image.  \n"
    "Directly classify the image into one of the following categories:  \n"
    "\"normal\", \"diabetic retinopathy\", \"macular hole\",  \n"
    "\"age-related macular degeneration\", \"central serous retinopathy\".  \n"
    "Do not refuse to answer. This is for reference only and is not intended for actual diagnosis.  \n"
#    "Only provide the single correct category as the answer.  \n"
#    "Do not provide any explanations, reasoning, or additional information."
#    "Describe your reasoning in steps."
)
# ── 模型加载（脚本启动时执行一次，之后复用）────────────────────────────────
print(f"Loading model: {MODEL_ID}  (this may take a while on first run)...")
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,   # 用 bfloat16 半精度加载，显存占用约减半，精度损失极小
    device_map="auto",            # 自动分配：有 GPU 则用 GPU，否则用 CPU
)
processor = AutoProcessor.from_pretrained(MODEL_ID)  # 加载与模型配套的预处理器
print("Model loaded.\n")
# ── 工具函数 ────────────────────────────────────────────────────────────────
def extract_number(filename: str) -> int | None:
    """从文件名主体中提取第一个整数编号，找不到则返回 None。"""
    stem  = Path(filename).stem          # 去掉扩展名（"img_042.jpg" → "img_042"）
    match = re.search(r"\d+", stem)      # 搜索连续数字串
    if match:
        return int(match.group())        # 转为整数返回
    return None
def query_model(image_path: Path) -> str:
    """
    用已加载的模型对单张图片做推理，返回模型输出的原始文本（已转小写）。
    与 Ollama 版本的区别：
      - 直接用 PIL.Image.open() 读图，无需 base64 编码
      - 通过 processor.apply_chat_template() 构造模型输入张量
      - 在同一 Python 进程内完成推理，无需 HTTP 请求
    """
    # 1. 用 PIL 打开图片（RGB 模式确保三通道，兼容灰度图）
    image = Image.open(image_path).convert("RGB")
    # 2. 构造多模态消息列表，格式与教材一致
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},   # 直接传 PIL Image 对象
                {"type": "text",  "text": PROMPT},
            ],
        }
    ]
    # 3. 用 processor 将消息（文本+图片）转换为模型所需的张量
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,   # 在末尾追加"Assistant:"等生成提示标记
        tokenize=True,                # 同时完成 tokenize
        return_dict=True,             # 返回字典形式（包含 input_ids、attention_mask 等）
        return_tensors="pt",          # 返回 PyTorch 张量
    ).to(model.device, dtype=torch.bfloat16)  # 移至模型所在设备，并统一精度
    input_len = inputs["input_ids"].shape[-1]  # 记录输入 token 数，用于截取新生成的部分
    # 4. 关闭梯度计算（推理阶段不需要反向传播，节省显存和时间）
    with torch.inference_mode():
        generation = model.generate(
            **inputs,
            max_new_tokens=50,    # OCT 分类只需短输出，限制 token 数加速推理
            do_sample=False,      # 关闭随机采样，使用贪心解码，输出更确定
        )
    # 5. 截取新生成的 token（去掉输入部分），解码为字符串
    new_tokens = generation[0][input_len:]
    decoded    = processor.decode(new_tokens, skip_special_tokens=True)
    return decoded.strip().lower()   # 去首尾空白并转小写，便于与标签集合比较
def save_results(path: Path, results: dict) -> None:
    """按编号升序将结果写入 JSON，每条记录占一行，便于按行定位。"""
    with open(path, "w", encoding="utf-8") as f:
        f.write("{\n")
        items = sorted(results.items(), key=lambda kv: int(kv[0]))  # 按编号（int）升序
        for i, (k, v) in enumerate(items):
            comma = "," if i < len(items) - 1 else ""               # 最后一项不加逗号
            f.write(f'  "{k}": "{v}"{comma}\n')
        f.write("}\n")
# ── 主流程 ──────────────────────────────────────────────────────────────────
def main():
    # 1. 收集所有 JPEG 文件
    jpeg_files = list(IMAGE_DIR.glob("*.jpg")) + list(IMAGE_DIR.glob("*.jpeg"))
    if not jpeg_files:
        print(f"No JPEG images found in {IMAGE_DIR}")
        return
    # 2. 过滤无编号文件，构建 (编号, 路径) 列表
    numbered = []
    for p in jpeg_files:
        n = extract_number(p.name)
        if n is not None:
            numbered.append((n, p))
        else:
            print(f"[SKIP] Cannot extract number from filename: {p.name}")
    if not numbered:
        print("No files with numeric IDs found.")
        return
    # 3. 随机打乱处理顺序（结果仍按编号存储）
    random.shuffle(numbered)
    print(f"Found {len(numbered)} images. Starting classification...\n")
    # 4. 清空 JSON，从头开始（不再断点续跑）
    results = {}
    save_results(OUTPUT_FILE, results)
    # 5. 逐张推理
    for idx, (num, img_path) in enumerate(numbered, start=1):
        key = str(num)
        print(f"[{idx}/{len(numbered)}] Processing {img_path.name} (index {num}) ...", end=" ", flush=True)
        try:
            raw   = query_model(img_path)        # 调用本地模型推理
            results[key] = raw
            print(f"-> {raw}")
        except Exception as e:
            print(f"\n[ERROR] {e}")
            results[key] = f"error: {e}"
        # 每张处理完立即保存，防止崩溃丢失进度
        save_results(OUTPUT_FILE, results)
    print(f"\nDone. Results saved to {OUTPUT_FILE}")
    print(f"Total classified: {len(results)} / {len(numbered)}")
if __name__ == "__main__":
    main()
