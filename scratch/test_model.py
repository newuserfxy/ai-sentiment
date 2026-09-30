import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

print("开始导入 transformers...", flush=True)
from transformers import pipeline
print("导入完成", flush=True)

MODEL_NAME = "IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment"

print("开始加载模型，第一次会下载，请耐心等...", flush=True)
classifier = pipeline("sentiment-analysis", model=MODEL_NAME)
print("模型加载完成", flush=True)

text = "这个电影太好看了，我很喜欢！"
print("开始推理...", flush=True)
result = classifier(text)
print("推理完成", flush=True)
print(result)