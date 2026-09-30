import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from transformers import pipeline

MODEL_NAME = "uer/roberta-base-finetuned-jd-binary-chinese"
classifier = pipeline("sentiment-analysis", model=MODEL_NAME)

texts = [
    "这个电影太好看了，我很喜欢！",
    "服务态度非常差，再也不来了。",
    "味道不错，下次还来。",
    "太难吃了，差评。",
    "还行吧，一般般。",
    "质量很好，物流也快，五星好评。",
]

for t in texts:
    r = classifier(t)[0]
    print(f"{t}  ->  {r['label']}  ({r['score']:.4f})")