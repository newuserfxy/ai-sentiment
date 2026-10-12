import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from transformers import pipeline

MODEL_PATH = "models/bert_sentiment"

classifier = pipeline(
    "sentiment-analysis",
    model=MODEL_PATH,
    tokenizer=MODEL_PATH,
)

texts = [
    "这个电影太好看了",
    "服务态度非常差",
    "还行吧，一般般",
    "手机流畅，不卡顿",
    "快递太慢了，等了一周",
    "性价比很高，推荐购买",
    "完全不符合预期，很失望",
    "中规中矩，没什么亮点",
]

for t in texts:
    r = classifier(t)[0]
    label = "正面" if r["label"] == "LABEL_1" else "负面"
    print(f"{t}  ->  {label}  ({r['score']:.4f})")