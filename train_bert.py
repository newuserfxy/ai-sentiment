import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import pandas as pd
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset

# ============ 配置 ============
MODEL_NAME = "hfl/chinese-roberta-wwm-ext"
OUTPUT_DIR = "models/bert_sentiment"
MAX_LEN = 64           # 句子短，64 够用
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5   # BERT 微调常用学习率

# ============ 读数据 ============
df = pd.read_csv("data/reviews.csv")
print("数据总数:", len(df))
print("标签分布:")
print(df["label"].value_counts())

# ============ 拆分 ============
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"],
)

print(f"\n训练集: {len(train_df)} 条")
print(f"测试集: {len(test_df)} 条")

# ============ 加载 tokenizer ============
print(f"\n加载 tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
    )

# ============ 构建数据集 ============
train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
test_ds = Dataset.from_pandas(test_df.reset_index(drop=True))

train_ds = train_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)

# 重命名 label 列，Trainer 要求叫 "labels"
train_ds = train_ds.rename_column("label", "labels")
test_ds = test_ds.rename_column("label", "labels")

# 只保留需要的列
train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# ============ 加载模型 ============
print(f"\n加载模型: {MODEL_NAME}")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
)

# ============ 评估指标 ============
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = (preds == labels).mean()
    return {"accuracy": acc}

# ============ 训练参数 ============
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    logging_steps=20,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none",
)

# ============ Trainer ============
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics,
)

# ============ 训练 ============
print("\n开始训练...")
trainer.train()

# ============ 评估 ============
print("\n=== 最终评估 ===")
results = trainer.evaluate()
print(results)

# ============ 保存 ============
print(f"\n保存模型到 {OUTPUT_DIR}")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("完成")