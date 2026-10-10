import pandas as pd
import joblib
import os
import jieba

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

# 停用词
STOPWORDS = set([
    "的", "了", "是", "在", "有", "和", "就", "都", "而", "及", "与",
    "着", "或", "一个", "没有", "我们", "你们", "他们",
    "很", "太", "非常", "特别", "十分", "更", "最",
    "，", "。", "！", "？", "、", "；", "：", "“", "”", "（", "）",
    " ", "\n", "\t",
])

def jieba_tokenizer(text):
    words = jieba.lcut(text)
    return [w for w in words if w.strip() and w not in STOPWORDS]

# 读数据
df = pd.read_csv("data/reviews.csv")
print("数据总数:", len(df))
print("标签分布:")
print(df["label"].value_counts())

# 拆分
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"],
    test_size=0.2, random_state=42, stratify=df["label"],
)

# 模型
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        tokenizer=jieba_tokenizer,
        token_pattern=None,
        max_features=1000,
        ngram_range=(1, 1),
        min_df=1,
    )),
    ("clf", MultinomialNB(alpha=1.0)),
])

print("\n开始训练...")
model.fit(X_train, y_train)
print("训练完成")

# 测试集评估
pred = model.predict(X_test)
print("\n=== 测试集评估 ===")
print(classification_report(y_test, pred, target_names=["负面", "正面"], zero_division=0))
print("=== 混淆矩阵 ===")
print(confusion_matrix(y_test, pred))

# 训练集评估
train_pred = model.predict(X_train)
print("\n=== 训练集评估 ===")
print(classification_report(y_train, train_pred, target_names=["负面", "正面"], zero_division=0))

# 交叉验证
print("\n=== 交叉验证（5 折） ===")
cv_scores = cross_val_score(model, df["text"], df["label"], cv=5, scoring="accuracy")
print("各折:", [f"{s:.4f}" for s in cv_scores])
print("平均:", f"{cv_scores.mean():.4f}")
print("标准差:", f"{cv_scores.std():.4f}")

# 保存
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/sentiment_sklearn.joblib")
print("\n模型已保存")