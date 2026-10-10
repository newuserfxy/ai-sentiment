import pandas as pd
import jieba

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score

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

df = pd.read_csv("data/reviews.csv")

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

# 5 折交叉验证
scores = cross_val_score(
    model,
    df["text"],
    df["label"],
    cv=5,
    scoring="accuracy",
)

print("每折准确率:", scores)
print("平均准确率:", scores.mean())
print("标准差:", scores.std())