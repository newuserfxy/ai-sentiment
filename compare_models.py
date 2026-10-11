import pandas as pd
import jieba

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
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
print("数据总数:", len(df))

def build_model(clf):
    """用同样的 TF-IDF 参数，只换分类器"""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            tokenizer=jieba_tokenizer,
            token_pattern=None,
            max_features=1000,
            ngram_range=(1, 1),
            min_df=1,
        )),
        ("clf", clf),
    ])

models = {
    "朴素贝叶斯": MultinomialNB(alpha=1.0),
    "逻辑回归": LogisticRegression(max_iter=1000, C=1.0),
    "线性SVM": LinearSVC(C=1.0, max_iter=2000),
}

results = []
for name, clf in models.items():
    print(f"\n正在评估 {name} ...")
    model = build_model(clf)
    scores = cross_val_score(model, df["text"], df["label"], cv=5, scoring="accuracy")
    results.append({
        "模型": name,
        "平均": f"{scores.mean():.4f}",
        "标准差": f"{scores.std():.4f}",
        "各折": [f"{s:.2f}" for s in scores],
    })

print("\n" + "=" * 60)
print("对比结果")
print("=" * 60)
df_result = pd.DataFrame(results)
print(df_result.to_string(index=False))