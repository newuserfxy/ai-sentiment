import pandas as pd
import jieba

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV

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
    )),
    ("clf", MultinomialNB()),
])

# 参数网格
param_grid = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__min_df": [1, 2],
    "tfidf__max_features": [1000, 3000, 5000],
    "clf__alpha": [0.1, 0.5, 1.0],
}

# 网格搜索 + 5 折交叉验证
grid = GridSearchCV(
    model,
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1,
)

print("开始搜索...")
grid.fit(df["text"], df["label"])

print("\n=== 最佳参数 ===")
print(grid.best_params_)
print("\n=== 最佳交叉验证分数 ===")
print(grid.best_score_)