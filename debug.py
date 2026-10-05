import joblib
import numpy as np
import jieba

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

model = joblib.load("models/sentiment_sklearn.joblib")

tfidf = model.named_steps["tfidf"]
clf = model.named_steps["clf"]

feature_names = tfidf.get_feature_names_out()

# 朴素贝叶斯：看类别的对数概率差
feature_log_prob = clf.feature_log_prob_
diff = feature_log_prob[1] - feature_log_prob[0]

top_pos = np.argsort(diff)[-20:][::-1]
print("=== 最正面的词 ===")
for i in top_pos:
    print(f"{feature_names[i]}  {diff[i]:.4f}")

top_neg = np.argsort(diff)[:20]
print("\n=== 最负面的词 ===")
for i in top_neg:
    print(f"{feature_names[i]}  {diff[i]:.4f}")