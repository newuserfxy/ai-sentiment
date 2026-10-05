import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import streamlit as st
import joblib
import jieba

from transformers import pipeline

# ============ 配置 ============
HF_MODEL_NAME = "uer/roberta-base-finetuned-jd-binary-chinese"
SKLEARN_MODEL_PATH = "models/sentiment_sklearn.joblib"

HF_LABEL_MAP = {
    "positive (stars 4 and 5)": "正面",
    "negative (stars 1, 2 and 3)": "负面",
}

# ============ sklearn 分词器（必须和 train.py 一致）============
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

# ============ 模型加载（带缓存）============
@st.cache_resource(show_spinner=False)
def load_hf_classifier():
    return pipeline("sentiment-analysis", model=HF_MODEL_NAME)

@st.cache_resource(show_spinner=False)
def load_sklearn_model():
    return joblib.load(SKLEARN_MODEL_PATH)

# ============ 页面配置 ============
st.set_page_config(
    page_title="AI 情感分析",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 AI 情感分析小项目")
st.caption("输入一句中文，模型判断它是正面还是负面情绪。")

# ============ 模型选择 ============
model_choice = st.radio(
    "选择模型：",
    options=["预训练模型（V1）", "自己训练的模型（V2）"],
    horizontal=True,
)

# ============ 输入 ============
text = st.text_area(
    "请输入一句话：",
    value="这个电影太好看了，我很喜欢！",
    height=120,
)

# ============ 推理 ============
if st.button("开始分析", type="primary"):
    if not text.strip():
        st.warning("请先输入内容。")
    else:
        with st.spinner("推理中..."):
            if model_choice == "预训练模型（V1）":
                classifier = load_hf_classifier()
                result = classifier(text)[0]
                raw_label = result["label"]
                score = float(result["score"])
                label_cn = HF_LABEL_MAP.get(raw_label, raw_label)
                raw_display = result
            else:
                model = load_sklearn_model()
                pred = model.predict([text])[0]
                proba = model.predict_proba([text])[0]
                # pred 是 0 或 1
                label_cn = "正面" if pred == 1 else "负面"
                score = float(proba[pred])
                raw_label = f"label={pred}"
                raw_display = {
                    "pred": int(pred),
                    "prob_负面": float(proba[0]),
                    "prob_正面": float(proba[1]),
                }

        st.write("原始输出：", raw_display)

        if label_cn == "正面":
            st.success(f"正面情绪，置信度：{score:.2%}")
        elif label_cn == "负面":
            st.error(f"负面情绪，置信度：{score:.2%}")
        else:
            st.info(f"预测标签：{label_cn}，置信度：{score:.2%}")