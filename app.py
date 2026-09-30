import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import streamlit as st
from transformers import pipeline

MODEL_NAME = "uer/roberta-base-finetuned-jd-binary-chinese"

LABEL_MAP = {
    "positive (stars 4 and 5)": "正面",
    "negative (stars 1, 2 and 3)": "负面",
}


@st.cache_resource(show_spinner=False)
def load_classifier():
    return pipeline("sentiment-analysis", model=MODEL_NAME)


st.set_page_config(
    page_title="AI 情感分析",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 AI 情感分析小项目")
st.caption("输入一句中文，模型判断它是正面还是负面情绪。")

text = st.text_area(
    "请输入一句话：",
    value="这个电影太好看了，我很喜欢！",
    height=120,
)

if st.button("开始分析", type="primary"):
    if not text.strip():
        st.warning("请先输入内容。")
    else:
        with st.spinner("模型加载和推理中，第一次会慢一点..."):
            classifier = load_classifier()
            result = classifier(text)[0]

        raw_label = result["label"]
        score = float(result["score"])
        label_cn = LABEL_MAP.get(raw_label, raw_label)

        st.write("原始输出：", result)

        if label_cn == "正面":
            st.success(f"正面情绪，置信度：{score:.2%}")
        elif label_cn == "负面":
            st.error(f"负面情绪，置信度：{score:.2%}")
        else:
            st.info(f"预测标签：{label_cn}，置信度：{score:.2%}")