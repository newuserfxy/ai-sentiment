import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import streamlit as st
import joblib
import jieba
import pandas as pd

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

# 初始化历史记录
if "history" not in st.session_state:
    st.session_state.history = []

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

        # 指标卡片：两列布局
        col1, col2 = st.columns(2)
        with col1:
            st.metric("预测标签", label_cn)
        with col2:
            st.metric("置信度", f"{score:.2%}")

        # 置信度进度条
        st.write("置信度可视化：")
        st.progress(min(max(score, 0.0), 1.0))

        # 记录到历史
        import datetime
        st.session_state.history.append({
            "时间": datetime.datetime.now().strftime("%H:%M:%S"),
            "模型": model_choice,
            "输入": text,
            "结果": label_cn,
            "置信度": f"{score:.2%}",
        })

# ============ 历史记录展示 ============
st.divider()
st.subheader("📜 历史记录")
if st.session_state.history:
    df_history = pd.DataFrame(st.session_state.history)
    st.dataframe(df_history, use_container_width=True)
    if st.button("清空历史"):
        st.session_state.history = []
        st.rerun()
else:
    st.caption("还没有分析记录。")

# ============ 批量分析 ============
st.divider()

with st.expander("📦 批量分析（上传 CSV）"):
    st.caption("CSV 文件必须包含一列名为 `text` 的文本列。")

    uploaded_file = st.file_uploader("选择 CSV 文件", type=["csv"])

    if uploaded_file is not None:
        try:
            df_batch = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"读取 CSV 失败：{e}")
            st.stop()

        if "text" not in df_batch.columns:
            st.error("CSV 必须包含名为 text 的列。")
        else:
            st.write(f"共读取到 {len(df_batch)} 条数据：")
            st.dataframe(df_batch.head(5), use_container_width=True)

            if st.button("开始批量分析", type="primary"):
                texts = df_batch["text"].astype(str).tolist()

                with st.spinner(f"正在分析 {len(texts)} 条，请稍候..."):
                    if model_choice == "预训练模型（V1）":
                        classifier = load_hf_classifier()
                        results = classifier(texts)
                        labels = [HF_LABEL_MAP.get(r["label"], r["label"]) for r in results]
                        scores = [float(r["score"]) for r in results]
                    else:
                        model = load_sklearn_model()
                        preds = model.predict(texts)
                        probas = model.predict_proba(texts)
                        labels = ["正面" if p == 1 else "负面" for p in preds]
                        scores = [float(probas[i][preds[i]]) for i in range(len(preds))]

                df_batch["预测结果"] = labels
                df_batch["置信度"] = [f"{s:.2%}" for s in scores]

                st.success(f"分析完成，共 {len(df_batch)} 条。")
                st.dataframe(df_batch, use_container_width=True)

                csv_bytes = df_batch.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="📥 下载结果 CSV",
                    data=csv_bytes,
                    file_name="batch_result.csv",
                    mime="text/csv",
                )