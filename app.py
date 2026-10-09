import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import streamlit as st
import joblib
import jieba
import pandas as pd
import datetime

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

def analyze_word_weights(text, model):
    tfidf = model.named_steps["tfidf"]
    clf = model.named_steps["clf"]

    if hasattr(clf, "coef_"):
        weights = clf.coef_[0]
    else:
        weights = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]

    feature_names = tfidf.get_feature_names_out()
    vocab = {name: i for i, name in enumerate(feature_names)}

    words = jieba.lcut(text)
    result = []
    for w in words:
        if not w.strip() or w in STOPWORDS:
            continue
        if w in vocab:
            idx = vocab[w]
            result.append((w, float(weights[idx]), True))
        else:
            # 整词不在词表，拆成单字逐个查
            chars = list(w)
            if len(chars) > 1:
                for c in chars:
                    if c in vocab:
                        idx = vocab[c]
                        result.append((c, float(weights[idx]), True))
                    else:
                        result.append((c, 0.0, False))
            else:
                result.append((w, 0.0, False))
    return result

def render_weighted_text(words_with_weights):
    html_parts = []
    for item in words_with_weights:
        word, weight = item[0], item[1]
        in_vocab = item[2] if len(item) > 2 else True

        if not in_vocab:
            color = "#888888"
            bg = "transparent"
            border = "1px dashed #666"
        elif weight > 0.3:
            color = "#0a7d2e"
            bg = "#d4f8d4"
            border = "none"
        elif weight > 0.05:
            color = "#3a9e5c"
            bg = "#eafcef"
            border = "none"
        elif weight < -0.3:
            color = "#a01020"
            bg = "#ffd6db"
            border = "none"
        elif weight < -0.05:
            color = "#c44050"
            bg = "#ffe9ec"
            border = "none"
        else:
            color = "#888888"
            bg = "transparent"
            border = "none"

        html_parts.append(
            f'<span style="color:{color};background:{bg};'
            f'border-bottom:{border};'
            f'padding:2px 4px;margin:2px;border-radius:4px;'
            f'display:inline-block;">{word}</span>'
        )
    return " ".join(html_parts)

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
    options=["预训练模型（V1）", "自己训练的模型（V2）", "🔍 双模型对比"],
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
        # ============ 双模型对比模式 ============
        if model_choice == "🔍 双模型对比":
            with st.spinner("V1 推理中..."):
                classifier_v1 = load_hf_classifier()
                result_v1 = classifier_v1(text)[0]
                label_v1 = HF_LABEL_MAP.get(result_v1["label"], result_v1["label"])
                score_v1 = float(result_v1["score"])

            with st.spinner("V2 推理中..."):
                model_v2 = load_sklearn_model()
                pred_v2 = model_v2.predict([text])[0]
                proba_v2 = model_v2.predict_proba([text])[0]
                label_v2 = "正面" if pred_v2 == 1 else "负面"
                score_v2 = float(proba_v2[pred_v2])

            st.write("原始输出：")
            col_raw1, col_raw2 = st.columns(2)
            with col_raw1:
                st.caption("V1 原始输出")
                st.json(result_v1)
            with col_raw2:
                st.caption("V2 原始输出")
                st.json({
                    "pred": int(pred_v2),
                    "prob_负面": float(proba_v2[0]),
                    "prob_正面": float(proba_v2[1]),
                })

            st.divider()

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🤖 预训练模型（V1）")
                if label_v1 == "正面":
                    st.success(f"正面情绪，置信度：{score_v1:.2%}")
                else:
                    st.error(f"负面情绪，置信度：{score_v1:.2%}")
                st.metric("预测标签", label_v1)
                st.metric("置信度", f"{score_v1:.2%}")
                st.progress(min(max(score_v1, 0.0), 1.0))

            with col2:
                st.subheader("🧪 自训练模型（V2）")
                if label_v2 == "正面":
                    st.success(f"正面情绪，置信度：{score_v2:.2%}")
                else:
                    st.error(f"负面情绪，置信度：{score_v2:.2%}")
                st.metric("预测标签", label_v2)
                st.metric("置信度", f"{score_v2:.2%}")
                st.progress(min(max(score_v2, 0.0), 1.0))

                st.markdown("**关键词贡献：**")
                words_w = analyze_word_weights(text, model_v2)
                st.markdown(render_weighted_text(words_w), unsafe_allow_html=True)

                with st.expander("查看每个词的权重"):
                    weights_data = [
                        {"词": item[0], "权重": f"{item[1]:.4f}", "在词表": item[2]}
                        for item in words_w
                    ]
                    st.dataframe(pd.DataFrame(weights_data), use_container_width=True)

            if label_v1 == label_v2:
                st.info(f"✅ 两个模型判断一致：{label_v1}")
            else:
                st.warning(f"⚠️ 两个模型判断不一致：V1 判为 {label_v1}，V2 判为 {label_v2}")

            # 记录到历史（作为两条）
            now = datetime.datetime.now().strftime("%H:%M:%S")
            st.session_state.history.append({
                "时间": now, "模型": "V1（对比）", "输入": text,
                "结果": label_v1, "置信度": f"{score_v1:.2%}",
            })
            st.session_state.history.append({
                "时间": now, "模型": "V2（对比）", "输入": text,
                "结果": label_v2, "置信度": f"{score_v2:.2%}",
            })

        # ============ 单模型模式 ============
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
                    label_cn = "正面" if pred == 1 else "负面"
                    score = float(proba[pred])
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

            col1, col2 = st.columns(2)
            with col1:
                st.metric("预测标签", label_cn)
            with col2:
                st.metric("置信度", f"{score:.2%}")

            st.write("置信度可视化：")
            st.progress(min(max(score, 0.0), 1.0))

            # 关键词高亮（仅 V2）
            if model_choice == "自己训练的模型（V2）":
                st.divider()
                st.subheader("🔍 关键词贡献分析")
                st.caption("绿色 = 支持正面，红色 = 支持负面，颜色越深影响越大。")

                words_with_weights = analyze_word_weights(text, model)
                html = render_weighted_text(words_with_weights)
                st.markdown(html, unsafe_allow_html=True)

                # 显示详细权重表
                with st.expander("查看每个词的权重"):
                    weights_data = [
                        {"词": item[0], "权重": f"{item[1]:.4f}", "在词表": item[2]}
                        for item in words_with_weights
                    ]
                    st.dataframe(pd.DataFrame(weights_data), use_container_width=True)

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

                # ============ 统计面板 ============
                st.divider()
                st.subheader("📊 结果统计")

                counts = df_batch["预测结果"].value_counts()
                total = len(df_batch)
                pos_count = int(counts.get("正面", 0))
                neg_count = int(counts.get("负面", 0))

                # 指标卡片
                if total == 0:
                    st.warning("数据为空，无法统计。")
                else:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("总数", total)
                    with col2:
                        st.metric("正面", f"{pos_count} ({pos_count / total:.1%})")
                    with col3:
                        st.metric("负面", f"{neg_count} ({neg_count / total:.1%})")

                # 柱状图
                st.bar_chart(counts)

                csv_bytes = df_batch.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="📥 下载结果 CSV",
                    data=csv_bytes,
                    file_name="batch_result.csv",
                    mime="text/csv",
                )