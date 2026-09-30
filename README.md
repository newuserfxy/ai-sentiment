# AI 中文情感分析

## 功能
输入一句中文，模型判断它是正面还是负面情绪，并给出置信度。

## 技术栈
- Python
- Hugging Face Transformers
- Streamlit

## 模型
uer/roberta-base-finetuned-jd-binary-chinese

## 运行方式
pip install -r requirements.txt
streamlit run app.py

## 注意事项
- 终端必须激活虚拟环境（.venv），看到 (.venv) 再运行命令。
- sentencepiece 必须为 0.1.99，否则 Windows 会崩溃。