# 信贷风险智能分析 Copilot｜RAG + Tool-Calling Agent

> 🚧 开发中

基于公开信贷数据（UCI Default of Credit Card Clients，30,000 样本）构建的 LLM Agent 分析助手，
能将「某类用户坏账率突然上升」等业务问题自动拆解为 SQL/Python 取数、用户分层、模型调用与 SHAP 风险归因的可执行流程。

## 快速开始

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
mkdir -p data/raw && curl -L -o data/raw/uci.zip "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip" && unzip -o data/raw/uci.zip -d data/raw
python src/data_prep.py      # 构建 DuckDB，补充模拟业务字段
python src/model/train.py    # 训练 XGBoost，写入评分与 SHAP 值
```

## 当前进度

- [x] 数据准备与业务场景模拟
- [x] XGBoost 风险模型（Test AUC 0.782 / KS 0.430）
- [ ] RAG 知识库
- [ ] Tool-Calling Agent + 校验层
- [ ] 50 题评测集
- [ ] Streamlit 演示界面
