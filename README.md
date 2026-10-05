# 信贷风险智能分析 Copilot｜RAG + Tool-Calling Agent

基于公开信贷数据构建的 LLM Agent 分析助手。输入「某类用户坏账率突然上升」这类业务问题，Agent 会自动拆解成 SQL/Python 取数、用户分层、模型调用和 SHAP 风险归因等可执行的分析步骤，最后输出一份带数据依据的分析报告。

```
用户问题："信息流广告渠道最近三个月坏账率突然上升，原因是什么？"
        │
   ┌────▼──────┐   检索：指标口径 / 数据字典 / 归因 SOP / 历史案例
   │ LLM Agent │◄──── RAG（bge-small-zh 向量 + BM25 混合检索）
   │           │
   └────┬──────┘
        │  Tool Calling，多轮编排
   ┌────▼───────────────────────────────────────────────┐
   │ run_sql / run_python      只读取数、隔离子进程执行代码    │
   │ segment_compare           分层对比 + 结构/风险效应拆解    │
   │ model_score_summary       模型评分、风险等级、PSI         │
   │ shap_compare              群体 SHAP 归因                │
   │ model_performance         测试集 AUC / KS               │
   │ data_quality_check        数据质量检查                   │
   └────┬───────────────────────────────────────────────┘
        │  每个工具结果都经过校验层
   ┌────▼──────┐  数据质量：未定义编码、空值、异常值
   │ Validator │  SQL 结果：空结果、比率越界、小样本、加权口径
   └────┬──────┘  模型输出：评分范围、SHAP 可加性、训练集泄漏
        │
        ▼  最终报告 → 数字溯源校验（报告中的数字必须能在工具结果中找到）
```

## 主要特点

- **业务口径对齐**：把数据字典、指标口径、标准分层、数据质量规则和历史分析案例做成 RAG 知识库，让模型按业务口径分析，不自己假设
- **工具编排**：8 个分析工具，覆盖取数、分层、结构拆解、模型评分、SHAP 归因和数据质量检查。工具报错时，模型会根据报错信息自行修正参数并重试
- **结果可靠性**：三层校验。① 每个工具结果附带自动校验提示；② 坏账率差异做显著性检验，样本量不足时给出提示；③ 最终报告中的每个数字都要能追溯到工具结果，否则打回让模型修正
- **模型可切换**：自建 LLM 适配层，同一套 Agent 可以在 Claude（Anthropic API）和智谱 GLM（OpenAI 兼容接口）之间切换，默认使用智谱 GLM-4.7-Flash 免费档
- **安全**：数据库只读、禁止访问外部文件和网络，Python 代码在子进程中执行，并有导入白名单和超时限制

## 数据与模型

| 项目 | 说明 |
|---|---|
| 数据 | [UCI Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)，30,000 个样本，23 个原始特征，坏账率 22.1% |
| 模拟业务字段 | 原始数据没有时间和渠道维度。脚本按「客群画像 → 渠道/月份」的方式补充 `loan_month`、`channel`、`city_tier` 三个字段，**不修改任何真实标签**，坏账率的异动完全来自客群结构变化 |
| 注入的异动场景 | S1：信息流广告 2025-10~12 放量引入年轻、低额度客群，坏账率从 21.2% 升到 29.5%<br>S2：合作方导流 2025-06 混入近期已有逾期的客户，坏账率达到 35.1% |
| 风险模型 | XGBoost，27 个特征（23 个原始特征 + 4 个衍生特征），**测试集 AUC 0.782 / KS 0.430** |

## 评测

自建 50 道业务分析评测题，分 5 类：指标取数 12、用户分层 10、异动归因 12、模型调用 8、数据质量与口径陷阱 8。标准答案全部由 SQL 从数据库实时计算（`eval/build_eval.py`）。评分规则：数字和关键词题按规则自动判分，开放性结论由 LLM 按 rubric 评审。

对比三种方案：

| 方案 | 说明 |
|---|---|
| 基础 Prompt | 把表结构写进 Prompt，模型一次性生成分析代码，执行一次后写结论。没有知识库，没有多轮工具调用，报错不重试 |
| Agent（无 RAG） | 完整工具和校验层，但不提供知识库 |
| Agent（完整） | RAG + 工具编排 + 校验层 |

**可执行分析成功率** = 分析流程执行完成，且全部检查项通过的题目占比。

### 评测结果（智谱 glm-4-flash，免费档）

| 方案 | 可执行分析成功率 | 平均耗时 | P90 耗时 | 平均工具调用 |
|---|---|---|---|---|
| 基础 Prompt | 26% | 10 秒 | 14 秒 | 1 |
| Agent（无 RAG） | 50% | 17 秒 | 37 秒 | 3.2 |
| **Agent（完整）** | **52%** | **16 秒** | **32 秒** | **2.9** |

| 类别 | 基础 Prompt | Agent（无 RAG） | Agent（完整） |
|---|---|---|---|
| A 指标取数 | 6/12 | 6/12 | 8/12 |
| B 用户分层 | 4/10 | 5/10 | 4/10 |
| C 异动归因 | 0/12 | 5/12 | 2/12 |
| D 模型调用 | 3/8 | 7/8 | 7/8 |
| E 数据质量/口径 | 0/8 | 2/8 | 5/8 |

**解读**
- 工具编排和校验层贡献了主要提升（26% → 50%）。基础方案有 12 题代码直接报错，Agent 50 题全部执行完成。
- RAG 的整体增益很小（+1 题，50 题规模下不显著）：在口径和数据质量类题目上提升明显（2/8 → 5/8），但在归因类题目上反而下降。查看轨迹发现，小模型检索到历史案例后，容易照搬案例的报告结构，对照分析做得不完整。
- 局限：评测用的是免费小模型；LLM 评审存在误判（例如 C04 因 0.1pp 的四舍五入差异被判失败）。评分规则没有在看到结果后修改。

每题明细见 [eval/results/](eval/results/)，汇总见 [summary.md](eval/results/summary.md)。

## 快速开始

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. 下载数据
mkdir -p data/raw && curl -L -o data/raw/uci.zip \
  "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip" \
  && unzip -o data/raw/uci.zip -d data/raw

# 2. 构建数据库、训练模型、建立知识库索引
python src/data_prep.py
python src/model/train.py
PYTHONPATH=src python src/rag/retriever.py

# 3. 配置 API Key（二选一）
cp .env.example .env   # 填入 ZHIPU_API_KEY（智谱，免费）或 ANTHROPIC_API_KEY

# 4. 运行
streamlit run src/app.py                                   # 演示界面
PYTHONPATH=src python src/agent/agent.py "合作方导流6月坏账率为什么这么高？"   # 命令行

# 5. 评测
python eval/build_eval.py
python eval/run_eval.py --system baseline
python eval/run_eval.py --system agent_norag
python eval/run_eval.py --system agent
python eval/report.py
```

## 目录结构

```
├── knowledge_base/          RAG 语料：数据字典、指标口径、归因 SOP、数据质量规则、模型说明、历史案例
├── src/
│   ├── data_prep.py         数据准备与业务场景模拟
│   ├── model/               XGBoost 训练、特征工程、SHAP
│   ├── rag/retriever.py     切块、向量化、混合检索
│   ├── tools/               分析工具、校验层、只读数据库、Python 执行器
│   ├── agent/               Agent 主循环、基础 Prompt 对照组、数字溯源校验、Prompt
│   └── app.py               Streamlit 演示界面
└── eval/                    评测集生成、评测运行、评分、结果汇总
```

## 技术栈

Python · LLM Tool Calling（智谱 GLM / Claude）· DuckDB · XGBoost · SHAP · fastembed（bge-small-zh）· BM25 · Streamlit
