"""
Streamlit 演示界面：streamlit run src/app.py
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ROOT  # noqa: E402
from tools.db import query  # noqa: E402

st.set_page_config(page_title="信贷风险智能分析 Copilot", page_icon="📊", layout="wide")

EXAMPLES = [
    "信息流广告渠道最近三个月（2025年10-12月）坏账率突然上升，帮我分析一下原因。",
    "合作方导流渠道6月份的坏账率很高，是什么原因？",
    "2025年有哪些渠道在哪些月份出现了坏账率的显著异动？",
    "按标准年龄段分层，各年龄段的坏账率是多少？哪个年龄段最高？",
    "学历编码为5的用户坏账率只有6.4%，是不是说明这类客群很优质？",
    "风险模型最重要的3个特征是什么？",
]

TOOL_ICON = {"search_knowledge": "📚", "run_sql": "🗄️", "run_python": "🐍", "segment_compare": "🧩",
             "shap_compare": "🔍", "model_score_summary": "📈", "model_performance": "🎯",
             "data_quality_check": "🧹"}


def page_agent():
    with st.sidebar:
        st.subheader("设置")
        use_rag = st.toggle("启用 RAG 知识库", value=True)
        st.caption(f"模型：{os.getenv('LLM_MODEL', 'claude-opus-5-5')}")
        st.subheader("示例问题")
        for i, q in enumerate(EXAMPLES):
            if st.button(q, key=f"ex{i}", use_container_width=True):
                st.session_state["question"] = q

    question = st.text_area("输入业务问题", value=st.session_state.get("question", EXAMPLES[0]), height=80)
    if not st.button("开始分析", type="primary"):
        return
    if not os.getenv("ANTHROPIC_API_KEY"):
        st.error("未配置 ANTHROPIC_API_KEY，请在项目根目录的 .env 文件中设置。")
        return

    from agent.agent import CreditRiskAgent, Step

    status = st.status("Agent 分析中…", expanded=True)

    def on_step(step: Step):
        with status:
            if step.kind == "thinking":
                st.caption(f"💭 {step.content[:400]}")
            elif step.kind == "tool_call":
                icon = TOOL_ICON.get(step.tool, "🔧")
                with st.expander(f"{icon} 调用 `{step.tool}`", expanded=False):
                    args = dict(step.args or {})
                    code = args.pop("sql", None) or args.pop("code", None)
                    if code:
                        st.code(code, language="sql" if step.tool == "run_sql" else "python")
                    if args:
                        st.json(args)
            elif step.kind == "tool_result":
                if step.is_error:
                    st.error(step.content[:600])
                for w in step.warnings:
                    st.warning(w, icon="⚠️")
            elif step.kind == "check":
                (st.warning if step.warnings else st.success)(f"🧾 {step.content}")

    run = CreditRiskAgent(use_rag=use_rag, on_step=on_step).run(question)
    status.update(label=f"分析完成：{run.n_tool_calls} 次工具调用，耗时 {run.elapsed:.0f} 秒",
                  state="complete" if run.status == "ok" else "error", expanded=False)
    st.markdown(run.answer)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("耗时", f"{run.elapsed:.0f}s")
    c2.metric("工具调用", run.n_tool_calls)
    c3.metric("工具报错并自我修正", run.n_tool_errors)
    c4.metric("输出 tokens", f"{run.usage['output']:,}")


def page_data():
    st.subheader("坏账率：渠道 × 放款月份")
    df = query("SELECT channel, loan_month, AVG(is_bad) AS bad_rate, COUNT(*) AS n FROM loans GROUP BY ALL")
    pivot = df.pivot(index="channel", columns="loan_month", values="bad_rate")
    st.dataframe(pivot.style.format("{:.1%}").background_gradient(cmap="Reds", axis=None),
                 use_container_width=True)
    st.subheader("放款量：渠道 × 放款月份")
    st.dataframe(df.pivot(index="channel", columns="loan_month", values="n"), use_container_width=True)
    metrics = json.loads((ROOT / "artifacts" / "model_metrics.json").read_text())
    st.subheader("风险模型")
    m1, m2, m3 = st.columns(3)
    m1.metric("测试集 AUC", metrics["test_auc"])
    m2.metric("测试集 KS", metrics["test_ks"])
    m3.metric("特征数", metrics["n_features"])


def page_eval():
    summary = ROOT / "eval" / "results" / "summary.md"
    if summary.exists():
        st.markdown(summary.read_text(encoding="utf-8"))
    else:
        st.info("还没有评测结果。运行 `python eval/run_eval.py --system agent` 后再查看。")
    qs = ROOT / "eval" / "questions.jsonl"
    if qs.exists():
        st.subheader("评测题目")
        rows = [json.loads(l) for l in qs.read_text(encoding="utf-8").splitlines() if l]
        st.dataframe(pd.DataFrame(rows)[["id", "question"]], use_container_width=True, hide_index=True)


st.title("📊 信贷风险智能分析 Copilot")
st.caption("RAG + Tool-Calling Agent｜把业务问题自动拆解为取数、分层、模型调用与 SHAP 归因的可执行分析流程")
tab1, tab2, tab3 = st.tabs(["🤖 智能分析", "🗂️ 数据概览", "🧪 评测结果"])
with tab1:
    page_agent()
with tab2:
    page_data()
with tab3:
    page_eval()
