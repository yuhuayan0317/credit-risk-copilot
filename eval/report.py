"""汇总评测结果，生成 eval/results/summary.md。"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "eval" / "results"

SYSTEMS = {"baseline": "基础 Prompt", "agent_norag": "Agent（无 RAG）", "agent": "Agent（完整）"}
CATEGORIES = {"A": "指标取数", "B": "用户分层", "C": "异动归因", "D": "模型调用", "E": "数据质量/口径"}

# 美元 / 百万 tokens：input, output, cache read, cache write；未列出的模型（如智谱 Flash 免费档）按 0 计
PRICES = {
    "claude-opus-5-5": {"input": 4.0, "output": 20.0, "cache_read": 0.20, "cache_write": 5.0},
    "claude-sonnet-5-5": {"input": 2.0, "output": 10.0, "cache_read": 0.20, "cache_write": 2.5},
    "claude-haiku-4-5": {"input": 1.0, "output": 5.0, "cache_read": 0.10, "cache_write": 1.25},
}


def _cost(row) -> float:
    price = PRICES.get(row.get("model", ""), {})
    return sum(row["usage"].get(k, 0) * p for k, p in price.items()) / 1e6


def load(system: str) -> pd.DataFrame | None:
    path = RESULTS / f"{system}.jsonl"
    if not path.exists():
        return None
    rows = [json.loads(l) for l in path.read_text().splitlines() if l]
    df = pd.DataFrame(rows).drop_duplicates("id", keep="last")
    df["cost"] = df.apply(_cost, axis=1)
    return df


def main():
    frames = {s: load(s) for s in SYSTEMS}
    frames = {s: d for s, d in frames.items() if d is not None}
    if not frames:
        print("没有评测结果")
        return

    rows = []
    for s, d in frames.items():
        rows.append({
            "方案": SYSTEMS[s], "模型": d["model"].iloc[0] if "model" in d else "", "题数": len(d),
            "可执行分析成功率": f"{d['success'].mean():.0%}",
            "流程执行完成率": f"{d['executed'].mean():.0%}",
            "平均耗时(秒)": f"{d['elapsed'].mean():.0f}",
            "P90耗时(秒)": f"{d['elapsed'].quantile(0.9):.0f}",
            "平均工具调用": f"{d['n_tool_calls'].mean():.1f}",
            "平均成本($)": f"{d['cost'].mean():.3f}",
        })
    overall = pd.DataFrame(rows)

    cat_rows = []
    for c, name in CATEGORIES.items():
        row = {"类别": f"{c} {name}"}
        for s, d in frames.items():
            sub = d[d["category"] == c]
            row[SYSTEMS[s]] = f"{sub['success'].sum()}/{len(sub)}" if len(sub) else "-"
        cat_rows.append(row)
    by_cat = pd.DataFrame(cat_rows)

    md = ["# 评测结果\n", overall.to_markdown(index=False), "\n\n## 分类别成功题数\n", by_cat.to_markdown(index=False)]
    if "agent" in frames:
        fails = frames["agent"][~frames["agent"]["success"]]
        if len(fails):
            md.append("\n\n## 完整 Agent 未通过的题目\n")
            for _, r in fails.iterrows():
                why = "; ".join(c["note"] for c in r["grade"]["checks"] if not c["passed"]) or r["status"]
                md.append(f"- **{r['id']}** {r['question']} —— {why[:200]}")
    text = "\n".join(md) + "\n"
    (RESULTS / "summary.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
