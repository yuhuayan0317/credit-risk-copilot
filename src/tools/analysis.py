"""
分析工具实现。每个函数返回 ToolResult：给 LLM 看的文本 + 校验 warnings + 可选的表格（给前端展示）。
"""
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve

from rag.retriever import get_retriever
from tools import validators as V
from tools.db import SEGMENT_DIMS, check_filter, query, query_view

MAX_ROWS = 60


@dataclass
class ToolResult:
    text: str
    warnings: list[str] = field(default_factory=list)
    table: pd.DataFrame | None = None
    is_error: bool = False

    def render(self) -> str:
        out = self.text
        if self.warnings:
            out += "\n\n[校验提示]\n" + "\n".join(f"- {w}" for w in self.warnings)
        else:
            out += "\n\n[校验提示] 无异常"
        return out


def _cell(x) -> str:
    if isinstance(x, (bool, np.bool_)):
        return str(x)
    if isinstance(x, (int, np.integer)):
        return str(x)
    if isinstance(x, (float, np.floating)):
        if np.isnan(x):
            return "NaN"
        if float(x).is_integer() and abs(x) < 1e12:
            return str(int(x))
        if abs(x) >= 100:
            return f"{x:.1f}"
        return f"{x:.4g}"
    return str(x)


def _fmt(df: pd.DataFrame) -> str:
    shown = df.head(MAX_ROWS).map(_cell)
    text = shown.to_markdown(index=False, disable_numparse=True)
    if len(df) > MAX_ROWS:
        text += f"\n（共 {len(df)} 行，仅展示前 {MAX_ROWS} 行，请用聚合或 LIMIT 缩小结果）"
    return text


def _two_prop_ztest(b1: int, n1: int, b2: int, n2: int) -> float:
    if min(n1, n2) == 0:
        return float("nan")
    p = (b1 + b2) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return float("nan")
    z = (b1 / n1 - b2 / n2) / se
    return float(2 * (1 - stats.norm.cdf(abs(z))))


# ---------------------------------------------------------------- 知识库

def search_knowledge(query_text: str, top_k: int = 4) -> ToolResult:
    hits = get_retriever().search(query_text, top_k=top_k)
    body = "\n\n---\n\n".join(f"[来源: {h['doc']} / {h['section']}]\n{h['text']}" for h in hits)
    return ToolResult(body)


# ---------------------------------------------------------------- SQL

def run_sql(sql: str) -> ToolResult:
    df = query(sql)
    return ToolResult(f"返回 {len(df)} 行：\n{_fmt(df)}", V.check_sql_result(sql, df), df)


# ---------------------------------------------------------------- 数据质量

def data_quality_check(filter_expr: str | None = None) -> ToolResult:
    f = check_filter(filter_expr)
    df = query_view(f"SELECT * FROM v WHERE {f}")
    n = len(df)
    rows = []
    nulls = df.isna().sum()
    rows.append(("空值", "无" if nulls.sum() == 0 else ", ".join(f"{k}:{v}" for k, v in nulls[nulls > 0].items())))
    edu_bad = int((~df["education"].isin([1, 2, 3, 4])).sum())
    mar_bad = int((~df["marriage"].isin([1, 2, 3])).sum())
    rows.append(("education 未定义编码(0/5/6)", f"{edu_bad} 人 ({edu_bad / max(n, 1):.1%})"))
    rows.append(("marriage 未定义编码(0)", f"{mar_bad} 人 ({mar_bad / max(n, 1):.1%})"))
    neg_bill = int((df["bill_amt_1"] < 0).sum())
    rows.append(("bill_amt_1 为负（溢缴款，非错误）", f"{neg_bill} 人"))
    age_bad = int(((df["age"] < 18) | (df["age"] > 100)).sum())
    rows.append(("年龄超出 18~100", f"{age_bad} 人"))
    lim_bad = int((df["credit_limit"] <= 0).sum())
    rows.append(("授信额度 ≤ 0", f"{lim_bad} 人"))
    pay_cols = [f"pay_status_{i}" for i in range(1, 7)]
    ps_bad = int(((df[pay_cols] < -2) | (df[pay_cols] > 9)).any(axis=1).sum())
    rows.append(("还款状态超出 -2~9", f"{ps_bad} 人"))
    table = pd.DataFrame(rows, columns=["检查项", "结果"])
    warnings = V.check_sample(n, "该群体")
    if edu_bad or mar_bad:
        warnings.append("存在未定义编码：按学历/婚姻分析时需单独列为「未知」。")
    return ToolResult(f"过滤条件：{f}\n样本量：{n}\n{_fmt(table)}", warnings, table)


# ---------------------------------------------------------------- 分层对比 + 结构拆解

def segment_compare(dimension: str, target_filter: str, base_filter: str | None = None) -> ToolResult:
    """
    对比目标群体与对照群体在某个分层维度上的结构和坏账率，并做结构效应/风险效应拆解。
    base_filter 为空时，只输出目标群体的分层结果。
    """
    if dimension not in SEGMENT_DIMS:
        raise ValueError(f"不支持的维度 {dimension}，可选：{list(SEGMENT_DIMS)}")
    expr = SEGMENT_DIMS[dimension]

    def agg(flt: str) -> pd.DataFrame:
        return query_view(f"""SELECT {expr} AS segment, COUNT(*) AS n, SUM(is_bad) AS bad,
            AVG(is_bad) AS bad_rate, AVG(risk_score) AS avg_score
            FROM v WHERE {check_filter(flt)} GROUP BY 1 ORDER BY 1""").assign(
            segment=lambda d: d["segment"].str.replace(r"^\d_", "", regex=True))

    tgt = agg(target_filter)
    warnings = []
    if tgt.empty:
        return ToolResult("目标群体为空，请检查过滤条件。", ["目标群体样本量为 0"], is_error=True)
    tgt["share"] = tgt["n"] / tgt["n"].sum()
    warnings += V.check_sql_result(f"{dimension} n", tgt[["segment", "n"]])

    if not base_filter:
        out = tgt[["segment", "n", "share", "bad_rate", "avg_score"]]
        overall = tgt["bad"].sum() / tgt["n"].sum()
        text = (f"目标群体：{target_filter}（n={int(tgt['n'].sum())}，坏账率 {overall:.2%}）\n"
                f"按 {dimension} 分层：\n{_fmt(out)}")
        return ToolResult(text, warnings, out)

    base = agg(base_filter)
    base["share"] = base["n"] / base["n"].sum()
    m = base.merge(tgt, on="segment", how="outer", suffixes=("_base", "_target")).fillna(0)
    m["mix_effect"] = (m["share_target"] - m["share_base"]) * m["bad_rate_base"]
    m["rate_effect"] = m["share_target"] * (m["bad_rate_target"] - m["bad_rate_base"])

    nb, nt = int(base["n"].sum()), int(tgt["n"].sum())
    bb, bt = int(base["bad"].sum()), int(tgt["bad"].sum())
    br_b, br_t = bb / nb, bt / nt
    diff = br_t - br_b
    mix, rate = m["mix_effect"].sum(), m["rate_effect"].sum()
    p = _two_prop_ztest(bt, nt, bb, nb)

    warnings += V.check_sample(nt, "目标群体") + V.check_sample(nb, "对照群体")
    if abs((mix + rate) - diff) > 1e-6:
        warnings.append("结构拆解加总与总差异不一致，请检查。")

    cols = ["segment", "n_base", "share_base", "bad_rate_base", "n_target", "share_target",
            "bad_rate_target", "mix_effect", "rate_effect"]
    table = m[cols]
    mix_pct = mix / diff if diff else float("nan")
    text = (
        f"对照群体：{base_filter}（n={nb}，坏账率 {br_b:.2%}）\n"
        f"目标群体：{target_filter}（n={nt}，坏账率 {br_t:.2%}）\n"
        f"坏账率差异：{diff * 100:+.2f} pp，两比例 z 检验 p = {p:.4g}\n"
        f"按 {dimension} 拆解：结构效应 {mix * 100:+.2f} pp（占 {mix_pct:.0%}），"
        f"风险效应 {rate * 100:+.2f} pp\n{_fmt(table)}\n"
        "说明：mix_effect = (目标占比 - 对照占比) × 对照坏账率；"
        "rate_effect = 目标占比 × (目标坏账率 - 对照坏账率)。"
    )
    return ToolResult(text, warnings, table)


# ---------------------------------------------------------------- 模型评分

@lru_cache(maxsize=1)
def _shap_base_value() -> float:
    df = query("""SELECT r.risk_score, s.shap_sum FROM risk_scores r JOIN
        (SELECT user_id, SUM(shap_value) AS shap_sum FROM shap_values GROUP BY user_id) s USING (user_id)""")
    return float(np.mean(V.logit(df["risk_score"].values) - df["shap_sum"].values))


def _psi(base: np.ndarray, target: np.ndarray, bins: int = 10) -> float:
    edges = np.unique(np.quantile(base, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    pb = np.histogram(base, edges)[0] / len(base)
    pt = np.histogram(target, edges)[0] / len(target)
    pb, pt = np.clip(pb, 1e-6, None), np.clip(pt, 1e-6, None)
    return float(np.sum((pt - pb) * np.log(pt / pb)))


def model_score_summary(target_filter: str, base_filter: str | None = None) -> ToolResult:
    """群体的风险分分布、风险等级占比、预测 vs 实际坏账率；给定对照群体时计算 PSI。"""
    def load(flt):
        return query_view(f"SELECT risk_score, risk_level, is_bad FROM v WHERE {check_filter(flt)}")

    def summarize(df, name):
        return {"群体": name, "n": len(df), "平均风险分": df["risk_score"].mean(),
                "实际坏账率": df["is_bad"].mean(),
                "高风险占比": (df["risk_level"] == "高").mean(),
                "中风险占比": (df["risk_level"] == "中").mean(),
                "低风险占比": (df["risk_level"] == "低").mean()}

    tgt = load(target_filter)
    warnings = V.check_sample(len(tgt), "目标群体") + V.check_scores(tgt["risk_score"])
    rows = [summarize(tgt, "目标")]
    extra = ""
    if base_filter:
        base = load(base_filter)
        warnings += V.check_sample(len(base), "对照群体")
        rows.insert(0, summarize(base, "对照"))
        psi = _psi(base["risk_score"].values, tgt["risk_score"].values)
        level = "稳定" if psi < 0.1 else ("轻微偏移" if psi < 0.25 else "显著偏移")
        extra = f"\n风险分 PSI（目标 vs 对照）= {psi:.4f}，{level}"
    table = pd.DataFrame(rows)
    text = f"目标群体：{target_filter}" + (f"\n对照群体：{base_filter}" if base_filter else "")
    return ToolResult(f"{text}\n{_fmt(table)}{extra}", warnings, table)


def model_performance(filter_expr: str | None = None) -> ToolResult:
    """在测试集（dataset='test'）上评估模型 AUC / KS，可按群体过滤。"""
    f = check_filter(filter_expr)
    df = query_view(f"SELECT risk_score, is_bad FROM v WHERE dataset = 'test' AND ({f})")
    warnings = V.check_sample(len(df), "测试集子群体")
    if df["is_bad"].nunique() < 2:
        return ToolResult(f"测试集子群体 n={len(df)}，只有一种标签，无法计算 AUC。",
                          warnings + ["只有一种标签"], is_error=True)
    auc = roc_auc_score(df["is_bad"], df["risk_score"])
    fpr, tpr, _ = roc_curve(df["is_bad"], df["risk_score"])
    ks = float(np.max(tpr - fpr))
    text = (f"评估范围：测试集 AND ({f})\n样本量 {len(df)}，实际坏账率 {df['is_bad'].mean():.2%}，"
            f"平均风险分 {df['risk_score'].mean():.4f}\nAUC = {auc:.4f}，KS = {ks:.4f}")
    return ToolResult(text, warnings)


# ---------------------------------------------------------------- SHAP 归因

def shap_compare(target_filter: str, base_filter: str | None = None, top_k: int = 8) -> ToolResult:
    """
    给定对照群体：按特征比较两群体的平均 SHAP，差值越大说明该特征越是推高目标群体风险的原因。
    不给对照群体：输出目标群体的全局特征重要性 mean(|SHAP|)。
    """
    def load(flt):
        return query_view(f"""SELECT s.feature, AVG(s.shap_value) AS mean_shap,
            AVG(ABS(s.shap_value)) AS mean_abs_shap, COUNT(DISTINCT s.user_id) AS n
            FROM shap_values s JOIN v USING (user_id) WHERE {check_filter(flt)}
            GROUP BY s.feature""")

    def additivity(flt, df, label):
        sc = query_view(f"SELECT risk_score FROM v WHERE {check_filter(flt)}")["risk_score"].values
        return V.check_shap_additivity(float(np.mean(V.logit(sc))), float(df["mean_shap"].sum()),
                                       _shap_base_value(), label)

    tgt = load(target_filter)
    if tgt.empty:
        return ToolResult("目标群体为空。", ["目标群体样本量为 0"], is_error=True)
    nt = int(tgt["n"].iloc[0])
    warnings = V.check_sample(nt, "目标群体") + additivity(target_filter, tgt, "目标群体")

    if not base_filter:
        table = tgt.sort_values("mean_abs_shap", ascending=False).head(top_k)[
            ["feature", "mean_abs_shap", "mean_shap"]]
        return ToolResult(f"目标群体：{target_filter}（n={nt}）\n特征重要性（按 mean|SHAP| 排序）：\n{_fmt(table)}",
                          warnings, table)

    base = load(base_filter)
    nb = int(base["n"].iloc[0]) if not base.empty else 0
    warnings += V.check_sample(nb, "对照群体") + additivity(base_filter, base, "对照群体")
    m = base[["feature", "mean_shap"]].merge(tgt[["feature", "mean_shap"]], on="feature",
                                             suffixes=("_base", "_target"))
    m["diff"] = m["mean_shap_target"] - m["mean_shap_base"]
    total = m["diff"].sum()
    m["share_of_total"] = m["diff"] / total if total else np.nan
    table = m.sort_values("diff", ascending=False).head(top_k)
    text = (f"对照群体：{base_filter}（n={nb}）\n目标群体：{target_filter}（n={nt}）\n"
            f"平均 log-odds 差异（SHAP 之和）：{total:+.4f}\n"
            f"推高目标群体风险的特征（按平均 SHAP 差值排序）：\n{_fmt(table)}")
    return ToolResult(text, warnings, table)
