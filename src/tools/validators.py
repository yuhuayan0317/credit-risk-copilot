"""
校验层：每个工具的输出在返回给 LLM 之前都要过一遍校验，把发现的问题以 warning 的形式附在结果后面。

三类校验：
  1. 数据质量校验：未定义编码、空值、异常取值
  2. SQL 结果检查：空结果、比率越界、小样本、分组加总不一致
  3. 模型输出验证：评分范围、SHAP 可加性、评估样本是否混入训练集
"""
import re

import numpy as np
import pandas as pd

MIN_SAMPLE = 100       # 低于此样本量不下结论
SUGGEST_SAMPLE = 300   # 低于此样本量需注明仅供参考

_RATE_COL = re.compile(r"(rate|ratio|share|pct|率|占比|比例)", re.IGNORECASE)
_COUNT_COL = re.compile(r"^(n|cnt|count|num|users?|n_users|样本量?|人数|用户数)$|^n_|_cnt$|_count$|_n$",
                        re.IGNORECASE)


KNOWN_VALUES = {
    "channel": ["APP自然流量", "信息流广告", "合作方导流", "线下门店"],
    "city_tier": ["一线", "二线", "三线及以下"],
    "risk_level": ["高", "中", "低"],
    "dataset": ["train", "valid", "test"],
}
_MONTH = re.compile(r"^2025-(0[1-9]|1[0-2])$")


def check_literals(sql: str) -> list[str]:
    """检查 SQL / 过滤条件中的字符串常量是否是该字段真实存在的取值。"""
    problems = []
    for col, valid in KNOWN_VALUES.items():
        pat = rf"\b{col}\s*(?:=|<>|!=)\s*'([^']*)'|\b{col}\s+(?:not\s+)?in\s*\(([^)]*)\)"
        for eq, in_list in re.findall(pat, sql, re.IGNORECASE):
            values = [eq] if eq else re.findall(r"'([^']*)'", in_list)
            for val in values:
                if val not in valid:
                    problems.append(f"{col} 不存在取值 '{val}'，有效取值为：{'/'.join(valid)}")
    for val in re.findall(r"\bloan_month\s*(?:=|<>|!=|>=|<=|>|<)\s*'([^']*)'", sql, re.IGNORECASE):
        if not _MONTH.match(val):
            problems.append(f"loan_month 取值 '{val}' 格式不对，应为 '2025-01' ~ '2025-12'")
    return problems


def check_sql_result(sql: str, df: pd.DataFrame) -> list[str]:
    warnings = check_literals(sql)
    if df.empty:
        warnings.append("查询结果为空。请检查过滤条件：loan_month 格式为 'YYYY-MM'，"
                        "channel 取值需完全匹配（APP自然流量/信息流广告/合作方导流/线下门店）。")
        return warnings

    numeric = df.select_dtypes("number")
    if len(df) == 1 and not numeric.empty and (numeric.isna().all(axis=None) or
                                               any(_COUNT_COL.search(str(c)) or "count" in str(c).lower()
                                                   for c in numeric.columns[(numeric == 0).all()])):
        warnings.append("过滤条件没有匹配到任何记录（计数为 0 或结果为空值），"
                        "这通常说明过滤条件写错了，而不是数据真的不存在。请先检查字段取值再下结论。")

    for col in df.columns:
        s = df[col]
        if s.isna().any():
            warnings.append(f"列 {col} 有 {int(s.isna().sum())} 个空值，确认是否由 JOIN 不匹配或除零导致。")
        if not pd.api.types.is_numeric_dtype(s):
            continue
        if _RATE_COL.search(str(col)):
            lo, hi = s.min(), s.max()
            if hi > 1 and hi <= 100:
                warnings.append(f"比率列 {col} 最大值为 {hi:.4g}，可能已乘以 100，展示时注意不要重复换算。")
            elif hi > 100 or lo < 0:
                warnings.append(f"比率列 {col} 超出合理范围 [{lo:.4g}, {hi:.4g}]，请检查计算逻辑。")
        if _COUNT_COL.search(str(col)) and len(df) > 1:
            small = df[s < MIN_SAMPLE]
            mid = df[(s >= MIN_SAMPLE) & (s < SUGGEST_SAMPLE)]
            if len(small):
                warnings.append(f"{len(small)} 个分组的样本量 < {MIN_SAMPLE}（列 {col}），"
                                "这些分组的比率不应作为结论依据。")
            elif len(mid):
                warnings.append(f"{len(mid)} 个分组的样本量在 {MIN_SAMPLE}~{SUGGEST_SAMPLE} 之间，"
                                "结论需注明「仅供参考」。")

    lowered = sql.lower()
    # 比率在 0~1 之间却只保留 0~2 位小数，会把 23.7% 变成 0.2，严重失真
    if _rounds_rate_coarsely(lowered):
        rate_cols = [c for c in df.columns if _RATE_COL.search(str(c)) and pd.api.types.is_numeric_dtype(df[c])]
        if any(df[c].between(0, 1).all() for c in rate_cols):
            warnings.append("比率被 ROUND 到 2 位小数以内，精度严重丢失（例如 0.237 会变成 0.2）。"
                            "请去掉 ROUND 或保留 4 位小数后重新查询，不要使用当前结果。")
    if re.search(r"\beducation\b", lowered):
        warnings.append("education 编码含义：1=研究生，2=本科，3=高中，4=其他；0/5/6 为未定义编码（约 345 人），"
                        "需单独列为「未知」，不能并入有效类别或做业务解读。")
    if re.search(r"\bmarriage\b", lowered):
        warnings.append("marriage 编码含义：1=已婚，2=单身，3=其他；0 为未定义编码（54 人），需单独列为「未知」。")
    if re.search(r"\bauc\b|roc", lowered) and "dataset" not in lowered:
        warnings.append("计算模型效果时未限定 dataset = 'test'，结果会包含训练样本，可能高估模型效果。")
    if re.search(r"avg\s*\(\s*bad_rate|avg\s*\(\s*\w*rate", lowered):
        warnings.append("检测到对比率再求平均，这是月度坏账率的简单平均。按口径应使用用户加权的 AVG(is_bad)。")
    return warnings


def _rounds_rate_coarsely(sql: str) -> bool:
    """是否存在 ROUND(<含 is_bad/rate 的表达式>, 0|1|2)。用括号配对解析，正则处理不了嵌套括号。"""
    for m in re.finditer(r"round\s*\(", sql):
        depth, i = 1, m.end()
        while i < len(sql) and depth:
            depth += {"(": 1, ")": -1}.get(sql[i], 0)
            i += 1
        inner = sql[m.end():i - 1]
        # 最后一个顶层逗号之后是小数位数
        level, cut = 0, -1
        for j, ch in enumerate(inner):
            level += {"(": 1, ")": -1}.get(ch, 0)
            if ch == "," and level == 0:
                cut = j
        if cut >= 0 and inner[cut + 1:].strip() in {"0", "1", "2"} and re.search(r"is_bad|rate|ratio", inner[:cut]):
            if "100" not in inner[:cut]:
                return True
    return False


def check_groups_sum(seg: pd.DataFrame, n_col: str, expected_total: int, label: str) -> list[str]:
    total = int(seg[n_col].sum())
    if total != expected_total:
        return [f"{label} 各分组样本量之和 {total} ≠ 总样本量 {expected_total}，可能有记录未被分组覆盖。"]
    return []


def check_sample(n: int, label: str) -> list[str]:
    if n < MIN_SAMPLE:
        return [f"{label} 样本量仅 {n}（< {MIN_SAMPLE}），不足以支撑结论。"]
    if n < SUGGEST_SAMPLE:
        return [f"{label} 样本量 {n}（< {SUGGEST_SAMPLE}），结论仅供参考。"]
    return []


def check_scores(scores: pd.Series) -> list[str]:
    if scores.isna().any():
        return [f"有 {int(scores.isna().sum())} 个用户没有模型评分。"]
    if scores.min() < 0 or scores.max() > 1:
        return [f"模型评分超出 [0, 1] 范围：[{scores.min()}, {scores.max()}]。"]
    return []


def check_shap_additivity(mean_logit: float, mean_shap_sum: float, base_value: float,
                          label: str, tol: float = 0.05) -> list[str]:
    """群体平均 log-odds 应等于 基准值 + 平均 SHAP 之和（SHAP 的可加性）。"""
    gap = abs(mean_logit - (base_value + mean_shap_sum))
    if gap > tol:
        return [f"{label} 的 SHAP 可加性校验未通过（偏差 {gap:.3f}），评分与 SHAP 数据可能不一致。"]
    return []


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))
