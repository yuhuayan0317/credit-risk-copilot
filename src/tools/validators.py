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


def check_sql_result(sql: str, df: pd.DataFrame) -> list[str]:
    warnings = []
    if df.empty:
        warnings.append("查询结果为空。请检查过滤条件：loan_month 格式为 'YYYY-MM'，"
                        "channel 取值需完全匹配（APP自然流量/信息流广告/合作方导流/线下门店）。")
        return warnings

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
    if re.search(r"\beducation\b", lowered) and not re.search(r"\b(0|5|6)\b", lowered):
        warnings.append("涉及 education 字段：存在未定义编码 0/5/6（约 345 人），"
                        "需单独列为「未知」，不能并入有效类别或做业务解读。")
    if re.search(r"\bmarriage\b", lowered) and "0" not in lowered:
        warnings.append("涉及 marriage 字段：存在未定义编码 0（54 人），需单独列为「未知」。")
    if re.search(r"\bauc\b|roc", lowered) and "dataset" not in lowered:
        warnings.append("计算模型效果时未限定 dataset = 'test'，结果会包含训练样本，可能高估模型效果。")
    if re.search(r"avg\s*\(\s*bad_rate|avg\s*\(\s*\w*rate", lowered):
        warnings.append("检测到对比率再求平均，这是月度坏账率的简单平均。按口径应使用用户加权的 AVG(is_bad)。")
    return warnings


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
