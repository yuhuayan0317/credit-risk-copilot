"""
结果可验证性校验：检查最终报告中的数字是否都能在工具返回结果中找到依据。

匹配规则（考虑报告中常见的换算和四舍五入）：
  - 报告中的 x 或 x% 能匹配到工具结果中的 y、y×100（容差按报告的小数位数决定）
  - 「xx pp / 个百分点」还可以匹配工具结果中两个比率之差
"""
import re

import numpy as np

_NUM = re.compile(r"(?<![A-Za-z0-9_.])([-+]?\d+(?:,\d{3})*(?:\.\d+)?)\s*(%)?")
_REPORT_NUM = re.compile(
    r"(?<![A-Za-z0-9_.\-])([-+]?\d+(?:,\d{3})*(?:\.\d+)?)\s*(%|％|pp|个百分点|百分点)?")

# 这些数字是常识/口径定义，不需要工具结果背书
_TRIVIAL = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 100, 300, 2025}


def extract_tool_numbers(texts: list[str]) -> np.ndarray:
    nums = []
    for t in texts:
        for raw, pct in _NUM.findall(t):
            val = float(raw.replace(",", ""))
            nums.append(val)
            if pct:  # 「21.15%」同时记为 0.2115，便于计算百分点差
                nums.append(val / 100)
    return np.unique(np.array(nums, dtype=float)) if nums else np.array([])


def _decimals(s: str) -> int:
    return len(s.split(".")[1]) if "." in s else 0


def ungrounded_numbers(report: str, tool_texts: list[str]) -> list[str]:
    pool = extract_tool_numbers(tool_texts)
    if pool.size == 0:
        return []
    rates = pool[(pool >= 0) & (pool <= 1)]
    # 比率两两之差（用于匹配 pp），比率数量很多时只保留前 600 个避免组合爆炸
    rates = rates[:600]
    diffs = np.abs(rates[:, None] - rates[None, :]).ravel() if rates.size else np.array([])

    # 去掉报告中的日期、月份、年龄段等形如 2025-10 / 26-30 的片段，避免误报
    cleaned = re.sub(r"\d{4}-\d{2}(?:-\d{2})?|\d+\s*[-~～至]\s*\d+\s*(?:岁|万|月)", " ", report)
    cleaned = re.sub(r"(?:pay_status|bill_amt|pay_amt|utilization|repay_ratio)_\d", " ", cleaned)
    cleaned = re.sub(r"[QqMm]\d\+?", " ", cleaned)

    missing = []
    for raw, unit in _REPORT_NUM.findall(cleaned):
        val = float(raw.replace(",", ""))
        if abs(val) in _TRIVIAL and not unit:
            continue
        tol = 0.5 * 10 ** (-_decimals(raw)) + 1e-9
        cands = [pool]
        if unit:
            cands.append(pool * 100)
            if unit in ("pp", "个百分点", "百分点") and diffs.size:
                cands.append(diffs * 100)
        hit = any(np.any(np.abs(np.abs(c) - abs(val)) <= tol) for c in cands)
        # 报告常把 1.234 万 写成 12340 等，这里只做严格匹配，未命中即报告
        if not hit:
            missing.append(raw + (unit or ""))
    # 去重并保持顺序
    seen, out = set(), []
    for m in missing:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out
