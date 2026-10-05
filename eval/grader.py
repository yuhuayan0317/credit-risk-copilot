"""评分：数值/关键词检查走规则，开放性结论走 LLM 评审。"""
import json
import re

_NUM = re.compile(r"(?<![A-Za-z0-9_.])([-+]?\d+(?:,\d{3})*(?:\.\d+)?)\s*(%|％)?")

JUDGE_SYSTEM = """你是信贷风险分析报告的评审员。根据「评分标准」判断「待评报告」是否正确回答了「业务问题」。
「参考事实」是由数据计算得到的标准答案，供你核对报告中的数字和结论。
- 只按评分标准判断，标准中的每一条要求都满足才算通过。
- 数字允许合理的四舍五入误差（例如 29.54% 写成 29.5% 或 30%）。
- 表述方式不同但含义一致，视为满足。
- 报告的结论与参考事实矛盾时，判为不通过。
- 评分标准要求的关键数字（如坏账率、占比）必须出现在报告中，且与参考事实一致；只有定性描述、没有数据支撑的，判为不通过。
- 报告的分析对象是 2025 年的数据。如果报告引用的是历史案例（如 2024 年）的结论或数字，而没有当前数据支撑，判为不通过。
- 宁严勿松：拿不准时判为不通过。
只输出一个 JSON 对象，不要输出其他内容：{"reason": "逐条对照评分标准的简要说明", "passed": true 或 false}"""


def _numbers(text: str) -> list[tuple[float, bool]]:
    out = []
    for raw, pct in _NUM.findall(text):
        try:
            out.append((float(raw.replace(",", "")), bool(pct)))
        except ValueError:
            pass
    return out


def check_number(answer: str, value: float, kind: str) -> bool:
    nums = _numbers(answer)
    if kind == "rate":
        # 报告可能写 22.1% / 22.12% / 0.221 / 8.4pp
        return any(abs(abs(x) - abs(value) * 100) <= 0.15 for x, _ in nums) or \
            any(abs(abs(x) - abs(value)) <= 0.0015 for x, pct in nums if not pct)
    if kind in ("auc", "psi", "score"):
        return any(abs(x - value) <= 0.006 for x, pct in nums if not pct)
    return any(abs(x - value) <= max(0.5, 0.01 * abs(value)) for x, _ in nums)


def check_contains(answer: str, words: list[str]) -> bool:
    low = answer.lower()
    return any(w.lower() in low for w in words)


def parse_judgement(text: str) -> tuple[bool, str]:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
            return bool(data.get("passed")), str(data.get("reason", ""))
        except json.JSONDecodeError:
            pass
    # JSON 解析失败时退化为关键词判断
    passed = bool(re.search(r'"?passed"?\s*[:：]\s*true', text, re.IGNORECASE))
    return passed, f"(JSON 解析失败) {text[:200]}"


def judge(backend, question: str, facts: str, rubric: str, answer: str) -> tuple[bool, str]:
    text, _ = backend.complete(
        JUDGE_SYSTEM,
        f"业务问题：{question}\n\n参考事实：{facts or '（无）'}\n\n评分标准：{rubric}\n\n待评报告：\n{answer}",
        json_mode=True)
    return parse_judgement(text)


def grade(judge_backend, q: dict, answer: str) -> dict:
    details = []
    for c in q["checks"]:
        if c["type"] == "number":
            ok, note = check_number(answer, c["value"], c["kind"]), f"期望数值 {c['value']}"
        elif c["type"] == "contains":
            ok, note = check_contains(answer, c["any"]), f"期望包含 {c['any']}"
        else:
            ok, note = judge(judge_backend, q["question"], q.get("facts", ""), c["rubric"], answer)
        details.append({"type": c["type"], "passed": ok, "note": note})
    return {"passed": all(d["passed"] for d in details), "checks": details}
