"""工具定义（JSON Schema）与统一分发入口。"""
import re

from tools import analysis as A
from tools.validators import check_literals
from tools.db import SEGMENT_DIMS, UnsafeSQLError
from tools.python_runner import UnsafeCodeError, run_python

FILTER_DESC = ("SQL WHERE 条件表达式，作用在用户明细+评分视图上（可用 loans 表全部字段及 risk_score、"
               "risk_level、dataset）。示例：\"channel = '信息流广告' AND loan_month BETWEEN '2025-10' AND '2025-12'\"")

TOOLS = [
    {
        "name": "search_knowledge",
        "description": "检索分析知识库：数据字典、指标口径、标准分层定义、数据质量规则、模型说明、"
                       "异动归因 SOP、历史分析案例。开始任何分析前，先用它确认相关字段含义和指标口径。",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "检索问题，例如「坏账率口径」「年龄段分层标准」"},
                "top_k": {"type": "integer", "description": "返回片段数，默认 4", "minimum": 1, "maximum": 8},
            },
            "required": ["query"],
        },
    },
    {
        "name": "run_sql",
        "description": "在 DuckDB 上执行只读 SQL（SELECT/WITH），表：loans、risk_scores、shap_values。"
                       "用于取数、聚合统计。结果最多展示 60 行，请尽量在 SQL 中完成聚合。"
                       "返回结果附带自动校验提示（空结果、比率越界、小样本、未定义编码等）。",
        "input_schema": {
            "type": "object",
            "properties": {"sql": {"type": "string", "description": "单条只读 SQL"}},
            "required": ["sql"],
        },
    },
    {
        "name": "run_python",
        "description": "在隔离子进程中执行 Python 分析代码。已预置：pd、np、stats（scipy.stats）、"
                       "只读连接 con、函数 sql(query) -> DataFrame。用 print() 输出结果。"
                       "适合 SQL 不方便完成的计算，例如统计检验、复杂透视。只允许导入 pandas/numpy/scipy/sklearn 等分析库。",
        "input_schema": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "Python 代码"}},
            "required": ["code"],
        },
    },
    {
        "name": "segment_compare",
        "description": "按标准分层口径对用户分层，输出各层样本量、占比、坏账率、平均风险分。"
                       "同时给出对照群体时，计算两个群体的坏账率差异和显著性，并把差异拆解为"
                       "结构效应（客群占比变化）和风险效应（同一客群坏账率变化）。是坏账率异动归因的核心工具。",
        "input_schema": {
            "type": "object",
            "properties": {
                "dimension": {"type": "string", "enum": list(SEGMENT_DIMS),
                              "description": "分层维度。age_band 年龄段；limit_band 额度段；young_low 年轻低额度客群；"
                                             "recent_overdue 近期逾期客群；max_overdue 历史最严重逾期；"
                                             "education/marriage 已处理未定义编码；其余为原始字段"},
                "target_filter": {"type": "string", "description": "目标群体。" + FILTER_DESC},
                "base_filter": {"type": "string", "description": "对照群体（可选）。" + FILTER_DESC},
            },
            "required": ["dimension", "target_filter"],
        },
    },
    {
        "name": "shap_compare",
        "description": "基于 SHAP 的群体风险归因。给出对照群体时，按特征比较两个群体的平均 SHAP 值，"
                       "差值越大的特征越是导致目标群体模型风险更高的原因；不给对照群体时，输出目标群体的特征重要性。"
                       "自动做 SHAP 可加性校验。",
        "input_schema": {
            "type": "object",
            "properties": {
                "target_filter": {"type": "string", "description": "目标群体。" + FILTER_DESC},
                "base_filter": {"type": "string", "description": "对照群体（可选）。" + FILTER_DESC},
                "top_k": {"type": "integer", "description": "展示特征数，默认 8", "minimum": 1, "maximum": 27},
            },
            "required": ["target_filter"],
        },
    },
    {
        "name": "model_score_summary",
        "description": "调用风险模型评分：输出群体的平均风险分、实际坏账率、高/中/低风险占比；"
                       "给出对照群体时计算风险分 PSI，判断客群是否发生偏移。",
        "input_schema": {
            "type": "object",
            "properties": {
                "target_filter": {"type": "string", "description": "目标群体。" + FILTER_DESC},
                "base_filter": {"type": "string", "description": "对照群体（可选）。" + FILTER_DESC},
            },
            "required": ["target_filter"],
        },
    },
    {
        "name": "model_performance",
        "description": "在测试集上评估风险模型的 AUC 和 KS，可限定子群体（例如某渠道）。",
        "input_schema": {
            "type": "object",
            "properties": {"filter": {"type": "string", "description": "子群体（可选）。" + FILTER_DESC}},
        },
    },
    {
        "name": "data_quality_check",
        "description": "检查某个群体的数据质量：空值、未定义编码、异常取值、溢缴款等。",
        "input_schema": {
            "type": "object",
            "properties": {"filter": {"type": "string", "description": "群体（可选，默认全量）。" + FILTER_DESC}},
        },
    },
]


def _dispatch(name: str, args: dict) -> A.ToolResult:
    # 过滤条件里写了不存在的取值时直接报错，否则会得到空群体并被误读为「没有数据」
    bad = [p for k in ("target_filter", "base_filter", "filter") if args.get(k) for p in check_literals(args[k])]
    if bad:
        return A.ToolResult("过滤条件有误：\n" + "\n".join(bad), is_error=True)
    if name == "search_knowledge":
        return A.search_knowledge(args["query"], int(args.get("top_k") or 4))
    if name == "run_sql":
        return A.run_sql(args["sql"])
    if name == "run_python":
        ok, out = run_python(args["code"])
        return A.ToolResult(out, [] if ok else ["代码执行失败，请根据报错修正后重试。"], is_error=not ok)
    if name == "segment_compare":
        return A.segment_compare(args["dimension"], args["target_filter"], args.get("base_filter"))
    if name == "shap_compare":
        return A.shap_compare(args["target_filter"], args.get("base_filter"), int(args.get("top_k") or 8))
    if name == "model_score_summary":
        return A.model_score_summary(args["target_filter"], args.get("base_filter"))
    if name == "model_performance":
        return A.model_performance(args.get("filter"))
    if name == "data_quality_check":
        return A.data_quality_check(args.get("filter"))
    raise KeyError(f"未知工具 {name}")


def execute_tool(name: str, args: dict) -> A.ToolResult:
    """统一入口：任何异常都转成 is_error 的结果返回给模型，让它自行修正，而不是让 Agent 崩溃。"""
    try:
        return _dispatch(name, args)
    except (UnsafeSQLError, UnsafeCodeError) as e:
        return A.ToolResult(f"安全检查未通过：{e}", is_error=True)
    except KeyError as e:
        return A.ToolResult(f"参数缺失或工具不存在：{e}", is_error=True)
    except Exception as e:  # noqa: BLE001 - SQL 语法错误、列名错误等都应反馈给模型
        return A.ToolResult(f"{type(e).__name__}: {str(e)[:800]}{_hint(str(e))}", is_error=True)


def _hint(err: str) -> str:
    """把常见报错翻译成可操作的修改建议。"""
    m = re.search(r'column "?(\w+)"? not found', err, re.IGNORECASE)
    if m and m.group(1) in ("risk_score", "risk_level", "dataset"):
        return f"\n修改建议：{m.group(1)} 在 risk_scores 表中，需要 JOIN risk_scores USING (user_id)。"
    if m and m.group(1) in SEGMENT_DIMS:
        return (f"\n修改建议：{m.group(1)} 不是表字段，而是 segment_compare 工具的分层维度。"
                f"请改用 segment_compare(dimension='{m.group(1)}', target_filter=..., base_filter=...)，"
                "或在 SQL 中用 CASE WHEN 按知识库的分层口径自己构造。")
    if m:
        return "\n修改建议：请对照数据字典检查字段名（可用 search_knowledge 检索「数据字典」）。"
    return ""
