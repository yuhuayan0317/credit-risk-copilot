"""只读数据库访问 + 标准分层口径（与知识库《指标口径》保持一致）。"""
import re
from functools import lru_cache

import duckdb
import pandas as pd

from config import DB_PATH

# 统一的分析视图：用户明细 + 模型评分
BASE_VIEW = """
WITH v AS (
    SELECT l.*, r.risk_score, r.risk_level, r.dataset
    FROM loans l JOIN risk_scores r USING (user_id)
)"""

# 标准分层口径 -> SQL 表达式
SEGMENT_DIMS: dict[str, str] = {
    "age_band": """CASE WHEN age <= 25 THEN '1_25岁及以下' WHEN age <= 30 THEN '2_26-30岁'
        WHEN age <= 40 THEN '3_31-40岁' WHEN age <= 50 THEN '4_41-50岁' ELSE '5_50岁以上' END""",
    "limit_band": """CASE WHEN credit_limit <= 30000 THEN '1_3万及以下'
        WHEN credit_limit <= 50000 THEN '2_3-5万' WHEN credit_limit <= 100000 THEN '3_5-10万'
        WHEN credit_limit <= 200000 THEN '4_10-20万' ELSE '5_20万以上' END""",
    "young_low": "CASE WHEN age <= 30 AND credit_limit <= 50000 THEN '年轻低额度客群' ELSE '其他客群' END",
    "recent_overdue": "CASE WHEN pay_status_1 >= 1 THEN '近期逾期客群' ELSE '近期未逾期' END",
    "max_overdue": """CASE WHEN GREATEST(pay_status_1, pay_status_2, pay_status_3, pay_status_4,
        pay_status_5, pay_status_6) >= 2 THEN '历史M2+' WHEN GREATEST(pay_status_1, pay_status_2,
        pay_status_3, pay_status_4, pay_status_5, pay_status_6) = 1 THEN '历史M1' ELSE '无逾期' END""",
    "education": """CASE education WHEN 1 THEN '研究生' WHEN 2 THEN '本科' WHEN 3 THEN '高中'
        WHEN 4 THEN '其他' ELSE '未知(未定义编码)' END""",
    "marriage": "CASE marriage WHEN 1 THEN '已婚' WHEN 2 THEN '单身' WHEN 3 THEN '其他' ELSE '未知(未定义编码)' END",
    "sex": "CASE sex WHEN 1 THEN '男' ELSE '女' END",
    "channel": "channel",
    "city_tier": "city_tier",
    "loan_month": "loan_month",
    "risk_level": "risk_level",
}

_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|create|alter|attach|detach|copy|pragma|install|load|"
    r"export|import|call|set|checkpoint|vacuum)\b", re.IGNORECASE)


class UnsafeSQLError(ValueError):
    pass


@lru_cache(maxsize=1)
def get_con() -> duckdb.DuckDBPyConnection:
    # 只读 + 禁止访问外部文件/网络：这是真正的安全边界，下面的正则检查只是提前给出友好报错
    return duckdb.connect(str(DB_PATH), read_only=True,
                          config={"enable_external_access": False})


def check_select(sql: str) -> str:
    sql = sql.strip().rstrip(";").strip()
    if ";" in sql:
        raise UnsafeSQLError("只允许执行单条 SQL 语句")
    if not re.match(r"^(select|with)\b", sql, re.IGNORECASE):
        raise UnsafeSQLError("只允许 SELECT / WITH 查询")
    if _FORBIDDEN.search(sql):
        raise UnsafeSQLError(f"SQL 包含禁止的关键字：{_FORBIDDEN.search(sql).group(0)}")
    return sql


def check_filter(expr: str | None) -> str:
    """过滤条件是拼到 WHERE 后面的表达式，例如 "channel = '信息流广告' AND loan_month >= '2025-10'"。"""
    if not expr or not expr.strip():
        return "TRUE"
    if ";" in expr or "--" in expr or "/*" in expr:
        raise UnsafeSQLError("过滤条件中不允许出现 ; 或注释")
    if re.search(r"\bselect\b", expr, re.IGNORECASE) or _FORBIDDEN.search(expr):
        raise UnsafeSQLError("过滤条件中不允许出现子查询或写操作关键字")
    return expr


def query(sql: str) -> pd.DataFrame:
    return get_con().execute(check_select(sql)).df()


def query_view(select_sql: str) -> pd.DataFrame:
    """在 BASE_VIEW（别名 v）上执行查询。"""
    return get_con().execute(BASE_VIEW + "\n" + select_sql).df()
