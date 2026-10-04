"""
数据准备：读取 UCI 信用卡违约数据，补充模拟业务字段，写入 DuckDB。

原始数据没有时间与渠道维度，无法回答「某类用户坏账率突然上升」这类问题。
这里按「客群画像 -> 渠道/月份」的方式分配业务字段，**不修改任何真实标签**：
坏账率的异动完全来自客群结构变化，因此 Agent 可以用分层对比 + SHAP 找到真实原因。

注入的两个场景（用于评测集的标准答案）：
  S1. 2025-10 ~ 2025-12，「信息流广告」渠道放量引入「年轻 + 低额度」客群，
      该渠道坏账率显著上升。
  S2. 2025-06，「合作方导流」渠道混入较多「近期已有逾期记录」的客户，坏账率单月冲高。
"""
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "default of credit card clients.xls"
DB_PATH = ROOT / "data" / "processed" / "credit.duckdb"
SEED = 42

MONTHS = [f"2025-{m:02d}" for m in range(1, 13)]
CHANNELS = ["APP自然流量", "信息流广告", "合作方导流", "线下门店"]
CITY_TIERS = ["一线", "二线", "三线及以下"]


def load_raw() -> pd.DataFrame:
    df = pd.read_excel(RAW, header=1)
    rename = {
        "ID": "user_id",
        "LIMIT_BAL": "credit_limit",
        "SEX": "sex",
        "EDUCATION": "education",
        "MARRIAGE": "marriage",
        "AGE": "age",
        "default payment next month": "is_bad",
    }
    # PAY_0 是最近一个月，原始命名跳过了 PAY_1，统一成 _1 ~ _6（1 = 最近）
    for i, src in enumerate(["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"], 1):
        rename[src] = f"pay_status_{i}"
    for i in range(1, 7):
        rename[f"BILL_AMT{i}"] = f"bill_amt_{i}"
        rename[f"PAY_AMT{i}"] = f"pay_amt_{i}"
    return df.rename(columns=rename)


def assign_business_fields(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    n = len(df)

    young_low = (df["age"] <= 30) & (df["credit_limit"] <= 50_000)
    recent_overdue = df["pay_status_1"] >= 1

    # --- 渠道：信息流偏年轻，线下门店偏年长，合作方导流略偏有逾期史 ---
    base = np.tile([0.35, 0.25, 0.20, 0.20], (n, 1))
    base[young_low.values] = [0.25, 0.45, 0.20, 0.10]
    base[(df["age"] >= 45).values] = [0.30, 0.10, 0.20, 0.40]
    base = base / base.sum(axis=1, keepdims=True)
    channel_idx = np.array([rng.choice(4, p=p) for p in base])
    df["channel"] = np.array(CHANNELS)[channel_idx]

    # --- 月份：默认均匀，按场景调整权重 ---
    month_w = np.ones((n, 12))
    # S1：信息流 + 年轻低额度 -> 集中在 10~12 月；信息流其他客群 -> 10~12 月减少
    s1_hit = (df["channel"] == "信息流广告") & young_low
    s1_other = (df["channel"] == "信息流广告") & ~young_low
    month_w[s1_hit.values, 9:12] *= 12.0
    month_w[s1_other.values, 9:12] *= 0.3
    # S2：合作方导流 + 近期逾期 -> 集中在 6 月
    s2_hit = (df["channel"] == "合作方导流") & recent_overdue
    month_w[s2_hit.values, 5] *= 8.0
    month_w = month_w / month_w.sum(axis=1, keepdims=True)
    month_idx = np.array([rng.choice(12, p=p) for p in month_w])
    df["loan_month"] = np.array(MONTHS)[month_idx]

    # --- 城市等级：与风险弱相关，作为干扰维度 ---
    df["city_tier"] = rng.choice(CITY_TIERS, size=n, p=[0.3, 0.4, 0.3])

    return df


def build_db(df: pd.DataFrame) -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()
    con = duckdb.connect(str(DB_PATH))
    cols = (
        ["user_id", "loan_month", "channel", "city_tier",
         "credit_limit", "sex", "education", "marriage", "age"]
        + [f"pay_status_{i}" for i in range(1, 7)]
        + [f"bill_amt_{i}" for i in range(1, 7)]
        + [f"pay_amt_{i}" for i in range(1, 7)]
        + ["is_bad"]
    )
    con.register("df_view", df[cols])
    con.execute("CREATE TABLE loans AS SELECT * FROM df_view")
    con.close()


def main() -> None:
    df = assign_business_fields(load_raw())
    build_db(df)

    print(f"rows={len(df)}, overall bad rate={df.is_bad.mean():.3f}")
    pivot = df.pivot_table(index="loan_month", columns="channel",
                           values="is_bad", aggfunc="mean").round(3)
    print(pivot)
    print(f"saved -> {DB_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
