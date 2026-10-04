"""模型特征定义：23 个原始特征 + 4 个衍生特征。训练与推理共用。"""
import numpy as np
import pandas as pd

RAW_FEATURES = (
    ["credit_limit", "sex", "education", "marriage", "age"]
    + [f"pay_status_{i}" for i in range(1, 7)]
    + [f"bill_amt_{i}" for i in range(1, 7)]
    + [f"pay_amt_{i}" for i in range(1, 7)]
)

DERIVED_FEATURES = [
    "utilization_1",       # 最近一期账单金额 / 授信额度
    "max_pay_status",      # 近 6 期最严重的还款状态
    "n_overdue_months",    # 近 6 期出现逾期（状态 >= 1）的月份数
    "repay_ratio_1",       # 最近一期还款额 / 上一期账单金额
]

FEATURES = RAW_FEATURES + DERIVED_FEATURES

PAY_STATUS_COLS = [f"pay_status_{i}" for i in range(1, 7)]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df[RAW_FEATURES].copy()
    out["utilization_1"] = df["bill_amt_1"] / df["credit_limit"]
    out["max_pay_status"] = df[PAY_STATUS_COLS].max(axis=1)
    out["n_overdue_months"] = (df[PAY_STATUS_COLS] >= 1).sum(axis=1)
    prev_bill = df["bill_amt_2"].where(df["bill_amt_2"] > 0)
    out["repay_ratio_1"] = (df["pay_amt_1"] / prev_bill).clip(upper=5).fillna(np.nan)
    return out[FEATURES]
