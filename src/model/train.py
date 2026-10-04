"""
训练 XGBoost 违约风险模型，并把评分与 SHAP 值写回 DuckDB 供 Agent 查询。

产出：
  artifacts/xgb_model.json      模型
  artifacts/model_metrics.json  AUC / KS 等指标
  DuckDB 表 risk_scores         user_id, risk_score, risk_level
  DuckDB 表 shap_values         user_id + 每个特征的 SHAP 值（长表）
"""
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from model.features import FEATURES, build_features  # noqa: E402

DB_PATH = ROOT / "data" / "processed" / "credit.duckdb"
ART = ROOT / "artifacts"
SEED = 42

PARAMS = dict(
    n_estimators=400,
    learning_rate=0.03,
    max_depth=4,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    eval_metric="auc",
    early_stopping_rounds=50,
    random_state=SEED,
)


def ks_stat(y_true, y_score) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def risk_level(score: np.ndarray) -> np.ndarray:
    return np.select([score >= 0.5, score >= 0.25], ["高", "中"], default="低")


def main() -> None:
    con = duckdb.connect(str(DB_PATH))
    df = con.sql("SELECT * FROM loans").df()
    X, y = build_features(df), df["is_bad"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tr, y_tr, test_size=0.2, stratify=y_tr, random_state=SEED)

    model = xgb.XGBClassifier(**PARAMS)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    p_te = model.predict_proba(X_te)[:, 1]
    p_tr = model.predict_proba(X_tr)[:, 1]
    metrics = {
        "test_auc": round(roc_auc_score(y_te, p_te), 4),
        "test_ks": round(ks_stat(y_te, p_te), 4),
        "train_auc": round(roc_auc_score(y_tr, p_tr), 4),
        "best_iteration": int(model.best_iteration),
        "n_features": len(FEATURES),
        "n_train": len(X_tr), "n_valid": len(X_va), "n_test": len(X_te),
    }
    print(json.dumps(metrics, indent=2))

    ART.mkdir(exist_ok=True)
    model.save_model(ART / "xgb_model.json")
    (ART / "model_metrics.json").write_text(json.dumps(metrics, indent=2))

    # 全量评分 + SHAP，写回数据库
    score = model.predict_proba(X)[:, 1]
    scores = pd.DataFrame({"user_id": df["user_id"], "risk_score": score.round(4),
                           "risk_level": risk_level(score)})
    sv = shap.TreeExplainer(model).shap_values(X)
    shap_long = (pd.DataFrame(sv, columns=FEATURES)
                 .assign(user_id=df["user_id"].values)
                 .melt(id_vars="user_id", var_name="feature", value_name="shap_value"))

    for name, frame in [("risk_scores", scores), ("shap_values", shap_long)]:
        con.register("tmp", frame)
        con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM tmp")
        con.unregister("tmp")
    con.close()

    top = pd.Series(np.abs(sv).mean(axis=0), index=FEATURES).sort_values(ascending=False)
    print("top features by mean |SHAP|:\n", top.head(8).round(4))


if __name__ == "__main__":
    main()
