"""
生成 50 题业务分析评测集，标准答案全部由 SQL 从数据库实时计算，保证可复现。

每道题包含若干 checks，全部通过才算答对：
  - number：报告中需出现该数值（比率按百分数匹配，容差 ±0.15pp；其他数值按相对误差 1% 匹配）
  - contains：报告中需出现候选词之一（any），或不能出现某些词（none）
  - judge：由 LLM 评审按 rubric 判断（用于归因类等开放问题）
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tools.analysis import _psi  # noqa: E402
from tools.db import query, query_view  # noqa: E402

OUT = ROOT / "eval" / "questions.jsonl"


def v(sql: str) -> float:
    return float(query(sql).iloc[0, 0])


def br(where: str) -> float:
    return v(f"SELECT AVG(is_bad) FROM loans WHERE {where}")


def n(where: str) -> int:
    return int(v(f"SELECT COUNT(*) FROM loans WHERE {where}"))


def rate(x: float) -> dict:
    return {"type": "number", "value": round(x, 6), "kind": "rate"}


def num(x: float) -> dict:
    return {"type": "number", "value": round(x, 6), "kind": "abs"}


def has(*words: str) -> dict:
    return {"type": "contains", "any": list(words)}


def judge(rubric: str) -> dict:
    return {"type": "judge", "rubric": rubric}


def auc(where: str) -> float:
    df = query_view(f"SELECT risk_score, is_bad FROM v WHERE dataset='test' AND ({where})")
    return float(roc_auc_score(df["is_bad"], df["risk_score"]))


def build() -> list[dict]:
    Q: list[dict] = []

    def add(cat: str, question: str, checks: list[dict], facts: str = ""):
        Q.append({"id": f"{cat}{sum(q['category'] == cat for q in Q) + 1:02d}", "category": cat,
                  "question": question, "checks": checks, "facts": facts})

    # 常用群体
    S1_T = "channel='信息流广告' AND loan_month>='2025-10'"
    S1_B = "channel='信息流广告' AND loan_month<'2025-10'"
    S2_T = "channel='合作方导流' AND loan_month='2025-06'"
    S2_B = "channel='合作方导流' AND loan_month<>'2025-06'"
    YL = "age<=30 AND credit_limit<=50000"

    # ============ A. 指标取数（12）============
    A = "A"
    add(A, "2025年全年的整体坏账率是多少？", [rate(br("TRUE"))])
    ch = query("SELECT channel, AVG(is_bad) b FROM loans GROUP BY 1 ORDER BY b DESC")
    add(A, "2025年各获客渠道的坏账率分别是多少？哪个渠道最高？",
        [rate(b) for b in ch["b"]] + [has(ch["channel"].iloc[0])],
        "; ".join(f"{c}:{b:.2%}" for c, b in zip(ch["channel"], ch["b"])))
    add(A, "2025年第四季度放款用户的整体坏账率是多少？", [rate(br("loan_month>='2025-10'"))])
    add(A, "合作方导流渠道2025年6月放款的用户，坏账率是多少？", [rate(br(S2_T))])
    add(A, "一线城市用户的坏账率是多少？", [rate(br("city_tier='一线'"))])
    add(A, "授信额度在3万及以下的用户坏账率是多少？", [rate(br("credit_limit<=30000"))])
    cur_od = v("SELECT AVG((pay_status_1>=1)::INT) FROM loans")
    add(A, "当前逾期率是多少？", [rate(cur_od)], "口径：pay_status_1>=1 的用户占比")
    add(A, "模型判定为高风险的用户占全部用户的比例是多少？",
        [rate(v("SELECT AVG((risk_level='高')::INT) FROM risk_scores"))])
    add(A, "年轻低额度客群有多少用户？坏账率是多少？", [num(n(YL)), rate(br(YL))],
        "口径：age<=30 且 credit_limit<=50000")
    hist_od = v("""SELECT AVG((GREATEST(pay_status_1,pay_status_2,pay_status_3,pay_status_4,
        pay_status_5,pay_status_6)>=1)::INT) FROM loans""")
    add(A, "近6个月内出现过逾期的用户占比（历史逾期率）是多少？", [rate(hist_od)])
    add(A, "额度使用率达到80%及以上的用户，坏账率是多少？",
        [rate(br("bill_amt_1*1.0/credit_limit>=0.8"))], "口径：bill_amt_1/credit_limit>=0.8")
    d = br(S1_T) - br(S1_B)
    add(A, "信息流广告渠道第四季度的坏账率，比前三季度高多少个百分点？",
        [{"type": "number", "value": round(d, 6), "kind": "rate"}],
        f"Q4 {br(S1_T):.2%}，Q1-Q3 {br(S1_B):.2%}（用户加权）")

    # ============ B. 用户分层（10）============
    B = "B"
    age = query("""SELECT CASE WHEN age<=25 THEN '25岁及以下' WHEN age<=30 THEN '26-30岁' WHEN age<=40 THEN '31-40岁'
        WHEN age<=50 THEN '41-50岁' ELSE '50岁以上' END seg, AVG(is_bad) b FROM loans GROUP BY 1 ORDER BY b DESC""")
    add(B, "按标准年龄段分层，各年龄段的坏账率是多少？哪个年龄段最高？",
        [rate(x) for x in age["b"]] + [has(age["seg"].iloc[0], age["seg"].iloc[0].replace("岁", ""))],
        "; ".join(f"{s}:{b:.2%}" for s, b in zip(age["seg"], age["b"])))
    lim = query("""SELECT CASE WHEN credit_limit<=30000 THEN '3万及以下' WHEN credit_limit<=50000 THEN '3-5万'
        WHEN credit_limit<=100000 THEN '5-10万' WHEN credit_limit<=200000 THEN '10-20万' ELSE '20万以上' END seg,
        AVG(is_bad) b FROM loans GROUP BY 1 ORDER BY b DESC""")
    add(B, "按授信额度段分层，坏账率随额度如何变化？",
        [rate(x) for x in lim["b"]] + [judge("需指出额度越低坏账率越高（3万及以下最高，20万以上最低）")],
        "; ".join(f"{s}:{b:.2%}" for s, b in zip(lim["seg"], lim["b"])))
    add(B, "男性用户和女性用户的坏账率分别是多少？", [rate(br("sex=1")), rate(br("sex=2"))])
    add(B, "不同学历用户的坏账率有什么差异？",
        [rate(br("education=1")), rate(br("education=2")), rate(br("education=3")),
         has("未定义", "未知", "编码")],
        "研究生/本科/高中；education 0/5/6 为未定义编码需单独说明")
    add(B, "已婚和单身用户，哪类坏账率更高？各是多少？",
        [rate(br("marriage=1")), rate(br("marriage=2")), has("已婚")])
    add(B, "近期逾期客群和近期未逾期客群的坏账率分别是多少？",
        [rate(br("pay_status_1>=1")), rate(br("pay_status_1<1"))])
    ct = query("SELECT city_tier, COUNT(*) n, SUM(is_bad) b FROM loans GROUP BY 1")
    from scipy.stats import chi2_contingency
    p_city = chi2_contingency(np.c_[ct["b"], ct["n"] - ct["b"]])[1]
    add(B, "不同城市等级之间的坏账率有显著差异吗？",
        [judge(f"需结论为差异不显著/差异很小（卡方检验 p={p_city:.2f}，各城市等级坏账率在 21.9%~22.7% 之间），"
               "不能说某个城市等级风险明显更高")],
        "; ".join(f"{c}:{b / m:.2%}" for c, b, m in zip(ct["city_tier"], ct["b"], ct["n"])))
    lv = query_view("SELECT risk_level, AVG(is_bad) b FROM v GROUP BY 1")
    add(B, "模型风险等级为高、中、低的用户，实际坏账率分别是多少？", [rate(x) for x in lv["b"]],
        "; ".join(f"{a}:{b:.2%}" for a, b in zip(lv["risk_level"], lv["b"])))
    w = "age BETWEEN 31 AND 40 AND credit_limit>100000 AND credit_limit<=200000"
    add(B, "31-40岁、授信额度10-20万的用户，坏账率是多少？", [rate(br(w)), num(n(w))])
    add(B, "循环信用客群的坏账率是多少？", [rate(br("pay_status_1=0"))], "口径：pay_status_1=0")

    # ============ C. 异动归因（12）============
    C = "C"
    s1_facts = (f"信息流广告 Q4 坏账率 {br(S1_T):.2%}（n={n(S1_T)}），Q1-Q3 {br(S1_B):.2%}（n={n(S1_B)}），"
                f"上升 {d * 100:.1f}pp；年轻低额度客群占比由 {n(S1_B + ' AND ' + YL) / n(S1_B):.1%} 升至 "
                f"{n(S1_T + ' AND ' + YL) / n(S1_T):.1%}；3万及以下额度占比由 "
                f"{n(S1_B + ' AND credit_limit<=30000') / n(S1_B):.1%} 升至 {n(S1_T + ' AND credit_limit<=30000') / n(S1_T):.1%}；"
                "按额度段拆解结构效应约占 94%")
    add(C, "信息流广告渠道最近三个月（2025年10-12月）坏账率突然上升，帮我分析一下原因。",
        [rate(br(S1_T)), judge("1) 给出坏账率上升的幅度（约 21% 升到约 30%，上升约 8pp）；"
               "2) 指出主要原因是客群结构变化：年轻和/或低额度客群占比大幅上升（结构效应为主），"
               "而不是同一客群自身风险变差；3) 有数据支撑（占比变化、分层坏账率或 SHAP/模型分）")], s1_facts)
    s2_rec_b = n(S2_B + " AND pay_status_1>=1") / n(S2_B)
    s2_rec_t = n(S2_T + " AND pay_status_1>=1") / n(S2_T)
    s2_facts = (f"合作方导流 6 月坏账率 {br(S2_T):.2%}（n={n(S2_T)}），其他月份 {br(S2_B):.2%}；"
                f"近期逾期客群占比由 {s2_rec_b:.1%} 升至 {s2_rec_t:.1%}，按近期逾期拆解结构效应 >100%；"
                "6 月放款量约为其他月份的 2 倍；SHAP 差异最大特征 pay_status_1、max_pay_status")
    add(C, "合作方导流渠道6月份的坏账率很高，是什么原因？",
        [rate(br(S2_T)), judge("1) 给出 6 月坏账率（约 35%）与其他月份（约 20%）的对比；"
               "2) 指出主要原因是近期已有逾期记录的客户（pay_status_1>=1 / 近期逾期客群）占比大幅上升；"
               "3) 判断为结构效应/客群质量问题")], s2_facts)
    add(C, "2025年有哪些渠道在哪些月份出现了坏账率的显著异动？",
        [judge("必须同时识别出两处坏账率显著上升：合作方导流 2025-06，以及信息流广告 2025-10~12（第四季度）。"
               "额外提到某些月份坏账率偏低/下降不扣分；但如果把其他渠道月份判定为坏账率显著上升，判为不通过")],
        "显著异动只有：合作方导流 2025-06（35.1%）、信息流广告 2025-10/11/12（28.5%~31.1%）")
    q4_all, q13_all = br("loan_month>='2025-10'"), br("loan_month<'2025-10'")
    add(C, "第四季度整体坏账率有没有上升？是全局问题还是个别渠道导致的？",
        [rate(q4_all), judge(f"1) Q4 整体坏账率 {q4_all:.1%} 相比前三季度 {q13_all:.1%} 小幅上升约 {(q4_all - q13_all) * 100:.1f}pp；"
               "2) 指出主要由信息流广告渠道导致，其他渠道没有同步上升，不是全局问题")],
        "Q4 各渠道：APP自然流量 20.1%、信息流广告 29.5%、合作方导流 19.2%、线下门店 23.1%")
    add(C, "信息流广告第四季度坏账率上升，属于结构效应还是风险效应？",
        [rate(br(S1_T)), judge("需判断为结构效应为主（客群结构变化，低额度/年轻客群占比上升），并给出拆解依据")], s1_facts)
    t, b_ = (query_view(f"SELECT risk_score FROM v WHERE {x}")["risk_score"].values for x in (S1_T, S1_B))
    psi1 = _psi(b_, t)
    add(C, "信息流广告渠道第四季度的客群，模型风险分有没有变化？客群有没有发生偏移？",
        [rate(float(t.mean())), rate(float(b_.mean())), judge(f"需指出平均风险分明显上升且 PSI≈{psi1:.2f}（>0.25，显著偏移）")],
        f"平均风险分 Q4 {t.mean():.4f}，Q1-Q3 {b_.mean():.4f}，PSI {psi1:.4f}")
    add(C, "用SHAP分析：信息流广告第四季度客群风险更高，主要是哪些特征驱动的？",
        [has("credit_limit", "授信额度", "额度"), has("max_pay_status", "pay_status_1", "逾期", "还款状态")],
        "SHAP 差值前三：credit_limit、max_pay_status、pay_status_1")
    add(C, "合作方导流6月客群相比该渠道其他月份，风险更高的主要驱动因素是什么？请用模型归因分析。",
        [has("pay_status_1", "近期还款状态", "最近一期还款", "近期逾期", "最近一个月"), has("max_pay_status", "历史最严重", "最严重")],
        "SHAP 差值前三：pay_status_1、max_pay_status、n_overdue_months")
    store = query("SELECT loan_month, COUNT(*) n, AVG(is_bad) b FROM loans WHERE channel='线下门店' GROUP BY 1 ORDER BY 1")
    add(C, "线下门店渠道今年的坏账率有没有出现异常？",
        [judge("需结论为没有显著异动/整体平稳（月度坏账率在约 19%~24% 之间正常波动）")],
        "; ".join(f"{m}:{b:.1%}(n={k})" for m, k, b in zip(store["loan_month"], store["n"], store["b"])))
    app10_where = "channel='APP自然流量' AND loan_month='2025-10'"
    app10 = br(app10_where)
    add(C, f"APP自然流量渠道10月份坏账率只有{app10 * 100:.1f}%，是不是这个渠道的风险变好了？",
        [judge("需指出与其他月份（约 20%~23%）相比差异不显著/属于正常波动，不能据此判断风险变好")],
        f"APP自然流量 10 月 {app10:.2%}（n={n(app10_where)}），与该渠道其他月份相比 p≈0.09，不显著")
    w = "city_tier='一线' AND channel='信息流广告' AND loan_month='2025-12'"
    add(C, "一线城市、信息流广告渠道、12月放款的用户坏账率是多少？这个数能说明问题吗？",
        [rate(br(w)), num(n(w)), judge("需提示样本量较小（不足 300），结论仅供参考/不宜单独下结论")],
        f"n={n(w)}，坏账率 {br(w):.2%}")
    vol = query("SELECT loan_month, COUNT(*) n FROM loans WHERE channel='合作方导流' GROUP BY 1")
    others = vol[vol.loan_month != "2025-06"]["n"].mean()
    add(C, "合作方导流渠道6月份的放款量有没有异常？",
        [num(n(S2_T)), judge(f"需指出 6 月放款量 {n(S2_T)} 明显高于其他月份（平均约 {others:.0f}），约为 2 倍")],
        f"6 月 {n(S2_T)}，其他月份平均 {others:.0f}")

    # ============ D. 模型相关（8）============
    D = "D"
    add(D, "风险模型在测试集上的AUC是多少？", [{"type": "number", "value": round(auc("TRUE"), 4), "kind": "auc"}])
    add(D, "风险模型最重要的3个特征是什么？",
        [has("pay_status_1"), has("max_pay_status")],
        "全局 mean|SHAP|：pay_status_1 0.368、max_pay_status 0.353、utilization_1 0.138、bill_amt_1 0.134")
    add(D, "模型在信息流广告渠道的AUC是多少？",
        [{"type": "number", "value": round(auc("channel='信息流广告'"), 4), "kind": "auc"}], "测试集口径")
    high = query_view("SELECT AVG(is_bad) b, AVG(risk_score) s FROM v WHERE risk_level='高'")
    add(D, "模型判为高风险的用户，实际坏账率是多少？", [rate(float(high["b"].iloc[0]))])
    ov = query_view("SELECT AVG(risk_score) s, AVG(is_bad) b FROM v")
    add(D, "模型预测的平均违约概率和实际坏账率是否一致？",
        [rate(float(ov["s"].iloc[0])), judge("需结论为两者基本一致/校准良好")],
        f"平均风险分 {ov['s'].iloc[0]:.4f}，实际坏账率 {ov['b'].iloc[0]:.4f}")
    t2, b2 = (query_view(f"SELECT risk_score FROM v WHERE {x}")["risk_score"].values for x in (S2_T, S2_B))
    add(D, "合作方导流6月的客群与该渠道其他月份相比，风险分PSI是多少？",
        [{"type": "number", "value": round(_psi(b2, t2), 4), "kind": "psi"}, has("显著")])
    u1 = query_view("SELECT risk_score FROM v WHERE user_id=1")["risk_score"].iloc[0]
    top = query("SELECT feature FROM shap_values WHERE user_id=1 ORDER BY shap_value DESC LIMIT 2")["feature"].tolist()
    add(D, "用户ID为1的客户，模型给出的风险分是多少？主要风险因素是什么？",
        [{"type": "number", "value": round(float(u1), 4), "kind": "score"}, has(top[0])],
        f"risk_score={u1}，SHAP 最高的特征：{top}")
    aucs = {c: auc(f"channel='{c}'") for c in ["APP自然流量", "信息流广告", "合作方导流", "线下门店"]}
    worst = min(aucs, key=aucs.get)
    add(D, "模型在哪个渠道上的区分度最差？", [has(worst)],
        "测试集 AUC：" + "; ".join(f"{c}:{a:.4f}" for c, a in aucs.items()))

    # ============ E. 数据质量与口径陷阱（8）============
    E = "E"
    add(E, f"学历编码为5的用户坏账率只有{br('education=5') * 100:.1f}%，是不是说明这类客群很优质？",
        [has("未定义"), judge("需明确指出 education=5 是数据字典中未定义的编码，不能据此得出客群优质的结论")])
    add(E, "婚姻状况编码为0的用户有多少？代表什么含义？", [num(n("marriage=0")), has("未定义")])
    add(E, "有多少用户最近一期的账单金额是负数？这是数据错误吗？",
        [num(n("bill_amt_1<0")), has("溢缴"), judge("需说明负账单表示溢缴款/多还款，不是数据错误")])
    add(E, "还款状态为0的用户算不算逾期？他们的坏账率是多少？",
        [has("不算", "不属于", "不是逾期", "未逾期", "并非逾期"), rate(br("pay_status_1=0"))],
        "pay_status=0 表示循环信用（只还最低还款额），不是逾期")
    monthly = query("SELECT AVG(is_bad) b FROM loans GROUP BY loan_month")["b"].mean()
    add(E, "2025年各月份坏账率的平均水平是多少？",
        [rate(br("TRUE"))], f"按口径应用户加权 {br('TRUE'):.4f}；月度简单平均 {monthly:.4f}")
    w = "city_tier='一线' AND channel='合作方导流' AND loan_month='2025-12' AND age>50"
    add(E, "12月一线城市合作方导流渠道、50岁以上用户的坏账率是多少？",
        [num(n(w)), judge("需提示样本量过小（不足 100），不应据此下结论")], f"n={n(w)}，坏账率 {br(w):.2%}")
    unk = "education NOT IN (1,2,3,4)"
    add(E, "学历未知的用户有多少人，坏账率是多少？", [num(n(unk)), rate(br(unk)), has("未定义")])
    add(E, "数据中有没有缺失值或异常值需要注意？",
        [judge("需提到：没有缺失值；education 存在未定义编码 0/5/6；marriage 存在未定义编码 0；"
               "账单金额存在负值（溢缴款）。至少覆盖其中 3 点，且不能把负账单说成数据错误")])

    assert len(Q) == 50, len(Q)
    return Q


if __name__ == "__main__":
    qs = build()
    OUT.write_text("\n".join(json.dumps(q, ensure_ascii=False) for q in qs) + "\n", encoding="utf-8")
    from collections import Counter
    print(f"wrote {len(qs)} questions -> {OUT.relative_to(ROOT)}", dict(Counter(q["category"] for q in qs)))
