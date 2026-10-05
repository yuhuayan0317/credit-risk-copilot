"""
运行评测：
    python eval/run_eval.py --system agent          # 完整 Agent（RAG + 工具 + 校验）
    python eval/run_eval.py --system agent_norag    # 消融：去掉知识库
    python eval/run_eval.py --system baseline       # 基础 Prompt：一次性生成代码
    python eval/run_eval.py --system agent --ids A01 C01 --workers 1   # 只跑指定题目

结果写入 eval/results/<system>.jsonl，支持断点续跑（已完成的题目会跳过）。
「可执行分析成功率」= 分析流程成功执行完成 且 所有检查项通过 的题目占比。
"""
import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))
from agent.agent import CreditRiskAgent  # noqa: E402
from agent.baseline import BaselinePrompt  # noqa: E402
from grader import grade  # noqa: E402

RESULTS = ROOT / "eval" / "results"
EXECUTED = {"ok", "step_limit"}


def make_system(name: str, client):
    if name == "agent":
        return CreditRiskAgent(use_rag=True, client=client)
    if name == "agent_norag":
        return CreditRiskAgent(use_rag=False, client=client)
    if name == "baseline":
        return BaselinePrompt(client=client)
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=["agent", "agent_norag", "baseline"])
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    questions = [json.loads(l) for l in (ROOT / "eval" / "questions.jsonl").read_text().splitlines() if l]
    if args.ids:
        questions = [q for q in questions if q["id"] in args.ids]
    if args.limit:
        questions = questions[:args.limit]

    RESULTS.mkdir(exist_ok=True)
    out_path = RESULTS / f"{args.system}.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(l)["id"] for l in out_path.read_text().splitlines() if l}
    todo = [q for q in questions if q["id"] not in done]
    print(f"[{args.system}] {len(todo)} 题待运行（已完成 {len(done)}）")

    client = anthropic.Anthropic(max_retries=5)
    lock = threading.Lock()

    def work(q):
        run = make_system(args.system, client).run(q["question"])
        executed = run.status in EXECUTED and bool(run.answer)
        g = grade(client, q, run.answer) if executed else {"passed": False, "checks": []}
        rec = {"id": q["id"], "category": q["category"], "question": q["question"],
               "executed": executed, "success": executed and g["passed"], "grade": g,
               **{k: v for k, v in run.to_dict().items() if k != "question"}}
        with lock, out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        return rec

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(work, q): q for q in todo}
        for fut in as_completed(futures):
            q = futures[fut]
            try:
                r = fut.result()
                mark = "✓" if r["success"] else ("✗" if r["executed"] else "⚠")
                print(f"{mark} {q['id']} {r['status']:<10} {r['elapsed']:6.1f}s  tools={r['n_tool_calls']:<3} {q['question'][:40]}")
            except Exception as e:  # noqa: BLE001
                print(f"! {q['id']} 运行异常：{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
