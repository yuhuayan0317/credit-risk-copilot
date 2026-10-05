"""
信贷风险分析 Agent：基于 LLM Tool Calling 的手写 agentic loop，模型后端可切换（Claude / 智谱 GLM）。

没有用 SDK 自带的 tool runner，原因是需要在循环中插入自定义逻辑：
  - 记录每一步的工具调用、耗时、校验提示（给前端展示和评测统计）
  - 工具调用轮数上限，到达上限后要求模型直接出报告
  - 最终报告的数字溯源校验，不通过时要求模型修正一次
"""
import time
from dataclasses import dataclass, field

from agent.grounding import ungrounded_numbers
from agent.llm import make_backend
from agent.prompts import (AGENT_SYSTEM, AGENT_SYSTEM_NO_RAG, GROUNDING_FEEDBACK,
                           STEP_LIMIT_NOTE)
from tools.registry import TOOLS, execute_tool

MAX_TOOL_ROUNDS = 15
MAX_GROUNDING_RETRIES = 1


@dataclass
class Step:
    kind: str                  # thinking / text / tool_call / tool_result / check
    content: str
    tool: str | None = None
    args: dict | None = None
    is_error: bool = False
    warnings: list[str] = field(default_factory=list)
    elapsed: float = 0.0


@dataclass
class AgentRun:
    question: str
    answer: str = ""
    status: str = "ok"         # ok / step_limit / refusal / max_tokens / api_error
    model: str = ""
    steps: list[Step] = field(default_factory=list)
    n_tool_calls: int = 0
    n_tool_errors: int = 0
    n_llm_calls: int = 0
    elapsed: float = 0.0
    usage: dict = field(default_factory=lambda: {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0})
    ungrounded: list[str] = field(default_factory=list)

    def add_usage(self, u: dict) -> None:
        for k, v in u.items():
            self.usage[k] = self.usage.get(k, 0) + v

    def to_dict(self) -> dict:
        return {**{k: v for k, v in self.__dict__.items() if k != "steps"},
                "steps": [s.__dict__ for s in self.steps]}


class CreditRiskAgent:
    def __init__(self, use_rag: bool = True, backend=None, on_step=None):
        self.llm = backend or make_backend()
        self.system = AGENT_SYSTEM if use_rag else AGENT_SYSTEM_NO_RAG
        self.tools = TOOLS if use_rag else [t for t in TOOLS if t["name"] != "search_knowledge"]
        self.on_step = on_step  # 回调：前端实时展示每一步

    def _log(self, run: AgentRun, step: Step) -> None:
        run.steps.append(step)
        if self.on_step:
            self.on_step(step)

    def run(self, question: str) -> AgentRun:
        run = AgentRun(question=question, model=getattr(self.llm, "model", ""))
        t0 = time.time()
        messages = self.llm.new_messages(self.system, question)
        tool_texts: list[str] = []
        rounds = grounding_retries = 0
        allow_tools = True

        try:
            while True:
                turn = self.llm.call(messages, self.tools, allow_tools)
                run.n_llm_calls += 1
                run.add_usage(turn.usage)
                self.llm.add_assistant(messages, turn)
                if turn.thinking.strip():
                    self._log(run, Step("thinking", turn.thinking))
                if turn.text.strip() and turn.stop == "tool_use":
                    self._log(run, Step("text", turn.text))

                if turn.stop in ("refusal", "max_tokens"):
                    run.status = turn.stop
                    run.answer = turn.text.strip()
                    break

                if turn.stop == "tool_use":
                    rounds += 1
                    results = []
                    for tc in turn.tool_calls:
                        self._log(run, Step("tool_call", "", tool=tc.name, args=tc.input))
                        ts = time.time()
                        if tc.parse_error:
                            rendered, is_error, warnings = tc.parse_error + "，请重新调用。", True, []
                        else:
                            r = execute_tool(tc.name, tc.input)
                            rendered, is_error, warnings = r.render(), r.is_error, r.warnings
                        tool_texts.append(rendered)
                        run.n_tool_calls += 1
                        run.n_tool_errors += int(is_error)
                        self._log(run, Step("tool_result", rendered, tool=tc.name, is_error=is_error,
                                            warnings=warnings, elapsed=time.time() - ts))
                        results.append((tc.id, rendered, is_error))
                    note = None
                    if rounds >= MAX_TOOL_ROUNDS:
                        note, allow_tools, run.status = STEP_LIMIT_NOTE, False, "step_limit"
                    self.llm.add_tool_results(messages, results, note)
                    continue

                # 模型给出了最终回答：做数字溯源校验
                answer = turn.text.strip()
                missing = ungrounded_numbers(answer, tool_texts)
                self._log(run, Step("check", "数字溯源校验通过" if not missing
                                    else f"未找到依据的数字：{', '.join(missing)}", warnings=missing))
                if missing and grounding_retries < MAX_GROUNDING_RETRIES and allow_tools:
                    grounding_retries += 1
                    self.llm.add_user(messages, GROUNDING_FEEDBACK.format(numbers="、".join(missing)))
                    continue
                run.answer = answer
                run.ungrounded = missing
                break
        except Exception as e:  # noqa: BLE001 - 两家 SDK 的异常类型不同，统一记录为 API 错误
            run.status = "api_error"
            run.answer = f"LLM 调用失败：{type(e).__name__}: {e}"

        run.elapsed = time.time() - t0
        return run


if __name__ == "__main__":
    import sys

    q = sys.argv[1] if len(sys.argv) > 1 else "信息流广告渠道最近三个月坏账率突然上升，帮我分析一下原因。"

    def show(step: Step):
        if step.kind == "thinking":
            print(f"\n💭 {step.content[:300]}")
        elif step.kind == "tool_call":
            print(f"\n>>> 调用 {step.tool}: {step.args}")
        elif step.kind == "tool_result":
            print(f"<<< {step.content[:600]}")
        elif step.kind == "check":
            print(f"\n[校验] {step.content}")

    result = CreditRiskAgent(on_step=show).run(q)
    print("\n" + "=" * 60 + f"\n{result.answer}\n" + "=" * 60)
    print(f"模型 {result.model}｜状态 {result.status}｜工具调用 {result.n_tool_calls} 次｜LLM 调用 "
          f"{result.n_llm_calls} 次｜耗时 {result.elapsed:.0f}s｜tokens {result.usage}")
