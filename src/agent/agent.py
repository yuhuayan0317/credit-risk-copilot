"""
信贷风险分析 Agent：基于 Claude Tool Calling 的手写 agentic loop。

没有用 SDK 的 tool_runner，原因是需要在循环中插入自定义逻辑：
  - 记录每一步的工具调用、耗时、校验提示（给前端展示和评测统计）
  - 工具调用次数上限，到达上限后要求模型直接出报告
  - 最终报告的数字溯源校验，不通过时要求模型修正一次
"""
import time
from dataclasses import dataclass, field

import anthropic

from agent.grounding import ungrounded_numbers
from agent.prompts import (AGENT_SYSTEM, AGENT_SYSTEM_NO_RAG, GROUNDING_FEEDBACK,
                           STEP_LIMIT_NOTE)
from config import LLM_EFFORT, LLM_MODEL
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
    steps: list[Step] = field(default_factory=list)
    n_tool_calls: int = 0
    n_tool_errors: int = 0
    n_llm_calls: int = 0
    elapsed: float = 0.0
    usage: dict = field(default_factory=lambda: {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0})
    ungrounded: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {**{k: v for k, v in self.__dict__.items() if k != "steps"},
                "steps": [s.__dict__ for s in self.steps]}


class CreditRiskAgent:
    def __init__(self, use_rag: bool = True, model: str = LLM_MODEL, effort: str = LLM_EFFORT,
                 client: anthropic.Anthropic | None = None, on_step=None):
        self.client = client or anthropic.Anthropic()
        self.model = model
        self.effort = effort
        self.system = AGENT_SYSTEM if use_rag else AGENT_SYSTEM_NO_RAG
        self.tools = TOOLS if use_rag else [t for t in TOOLS if t["name"] != "search_knowledge"]
        self.on_step = on_step  # 回调：前端实时展示每一步

    def _log(self, run: AgentRun, step: Step) -> None:
        run.steps.append(step)
        if self.on_step:
            self.on_step(step)

    def _call(self, run: AgentRun, messages: list, allow_tools: bool = True):
        resp = self.client.beta.messages.create(
            model=self.model,
            max_tokens=16000,
            system=self.system,
            tools=self.tools,
            tool_choice={"type": "auto" if allow_tools else "none"},
            messages=messages,
            thinking={"type": "adaptive", "display": "summarized"},
            output_config={"effort": self.effort},
            cache_control={"type": "ephemeral"},
            # 安全分类器误拒时由服务端自动切换到推荐的备用模型
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        run.n_llm_calls += 1
        u = resp.usage
        run.usage["input"] += u.input_tokens or 0
        run.usage["output"] += u.output_tokens or 0
        run.usage["cache_read"] += u.cache_read_input_tokens or 0
        run.usage["cache_write"] += u.cache_creation_input_tokens or 0
        return resp

    def run(self, question: str) -> AgentRun:
        run = AgentRun(question=question)
        t0 = time.time()
        messages: list = [{"role": "user", "content": question}]
        tool_texts: list[str] = []
        rounds = grounding_retries = 0
        allow_tools = True

        try:
            while True:
                resp = self._call(run, messages, allow_tools)
                # 历史只追加不修改：thinking 块需要原样回传
                messages.append({"role": "assistant", "content": resp.content})

                for block in resp.content:
                    if block.type == "thinking" and block.thinking:
                        self._log(run, Step("thinking", block.thinking))
                    elif block.type == "text" and block.text.strip():
                        self._log(run, Step("text", block.text))

                if resp.stop_reason == "refusal":
                    run.status = "refusal"
                    break
                if resp.stop_reason == "max_tokens":
                    run.status = "max_tokens"
                    break

                tool_uses = [b for b in resp.content if b.type == "tool_use"]
                if resp.stop_reason == "tool_use" and tool_uses:
                    rounds += 1
                    results = []
                    for tu in tool_uses:
                        self._log(run, Step("tool_call", "", tool=tu.name, args=tu.input))
                        ts = time.time()
                        r = execute_tool(tu.name, tu.input)
                        rendered = r.render()
                        tool_texts.append(rendered)
                        run.n_tool_calls += 1
                        run.n_tool_errors += int(r.is_error)
                        self._log(run, Step("tool_result", rendered, tool=tu.name, is_error=r.is_error,
                                            warnings=r.warnings, elapsed=time.time() - ts))
                        results.append({"type": "tool_result", "tool_use_id": tu.id,
                                        "content": rendered, "is_error": r.is_error})
                    if rounds >= MAX_TOOL_ROUNDS:
                        results.append({"type": "text", "text": STEP_LIMIT_NOTE})
                        allow_tools = False
                        run.status = "step_limit"
                    messages.append({"role": "user", "content": results})
                    continue

                # 模型给出了最终回答：做数字溯源校验
                answer = "\n".join(b.text for b in resp.content if b.type == "text").strip()
                missing = ungrounded_numbers(answer, tool_texts)
                self._log(run, Step("check", "数字溯源校验通过" if not missing
                                    else f"未找到依据的数字：{', '.join(missing)}", warnings=missing))
                if missing and grounding_retries < MAX_GROUNDING_RETRIES and allow_tools:
                    grounding_retries += 1
                    messages.append({"role": "user", "content": GROUNDING_FEEDBACK.format(
                        numbers="、".join(missing))})
                    continue
                run.answer = answer
                run.ungrounded = missing
                break
        except anthropic.APIError as e:
            run.status = "api_error"
            run.answer = f"API 调用失败：{e}"

        run.elapsed = time.time() - t0
        return run


if __name__ == "__main__":
    import sys

    q = sys.argv[1] if len(sys.argv) > 1 else "信息流广告渠道最近三个月坏账率突然上升，帮我分析一下原因。"

    def show(step: Step):
        if step.kind == "tool_call":
            print(f"\n>>> 调用 {step.tool}: {step.args}")
        elif step.kind == "tool_result":
            print(f"<<< {step.content[:600]}")
        elif step.kind == "check":
            print(f"\n[校验] {step.content}")

    result = CreditRiskAgent(on_step=show).run(q)
    print("\n" + "=" * 60 + f"\n{result.answer}\n" + "=" * 60)
    print(f"状态 {result.status}｜工具调用 {result.n_tool_calls} 次｜LLM 调用 {result.n_llm_calls} 次｜"
          f"耗时 {result.elapsed:.0f}s｜tokens {result.usage}")
