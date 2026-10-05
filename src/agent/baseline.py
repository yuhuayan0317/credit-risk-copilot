"""
基础 Prompt 方案（对照组）：把表结构写进 Prompt，让模型一次性生成分析代码，执行一次，再根据输出写结论。
没有知识库、没有多轮工具调用、没有校验反馈、代码报错也不重试。
"""
import re
import time

import anthropic

from agent.agent import AgentRun, Step
from agent.prompts import BASELINE_ANSWER_SYSTEM, BASELINE_CODE_SYSTEM
from config import LLM_EFFORT, LLM_MODEL
from tools.python_runner import UnsafeCodeError, run_python

_CODE_BLOCK = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)


class BaselinePrompt:
    def __init__(self, model: str = LLM_MODEL, effort: str = LLM_EFFORT,
                 client: anthropic.Anthropic | None = None):
        self.client = client or anthropic.Anthropic()
        self.model = model
        self.effort = effort

    def _ask(self, run: AgentRun, system: str, content: str) -> str:
        resp = self.client.beta.messages.create(
            model=self.model, max_tokens=16000, system=system,
            messages=[{"role": "user", "content": content}],
            output_config={"effort": self.effort},
            betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        )
        run.n_llm_calls += 1
        run.usage["input"] += resp.usage.input_tokens or 0
        run.usage["output"] += resp.usage.output_tokens or 0
        if resp.stop_reason == "refusal":
            run.status = "refusal"
        return "\n".join(b.text for b in resp.content if b.type == "text")

    def run(self, question: str) -> AgentRun:
        run = AgentRun(question=question)
        t0 = time.time()
        try:
            reply = self._ask(run, BASELINE_CODE_SYSTEM, question)
            m = _CODE_BLOCK.search(reply)
            code = m.group(1) if m else reply
            run.steps.append(Step("tool_call", "", tool="run_python", args={"code": code}))
            try:
                ok, output = run_python(code)
            except UnsafeCodeError as e:
                ok, output = False, f"安全检查未通过：{e}"
            run.n_tool_calls = 1
            run.n_tool_errors = int(not ok)
            run.steps.append(Step("tool_result", output, tool="run_python", is_error=not ok))
            if not ok:
                run.status = "code_error"
                run.answer = f"分析代码执行失败：\n{output}"
            else:
                run.answer = self._ask(run, BASELINE_ANSWER_SYSTEM,
                                       f"业务问题：{question}\n\n分析代码：\n```python\n{code}\n```\n\n运行结果：\n{output}")
        except anthropic.APIError as e:
            run.status = "api_error"
            run.answer = f"API 调用失败：{e}"
        run.elapsed = time.time() - t0
        return run
