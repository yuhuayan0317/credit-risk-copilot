"""
LLM 适配层：Agent 主循环只依赖这里定义的统一接口，可以在 Claude 和智谱 GLM 之间切换。

两家的消息格式不同（Claude 是 content blocks + tool_result；OpenAI 兼容接口是 tool_calls + role=tool），
所以「历史消息怎么追加」也由各自的后端负责，主循环只处理统一的 Turn。
"""
import json
import time
from dataclasses import dataclass, field

from config import LLM_EFFORT, LLM_MODEL, LLM_PROVIDER, ZHIPU_API_KEY, ZHIPU_BASE_URL


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict
    parse_error: str | None = None   # 模型给出的参数不是合法 JSON 时记录原因


@dataclass
class Turn:
    text: str = ""
    thinking: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop: str = "end"                # end / tool_use / max_tokens / refusal
    usage: dict = field(default_factory=dict)
    raw: object = None               # 原始响应，追加历史时使用


class AnthropicBackend:
    provider = "anthropic"

    def __init__(self, model: str = LLM_MODEL, effort: str = LLM_EFFORT, client=None):
        import anthropic
        self.client = client or anthropic.Anthropic(max_retries=5)
        self.model, self.effort = model, effort

    def new_messages(self, system: str, question: str) -> list:
        self.system = system
        return [{"role": "user", "content": question}]

    def call(self, messages: list, tools: list, allow_tools: bool = True) -> Turn:
        resp = self.client.beta.messages.create(
            model=self.model, max_tokens=16000, system=self.system, tools=tools,
            tool_choice={"type": "auto" if allow_tools else "none"}, messages=messages,
            thinking={"type": "adaptive", "display": "summarized"},
            output_config={"effort": self.effort}, cache_control={"type": "ephemeral"},
            # 安全分类器误拒时由服务端自动切换到推荐的备用模型
            betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        )
        u = resp.usage
        turn = Turn(raw=resp, usage={"input": u.input_tokens or 0, "output": u.output_tokens or 0,
                                     "cache_read": u.cache_read_input_tokens or 0,
                                     "cache_write": u.cache_creation_input_tokens or 0})
        for b in resp.content:
            if b.type == "thinking":
                turn.thinking += b.thinking or ""
            elif b.type == "text":
                turn.text += b.text
            elif b.type == "tool_use":
                turn.tool_calls.append(ToolCall(b.id, b.name, b.input))
        turn.stop = {"tool_use": "tool_use", "max_tokens": "max_tokens", "refusal": "refusal"}.get(
            resp.stop_reason, "end")
        return turn

    def add_assistant(self, messages: list, turn: Turn) -> None:
        # 历史只追加不修改：thinking 块需要原样回传
        messages.append({"role": "assistant", "content": turn.raw.content})

    def add_tool_results(self, messages: list, results: list[tuple[str, str, bool]], note: str | None) -> None:
        content = [{"type": "tool_result", "tool_use_id": i, "content": c, "is_error": e} for i, c, e in results]
        if note:
            content.append({"type": "text", "text": note})
        messages.append({"role": "user", "content": content})

    def add_user(self, messages: list, text: str) -> None:
        messages.append({"role": "user", "content": text})

    def complete(self, system: str, user: str, json_mode: bool = False) -> tuple[str, dict]:
        resp = self.client.beta.messages.create(
            model=self.model, max_tokens=16000, system=system,
            messages=[{"role": "user", "content": user}], output_config={"effort": self.effort},
            betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        )
        text = "\n".join(b.text for b in resp.content if b.type == "text")
        return text, {"input": resp.usage.input_tokens or 0, "output": resp.usage.output_tokens or 0}


class OpenAICompatBackend:
    """智谱 GLM 等 OpenAI 兼容接口。"""
    provider = "zhipu"

    def __init__(self, model: str = LLM_MODEL, base_url: str = ZHIPU_BASE_URL,
                 api_key: str = ZHIPU_API_KEY, client=None):
        from openai import OpenAI
        self.client = client or OpenAI(base_url=base_url, api_key=api_key, max_retries=1, timeout=300)
        self.model = model

    def _create(self, **kwargs):
        """免费档经常限流（429）或排队超时，这里用更长的退避时间重试，最多等待约 10 分钟。"""
        from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
        delay = 5
        for attempt in range(12):
            try:
                return self.client.chat.completions.create(model=self.model, **kwargs)
            except (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError):
                if attempt == 11:
                    raise
                time.sleep(delay)
                delay = min(delay * 1.6, 60)

    @staticmethod
    def convert_tools(tools: list) -> list:
        return [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                                  "parameters": t["input_schema"]}} for t in tools]

    def new_messages(self, system: str, question: str) -> list:
        return [{"role": "system", "content": system}, {"role": "user", "content": question}]

    def call(self, messages: list, tools: list, allow_tools: bool = True) -> Turn:
        kwargs = {}
        if allow_tools:
            kwargs = {"tools": self.convert_tools(tools), "tool_choice": "auto"}
        resp = self._create(messages=messages, max_tokens=16000, temperature=0.2, **kwargs)
        choice = resp.choices[0]
        msg = choice.message
        u = resp.usage
        turn = Turn(raw=msg, text=msg.content or "",
                    thinking=getattr(msg, "reasoning_content", None) or "",
                    usage={"input": getattr(u, "prompt_tokens", 0) or 0,
                           "output": getattr(u, "completion_tokens", 0) or 0})
        for tc in msg.tool_calls or []:
            try:
                args, err = json.loads(tc.function.arguments or "{}"), None
            except json.JSONDecodeError as e:
                args, err = {}, f"工具参数不是合法 JSON：{e}"
            turn.tool_calls.append(ToolCall(tc.id, tc.function.name, args, err))
        if turn.tool_calls:
            turn.stop = "tool_use"
        elif choice.finish_reason == "length":
            turn.stop = "max_tokens"
        elif choice.finish_reason == "sensitive":   # 智谱的内容安全拦截
            turn.stop = "refusal"
        return turn

    def add_assistant(self, messages: list, turn: Turn) -> None:
        msg = {"role": "assistant", "content": turn.text or ""}
        if turn.tool_calls:
            msg["tool_calls"] = [{"id": t.id, "type": "function",
                                  "function": {"name": t.name, "arguments": json.dumps(t.input, ensure_ascii=False)}}
                                 for t in turn.tool_calls]
        messages.append(msg)

    def add_tool_results(self, messages: list, results: list[tuple[str, str, bool]], note: str | None) -> None:
        for i, c, _ in results:
            messages.append({"role": "tool", "tool_call_id": i, "content": c})
        if note:
            messages.append({"role": "user", "content": note})

    def add_user(self, messages: list, text: str) -> None:
        messages.append({"role": "user", "content": text})

    def complete(self, system: str, user: str, json_mode: bool = False) -> tuple[str, dict]:
        extra = {"response_format": {"type": "json_object"}} if json_mode else {}
        resp = self._create(max_tokens=16000, temperature=0.2,
                            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                            **extra)
        u = resp.usage
        return resp.choices[0].message.content or "", {"input": getattr(u, "prompt_tokens", 0) or 0,
                                                        "output": getattr(u, "completion_tokens", 0) or 0}


def make_backend(provider: str = LLM_PROVIDER, model: str | None = None, **kw):
    if provider == "anthropic":
        return AnthropicBackend(model=model or LLM_MODEL, **kw)
    if provider == "zhipu":
        return OpenAICompatBackend(model=model or LLM_MODEL, **kw)
    raise ValueError(f"未知的 LLM_PROVIDER：{provider}")
