"""对 OpenAI 兼容聊天补全接口的一层轻量封装。

项目中的所有 LLM 调用最终都会走到 :class:`Agent`。
`Workflow` 会创建 manager agent 来生成或改写 criterion，
`Evaluator` 会创建大量短生命周期的 worker agent 来对样本做判断。
"""

import json
import os
import random
import time
from time import sleep
from typing import Any

from openai import OpenAI, RateLimitError

RATE_LIMIT_RETRY_DELAY = 60
RATE_LIMIT_RETRY_ATTEMPTS = 50
FORBIDDEN_RETRY_BASE_DELAY = 1
FORBIDDEN_RETRY_MAX_DELAY = 60
WORKFLOW_AGENT_LOGFILE = os.getenv("WORKFLOW_AGENT_LOGFILE", None)


class Agent:
    """带有重试、日志和历史分叉能力的有状态聊天代理。"""

    def __init__(
        self,
        system: str | None = None,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
        api_keys: str | list[str] | None = None,
        request_kwargs: dict[str, Any] = None,
        tensor_parallel_size: int | None = None,
    ):
        # `system` 是可选的，因为大多数 workflow prompt 都是在外部手动拼接的。
        self.system = system
        if self.system is None:
            self.history = []
        else:
            self.history = [{"role": "system", "content": self.system}]
        self.model = model
        self.base_url = base_url

        if api_keys is not None:
            if isinstance(api_keys, str):
                api_keys = [api_keys]
        else:
            api_keys = [os.getenv("OPENAI_API_KEY", "EMPTY")]
        self.api_keys = api_keys

        self.request_kwargs = {}
        if request_kwargs is not None:
            self.request_kwargs.update(request_kwargs)

        # 每个 agent 持有自己的 client，这样请求配置始终局限在当前实例内。
        self.client = OpenAI(
            api_key=random.choice(self.api_keys), base_url=self.base_url
        )

    @staticmethod
    def _extract_status_code(error: Exception) -> int | None:
        """Try best-effort extraction of HTTP status code from SDK exceptions."""
        status_code = getattr(error, "status_code", None)
        if status_code is not None:
            return status_code
        response = getattr(error, "response", None)
        if response is not None:
            return getattr(response, "status_code", None)
        return None

    def _retry_with_new_client(self):
        """Rotate API key before retry to reduce shared-key throttling impact."""
        self.client = OpenAI(
            api_key=random.choice(self.api_keys), base_url=self.base_url
        )

    def chat_completion_openai(
        self, messages, stream: bool = True, ttl: int = RATE_LIMIT_RETRY_ATTEMPTS
    ):
        """将当前消息列表发送到 OpenAI 兼容后端。"""
        response = ""
        if ttl >= 0:
            try:
                if stream:
                    chunk_stream = self.client.chat.completions.create(
                        stream=True,
                        model=self.model,
                        messages=messages,
                        **self.request_kwargs,
                    )
                    for chunk in chunk_stream:  # pylint: disable=E1133:not-an-iterable
                        if (
                            not chunk.choices[0].finish_reason
                            and chunk.choices[0].delta.content
                        ):
                            print(chunk.choices[0].delta.content, end="", flush=True)
                            response += chunk.choices[0].delta.content
                    print()
                else:
                    response = (
                        self.client.chat.completions.create(
                            model=self.model, messages=messages, **self.request_kwargs
                        )
                        .choices[0]
                        .message.content
                    )
            except RateLimitError as e:
                if ttl > 0:
                    print(
                        f"Rate limit exceeded, waiting for {RATE_LIMIT_RETRY_DELAY} seconds and retrying... {ttl=}",
                        e,
                    )
                    # 用递归重试，调用方可以保持简单的同步控制流。
                    sleep(RATE_LIMIT_RETRY_DELAY)
                    self._retry_with_new_client()
                    return self.chat_completion_openai(
                        messages, stream=stream, ttl=ttl - 1
                    )
                raise
            except Exception as e:  # pylint: disable=W0718:broad-exception-caught
                # status_code = self._extract_status_code(e)
                # if status_code == 403 and ttl > 0:
                if ttl > 0:
                    retry_index = RATE_LIMIT_RETRY_ATTEMPTS - ttl
                    backoff = min(
                        FORBIDDEN_RETRY_MAX_DELAY,
                        FORBIDDEN_RETRY_BASE_DELAY * (2**retry_index),
                    )
                    # 增加少量随机抖动，降低并发重试雪崩。
                    delay = backoff + random.uniform(0, 1)
                    print(
                        f"other received, waiting {delay:.2f}s before retry... {ttl=}",
                        e,
                    )
                    sleep(delay)
                    self._retry_with_new_client()
                    return self.chat_completion_openai(
                        messages, stream=stream, ttl=ttl - 1
                    )
                raise
        return response

    def chat_completion(
        self, messages, stream: bool = True, ttl: int = RATE_LIMIT_RETRY_ATTEMPTS
    ):
        """为未来接入不同后端保留的兼容分发层。"""
        return self.chat_completion_openai(messages, stream=stream, ttl=ttl)

    @staticmethod
    def _sanitize_for_log(value):
        """复制日志数据，并移除内嵌图片的 base64 内容。"""
        if isinstance(value, str):
            if value.startswith("data:image/"):
                return "<image omitted>"
            return value
        if isinstance(value, dict):
            return {key: Agent._sanitize_for_log(item) for key, item in value.items()}
        if isinstance(value, list):
            return [Agent._sanitize_for_log(item) for item in value]
        if isinstance(value, tuple):
            return tuple(Agent._sanitize_for_log(item) for item in value)
        return value

    def __call__(self, prompt, stream: bool = True) -> str | None:
        """追加一轮用户输入，调用模型，并保存 assistant 回复。"""
        self.history.append({"role": "user", "content": prompt})
        try:
            response = self.chat_completion(self.history, stream=stream)
            assert response is not None
        except Exception as e:  # pylint: disable=W0718:broad-exception-caught
            self.history.pop()
            print(e)
            return None
        self.history.append({"role": "assistant", "content": response})
        if WORKFLOW_AGENT_LOGFILE:
            # 可选的原始日志有助于排查 prompt 质量和运行过程。
            # 多模态请求中的 base64 图片只在发送给模型时使用；写日志时替换为
            # 占位符，避免 prompt 和 history 重复记录图片导致日志文件过大。
            log_prompt = self._sanitize_for_log(prompt)
            log_history = self._sanitize_for_log(self.history)
            # Windows 的默认编码通常是 GBK，而模型回复可能包含 emoji 或其他
            # Unicode 字符，因此日志文件必须显式使用 UTF-8 编码。
            with open(WORKFLOW_AGENT_LOGFILE, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "time": time.time(),
                            "model": self.model,
                            "prompt": log_prompt,
                            "response": response,
                            "history": log_history,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        return response

    def get_last_reply(self):
        """如果最后一条消息来自 assistant，就返回其内容。"""
        if self.history[-1]["role"] == "assistant":
            return self.history[-1]["content"]
        return None

    def forget_last_turn(self):
        """删除最近一轮用户输入及其后续 assistant 回复。"""
        while self.history[-1]["role"] != "user":
            self.history.pop()
        if self.history[-1]["role"] == "user":
            self.history.pop()

    def fork(self) -> "Agent":
        """复制当前 agent 状态，让新分支在独立上下文中继续对话。"""
        forked = Agent(
            system=None,
            model=self.model,
            base_url=self.base_url,
            api_keys=self.api_keys,
            request_kwargs=self.request_kwargs,
        )
        for turn in self.history:
            forked.history.append(turn)
        return forked
