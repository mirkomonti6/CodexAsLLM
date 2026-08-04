import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from codex_llm_proxy import DEFAULT_MODEL
from codex_llm_proxy.subscription_client import _ChatGPTCodexResponsesProxy


class _FakeStream:
    def __init__(self, events):
        self._events = events

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def __iter__(self):
        return iter(self._events)


class _FakeResponses:
    def __init__(self):
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        response = SimpleNamespace(
            id="resp_gpt56_test",
            output=[],
            status="completed",
        )
        return _FakeStream(
            [SimpleNamespace(type="response.completed", response=response)]
        )


class _FakeOpenAI:
    last_instance = None

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.responses = _FakeResponses()
        _FakeOpenAI.last_instance = self


class GPT56SupportTests(unittest.TestCase):
    def test_default_model_is_gpt56(self):
        self.assertEqual(DEFAULT_MODEL, "gpt-5.6-luna")

    def test_gpt56_and_reasoning_are_forwarded_to_responses(self):
        fake_openai = types.SimpleNamespace(OpenAI=_FakeOpenAI)
        token = (
            "header.eyJodHRwczovL2FwaS5vcGVuYWkuY29tL2F1dGgiOnsi"
            "Y2hhdGdwdF9hY2NvdW50X2lkIjoiYWNjdC10ZXN0In19.signature"
        )
        proxy = _ChatGPTCodexResponsesProxy(lambda: token)

        with patch.dict(sys.modules, {"openai": fake_openai}):
            response = proxy.create(
                model="gpt-5.6-luna",
                instructions="Be concise.",
                reasoning={"effort": "medium"},
                input=[{"role": "user", "content": "Say hello."}],
            )

        sent = _FakeOpenAI.last_instance.responses.last_kwargs
        self.assertEqual(response.id, "resp_gpt56_test")
        self.assertEqual(sent["model"], "gpt-5.6-luna")
        self.assertEqual(sent["reasoning"], {"effort": "medium"})
        self.assertTrue(sent["stream"])
        self.assertFalse(sent["store"])


if __name__ == "__main__":
    unittest.main()
