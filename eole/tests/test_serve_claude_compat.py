"""Unit tests for Claude-compatible serve endpoint."""

import asyncio
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import AsyncMock, patch

import yaml

MOCK_SCORE = -0.1


async def _drain_async_generator(gen):
    async for _ in gen:
        pass


def _install_stub_modules():
    """Install lightweight module stubs required to import serve.py."""
    previous_modules = {}

    def _stub(name, module):
        previous_modules[name] = sys.modules.get(name)
        sys.modules[name] = module

    # --- pydantic ---
    pydantic_mod = types.ModuleType("pydantic")

    class BaseModel:
        model_config = {}

        def __init__(self, **kwargs):
            annotations = {}
            for cls in reversed(self.__class__.__mro__):
                annotations.update(getattr(cls, "__annotations__", {}))
            for field_name in annotations:
                if field_name in kwargs:
                    setattr(self, field_name, kwargs[field_name])
                elif hasattr(self.__class__, field_name):
                    setattr(self, field_name, getattr(self.__class__, field_name))
                else:
                    setattr(self, field_name, None)
            for k, v in kwargs.items():
                if k not in annotations:
                    setattr(self, k, v)

        def model_dump(self):
            def _convert(value):
                if isinstance(value, BaseModel):
                    return value.model_dump()
                if isinstance(value, list):
                    return [_convert(v) for v in value]
                if isinstance(value, dict):
                    return {k: _convert(v) for k, v in value.items()}
                return value

            return {k: _convert(v) for k, v in self.__dict__.items()}

        def model_dump_json(self):
            return json.dumps(self.model_dump())

    def ConfigDict(**kwargs):
        return kwargs

    def Field(default=None, **kwargs):
        return default

    def model_validator(*args, **kwargs):
        def _decorator(fn):
            return fn

        return _decorator

    pydantic_mod.BaseModel = BaseModel
    pydantic_mod.ConfigDict = ConfigDict
    pydantic_mod.Field = Field
    pydantic_mod.model_validator = model_validator
    _stub("pydantic", pydantic_mod)

    # --- fastapi ---
    fastapi_mod = types.ModuleType("fastapi")

    class FastAPI:
        def __init__(self, **kwargs):
            self.routes = {}

        def get(self, path, **kwargs):
            def _decorator(fn):
                self.routes[("GET", path)] = fn
                return fn

            return _decorator

        def post(self, path, **kwargs):
            def _decorator(fn):
                self.routes[("POST", path)] = fn
                return fn

            return _decorator

    class Request:
        def __init__(self, url="http://localhost/"):
            self.url = url

    def Body(*args, **kwargs):
        return None

    fastapi_mod.FastAPI = FastAPI
    fastapi_mod.Request = Request
    fastapi_mod.Body = Body
    _stub("fastapi", fastapi_mod)

    fastapi_responses_mod = types.ModuleType("fastapi.responses")

    class HTMLResponse:
        def __init__(self, content, status_code=200):
            self.content = content
            self.status_code = status_code

    class StreamingResponse:
        def __init__(self, content, media_type=None, headers=None):
            self.content = content
            self.media_type = media_type
            self.headers = headers or {}

    class JSONResponse:
        def __init__(self, status_code=200, content=None):
            self.status_code = status_code
            self.content = content or {}

    fastapi_responses_mod.HTMLResponse = HTMLResponse
    fastapi_responses_mod.StreamingResponse = StreamingResponse
    fastapi_responses_mod.JSONResponse = JSONResponse
    _stub("fastapi.responses", fastapi_responses_mod)

    # --- runtime deps used by serve.py ---
    torch_mod = types.ModuleType("torch")
    torch_mod.cuda = types.SimpleNamespace(empty_cache=lambda: None)
    _stub("torch", torch_mod)

    uvicorn_mod = types.ModuleType("uvicorn")
    uvicorn_mod.run = lambda **kwargs: None
    _stub("uvicorn", uvicorn_mod)

    eole_mod = types.ModuleType("eole")
    eole_mod.__version__ = "test"
    _stub("eole", eole_mod)

    inference_engine_mod = types.ModuleType("eole.inference_engine")

    class InferenceEnginePY:
        def __init__(self, config):
            self.config = config

    inference_engine_mod.InferenceEnginePY = InferenceEnginePY
    _stub("eole.inference_engine", inference_engine_mod)

    config_run_mod = types.ModuleType("eole.config.run")

    class PredictConfig:
        def __init__(self, **kwargs):
            self.chat_template = kwargs.get("chat_template", "{{ messages }}")

    config_run_mod.PredictConfig = PredictConfig
    _stub("eole.config.run", config_run_mod)

    config_inference_mod = types.ModuleType("eole.config.inference")

    class DecodingConfig(BaseModel):
        pass

    config_inference_mod.DecodingConfig = DecodingConfig
    _stub("eole.config.inference", config_inference_mod)

    bin_mod = types.ModuleType("eole.bin")

    class BaseBin:
        pass

    def register_bin(name):
        def _decorator(cls):
            return cls

        return _decorator

    bin_mod.BaseBin = BaseBin
    bin_mod.register_bin = register_bin
    _stub("eole.bin", bin_mod)

    logging_mod = types.ModuleType("eole.utils.logging")

    def _noop(*args, **kwargs):
        return None

    logging_mod.logger = types.SimpleNamespace(info=_noop, error=_noop)
    _stub("eole.utils.logging", logging_mod)

    constants_mod = types.ModuleType("eole.constants")
    constants_mod.DefaultTokens = types.SimpleNamespace(SEP="<sep>")
    _stub("eole.constants", constants_mod)

    return previous_modules


def _restore_modules(previous_modules):
    for name, previous in previous_modules.items():
        if previous is None:
            del sys.modules[name]
        else:
            sys.modules[name] = previous


def _import_serve_module():
    """Import serve.py with dependency stubs."""
    serve_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "bin", "run", "serve.py"))
    spec = importlib.util.spec_from_file_location("eole_test_serve", serve_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestClaudeServeCompatibility(unittest.TestCase):
    def setUp(self):
        self._previous_modules = _install_stub_modules()
        self.serve = _import_serve_module()
        fd, self.config_path = tempfile.mkstemp(suffix=".yaml")
        os.close(fd)
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                {
                    "models_root": ".",
                    "models": [
                        {
                            "id": "test-model",
                            "path": "dummy/path",
                            "preload": False,
                            "config": {},
                        }
                    ],
                },
                f,
            )

    def tearDown(self):
        if os.path.exists(self.config_path):
            os.remove(self.config_path)
        _restore_modules(self._previous_modules)

    def test_map_claude_settings(self):
        request = self.serve.ClaudeMessagesRequest(
            model="test-model",
            messages=[self.serve.ClaudeMessage(role="user", content="hello")],
            temperature=0.2,
            top_p=0.8,
            max_tokens=42,
            stop_sequences=["END"],
        )
        settings = self.serve.map_claude_to_eole_settings(request)
        self.assertEqual(settings["temperature"], 0.2)
        self.assertEqual(settings["top_p"], 0.8)
        self.assertEqual(settings["max_length"], 42)
        self.assertEqual(settings["stop"], ["END"])

    def test_v1_messages_non_streaming(self):
        app = self.serve.create_app(self.config_path)
        endpoint = app.routes[("POST", "/v1/messages")]
        infer_mock = AsyncMock(return_value=([[MOCK_SCORE]], [["Hello Claude"]]))

        with patch.object(self.serve.Model, "infer_async", infer_mock):
            request = self.serve.ClaudeMessagesRequest(
                model="test-model",
                system="You are a helper",
                messages=[
                    self.serve.ClaudeMessage(
                        role="user",
                        content=[self.serve.ClaudeContentBlock(type="text", text="Hi")],
                    )
                ],
                max_tokens=16,
                temperature=0.5,
                top_p=0.9,
                stream=False,
            )
            response = asyncio.run(endpoint(request))

        payload = response.model_dump()
        self.assertEqual(payload["type"], "message")
        self.assertEqual(payload["role"], "assistant")
        self.assertEqual(payload["content"][0]["type"], "text")
        self.assertEqual(payload["content"][0]["text"], "Hello Claude")
        self.assertIn("input_tokens", payload["usage"])
        self.assertIn("output_tokens", payload["usage"])

        infer_kwargs = infer_mock.await_args.kwargs
        self.assertTrue(infer_kwargs["is_chat"])
        self.assertEqual(infer_kwargs["inputs"][0], {"role": "system", "content": "You are a helper"})
        self.assertEqual(infer_kwargs["inputs"][1], {"role": "user", "content": "Hi"})
        self.assertEqual(infer_kwargs["settings"]["max_length"], 16)

    def test_v1_messages_normalizes_sep_token(self):
        app = self.serve.create_app(self.config_path)
        endpoint = app.routes[("POST", "/v1/messages")]
        infer_mock = AsyncMock(return_value=([[MOCK_SCORE]], [[f"line1{self.serve.DefaultTokens.SEP}line2"]]))

        with patch.object(self.serve.Model, "infer_async", infer_mock):
            request = self.serve.ClaudeMessagesRequest(
                model="test-model",
                messages=[self.serve.ClaudeMessage(role="user", content="Hi")],
                max_tokens=16,
                stream=False,
            )
            response = asyncio.run(endpoint(request))

        payload = response.model_dump()
        self.assertEqual(payload["content"][0]["text"], "line1\nline2")

    def test_v1_messages_preserves_multipart_text_exactly(self):
        app = self.serve.create_app(self.config_path)
        endpoint = app.routes[("POST", "/v1/messages")]
        infer_mock = AsyncMock(return_value=([[MOCK_SCORE]], [["ok"]]))

        with patch.object(self.serve.Model, "infer_async", infer_mock):
            request = self.serve.ClaudeMessagesRequest(
                model="test-model",
                messages=[
                    self.serve.ClaudeMessage(
                        role="user",
                        content=[
                            self.serve.ClaudeContentBlock(type="text", text="\n"),
                            self.serve.ClaudeContentBlock(type="text", text="\n\n"),
                            self.serve.ClaudeContentBlock(type="text", text="show me files"),
                        ],
                    )
                ],
                stream=False,
            )
            asyncio.run(endpoint(request))

        infer_kwargs = infer_mock.await_args.kwargs
        self.assertEqual(infer_kwargs["inputs"][0], {"role": "user", "content": "\n\n\nshow me files"})

    def test_v1_messages_preserves_system_blocks_exactly(self):
        app = self.serve.create_app(self.config_path)
        endpoint = app.routes[("POST", "/v1/messages")]
        infer_mock = AsyncMock(return_value=([[MOCK_SCORE]], [["ok"]]))

        with patch.object(self.serve.Model, "infer_async", infer_mock):
            request = self.serve.ClaudeMessagesRequest(
                model="test-model",
                system=[
                    self.serve.ClaudeContentBlock(type="text", text="You are Claude."),
                    self.serve.ClaudeContentBlock(type="text", text="\nFollow instructions."),
                ],
                messages=[self.serve.ClaudeMessage(role="user", content="Hi")],
                stream=False,
            )
            asyncio.run(endpoint(request))

        infer_kwargs = infer_mock.await_args.kwargs
        self.assertEqual(
            infer_kwargs["inputs"][0],
            {"role": "system", "content": "You are Claude.\nFollow instructions."},
        )

    def test_content_to_text_handles_dict_content_blocks(self):
        blocks = [{"type": "text", "text": "a"}, {"type": "tool_result", "content": [{"type": "text", "text": "b"}]}]
        self.assertEqual(self.serve._content_to_text(blocks), "ab")

    def test_anthropic_path_model_not_found(self):
        app = self.serve.create_app(self.config_path)
        endpoint = app.routes[("POST", "/anthropic/v1/messages")]
        request = self.serve.ClaudeMessagesRequest(
            model="missing-model",
            messages=[self.serve.ClaudeMessage(role="user", content="Hi")],
            max_tokens=8,
        )
        response = asyncio.run(endpoint(request))
        self.assertEqual(response.status_code, 404)
        payload = response.content
        self.assertEqual(payload["type"], "error")
        self.assertEqual(payload["error"]["type"], "not_found_error")

    def test_openai_non_streaming_normalizes_sep_token(self):
        app = self.serve.create_app(self.config_path)
        endpoint = app.routes[("POST", "/v1/chat/completions")]
        infer_mock = AsyncMock(return_value=([[MOCK_SCORE]], [[f"hello{self.serve.DefaultTokens.SEP}world"]]))

        with patch.object(self.serve.Model, "infer_async", infer_mock):
            request = self.serve.OpenAIChatRequest(
                model="test-model",
                messages=[self.serve.OpenAIMessage(role="user", content="Hi")],
                stream=False,
            )
            response = asyncio.run(endpoint(request))

        payload = response.model_dump()
        self.assertEqual(payload["choices"][0]["message"]["content"], "hello\nworld")

    def test_claude_alias_model_id_is_accepted(self):
        app = self.serve.create_app(self.config_path)
        endpoint = app.routes[("POST", "/v1/messages")]
        infer_mock = AsyncMock(return_value=([[MOCK_SCORE]], [["Hello Claude"]]))

        with patch.object(self.serve.Model, "infer_async", infer_mock):
            request = self.serve.ClaudeMessagesRequest(
                model="claude-haiku-4-5-20251001",
                messages=[self.serve.ClaudeMessage(role="user", content="Hi")],
                max_tokens=16,
                stream=False,
            )
            response = asyncio.run(endpoint(request))

        payload = response.model_dump()
        self.assertEqual(payload["model"], "claude-haiku-4-5-20251001")
        self.assertEqual(payload["content"][0]["text"], "Hello Claude")

    def test_openai_alias_model_id_is_accepted(self):
        app = self.serve.create_app(self.config_path)
        endpoint = app.routes[("POST", "/v1/chat/completions")]
        infer_mock = AsyncMock(return_value=([[MOCK_SCORE]], [["Hello OpenAI"]]))

        with patch.object(self.serve.Model, "infer_async", infer_mock):
            request = self.serve.OpenAIChatRequest(
                model="claude-haiku-4-5-20251001",
                messages=[self.serve.OpenAIMessage(role="user", content="Hi")],
                stream=False,
            )
            response = asyncio.run(endpoint(request))

        payload = response.model_dump()
        self.assertEqual(payload["model"], "claude-haiku-4-5-20251001")
        self.assertEqual(payload["choices"][0]["message"]["content"], "Hello OpenAI")

    def test_openai_tools_are_forwarded_to_chat_template(self):
        app = self.serve.create_app(self.config_path)
        endpoint = app.routes[("POST", "/v1/chat/completions")]
        infer_mock = AsyncMock(return_value=([[MOCK_SCORE]], [["Hello with tools"]]))
        tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]
        tool_choice = {"type": "function", "function": {"name": "get_weather"}}

        with patch.object(self.serve.Model, "infer_async", infer_mock):
            request = self.serve.OpenAIChatRequest(
                model="test-model",
                messages=[self.serve.OpenAIMessage(role="user", content="Hi")],
                tools=tools,
                tool_choice=tool_choice,
                stream=False,
            )
            asyncio.run(endpoint(request))

        infer_kwargs = infer_mock.await_args.kwargs
        self.assertEqual(infer_kwargs["chat_template_kwargs"]["tools"], tools)
        self.assertEqual(infer_kwargs["chat_template_kwargs"]["tool_choice"], tool_choice)

    def test_claude_tools_are_forwarded_to_chat_template(self):
        app = self.serve.create_app(self.config_path)
        endpoint = app.routes[("POST", "/v1/messages")]
        infer_mock = AsyncMock(return_value=([[MOCK_SCORE]], [["Hello with tools"]]))
        tools = [{"name": "get_weather", "description": "Fetch weather", "input_schema": {"type": "object"}}]
        tool_choice = {"type": "tool", "name": "get_weather"}

        with patch.object(self.serve.Model, "infer_async", infer_mock):
            request = self.serve.ClaudeMessagesRequest(
                model="test-model",
                messages=[self.serve.ClaudeMessage(role="user", content="Hi")],
                tools=tools,
                tool_choice=tool_choice,
                stream=False,
            )
            asyncio.run(endpoint(request))

        infer_kwargs = infer_mock.await_args.kwargs
        self.assertEqual(infer_kwargs["chat_template_kwargs"]["tools"], tools)
        self.assertEqual(infer_kwargs["chat_template_kwargs"]["tool_choice"], tool_choice)

    def test_claude_tools_default_tool_choice_auto_when_missing(self):
        app = self.serve.create_app(self.config_path)
        endpoint = app.routes[("POST", "/v1/messages")]
        infer_mock = AsyncMock(return_value=([[MOCK_SCORE]], [["Hello with tools"]]))
        tools = [{"name": "Agent", "description": "Run agent", "input_schema": {"type": "object"}}]

        with patch.object(self.serve.Model, "infer_async", infer_mock):
            request = self.serve.ClaudeMessagesRequest(
                model="test-model",
                messages=[self.serve.ClaudeMessage(role="user", content="Hi")],
                tools=tools,
                stream=False,
            )
            asyncio.run(endpoint(request))

        infer_kwargs = infer_mock.await_args.kwargs
        self.assertEqual(infer_kwargs["chat_template_kwargs"]["tools"], tools)
        self.assertEqual(infer_kwargs["chat_template_kwargs"]["tool_choice"], "auto")

    def test_openai_tools_default_tool_choice_auto_when_missing(self):
        app = self.serve.create_app(self.config_path)
        endpoint = app.routes[("POST", "/v1/chat/completions")]
        infer_mock = AsyncMock(return_value=([[MOCK_SCORE]], [["Hello with tools"]]))
        tools = [{"type": "function", "function": {"name": "Agent", "parameters": {"type": "object"}}}]

        with patch.object(self.serve.Model, "infer_async", infer_mock):
            request = self.serve.OpenAIChatRequest(
                model="test-model",
                messages=[self.serve.OpenAIMessage(role="user", content="Hi")],
                tools=tools,
                stream=False,
            )
            asyncio.run(endpoint(request))

        infer_kwargs = infer_mock.await_args.kwargs
        self.assertEqual(infer_kwargs["chat_template_kwargs"]["tools"], tools)
        self.assertEqual(infer_kwargs["chat_template_kwargs"]["tool_choice"], "auto")

    def test_openai_logs_request_and_response_payloads(self):
        app = self.serve.create_app(self.config_path)
        endpoint = app.routes[("POST", "/v1/chat/completions")]
        infer_mock = AsyncMock(return_value=([[MOCK_SCORE]], [["Hello OpenAI"]]))

        with (
            patch.object(self.serve.Model, "infer_async", infer_mock),
            patch.object(self.serve.logger, "info") as info_mock,
        ):
            request = self.serve.OpenAIChatRequest(
                model="test-model",
                messages=[self.serve.OpenAIMessage(role="user", content="Hi")],
                stream=False,
            )
            asyncio.run(endpoint(request))

        logged_lines = [call.args[0] for call in info_mock.call_args_list]
        self.assertTrue(any(line.startswith("OpenAI request: ") for line in logged_lines))
        self.assertTrue(any(line.startswith("OpenAI response: ") for line in logged_lines))

    def test_claude_logs_request_and_response_payloads(self):
        app = self.serve.create_app(self.config_path)
        endpoint = app.routes[("POST", "/v1/messages")]
        infer_mock = AsyncMock(return_value=([[MOCK_SCORE]], [["Hello Claude"]]))

        with (
            patch.object(self.serve.Model, "infer_async", infer_mock),
            patch.object(self.serve.logger, "info") as info_mock,
        ):
            request = self.serve.ClaudeMessagesRequest(
                model="test-model",
                messages=[self.serve.ClaudeMessage(role="user", content="Hi")],
                stream=False,
            )
            asyncio.run(endpoint(request))

        logged_lines = [call.args[0] for call in info_mock.call_args_list]
        self.assertTrue(any(line.startswith("Claude request: ") for line in logged_lines))
        self.assertTrue(any(line.startswith("Claude response: ") for line in logged_lines))

    def test_claude_stream_logs_structured_start_and_final_response(self):
        app = self.serve.create_app(self.config_path)
        endpoint = app.routes[("POST", "/v1/messages")]

        def fake_load(model_self):
            model_self.loaded = True
            model_self.config = type("Cfg", (), {"chat_template": "{{ messages[0]['content'] }}"})()
            model_self.engine = types.SimpleNamespace(infer_list_stream=lambda _chat_input, settings=None: iter(["hello"]))

        with (
            patch.object(self.serve.Model, "load", fake_load),
            patch.object(self.serve.logger, "info") as info_mock,
        ):
            request = self.serve.ClaudeMessagesRequest(
                model="test-model",
                messages=[self.serve.ClaudeMessage(role="user", content="Hi")],
                stream=True,
            )
            response = asyncio.run(endpoint(request))
            asyncio.run(_drain_async_generator(response.content))

        logged_lines = [call.args[0] for call in info_mock.call_args_list]
        self.assertTrue(any('"type": "message_start"' in line for line in logged_lines))
        self.assertTrue(any('"type": "message"' in line and '"text": "hello"' in line for line in logged_lines))

    def test_apply_chat_template_supports_generation_block(self):
        model = self.serve.Model(
            model_id="test-model",
            model_path="dummy/path",
            preload=False,
            models_root=".",
            pre_config={},
        )
        model.config = type("Cfg", (), {"chat_template": "{% for m in messages %}{% generation %}{{ m['content'] }}{% endgeneration %}{% endfor %}"})()

        rendered = model.apply_chat_template([{"role": "user", "content": "hello"}])
        self.assertEqual(rendered, "hello")

    def test_apply_chat_template_supports_strftime_now_global(self):
        model = self.serve.Model(
            model_id="test-model",
            model_path="dummy/path",
            preload=False,
            models_root=".",
            pre_config={},
        )
        model.config = type("Cfg", (), {"chat_template": "{{ strftime_now('%Y') }}"})()

        rendered = model.apply_chat_template([{"role": "user", "content": "hello"}])
        self.assertEqual(len(rendered), 4)
        self.assertTrue(rendered.isdigit())

    def test_apply_chat_template_logs_rendered_prompt(self):
        model = self.serve.Model(
            model_id="test-model",
            model_path="dummy/path",
            preload=False,
            models_root=".",
            pre_config={},
        )
        model.config = type("Cfg", (), {"chat_template": "{{ messages[0]['content'] }}"})()
        model.local_path = "."

        with patch.object(self.serve.logger, "info") as info_mock:
            rendered = model.apply_chat_template([{"role": "user", "content": "hello"}])

        self.assertEqual(rendered, "hello")
        logged_lines = [call.args[0] for call in info_mock.call_args_list]
        self.assertTrue(any(line.startswith("Rendered chat prompt: ") and '"prompt": "hello"' in line for line in logged_lines))


if __name__ == "__main__":
    unittest.main()
