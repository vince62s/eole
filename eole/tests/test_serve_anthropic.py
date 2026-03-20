"""Unit tests for the Anthropic-compatible serve helpers.

These tests exercise the pure-Python helper functions added to
``eole/bin/run/serve.py`` without requiring torch, a real model, or a
running server.
"""

import importlib.util
import json
import os
import sys
import unittest


# ---------------------------------------------------------------------------
# Import serve.py helpers without pulling in torch or the full eole package
# ---------------------------------------------------------------------------

def _load_serve_module():
    """Load serve.py as an isolated module, stubbing heavy dependencies.

    serve.py is imported via importlib rather than the normal package import
    path to avoid pulling in the eole package-level ``__init__.py`` which
    requires torch.  All heavy dependencies (torch, uvicorn, eole.*) are
    replaced with minimal stubs so that only the pure-Python helper functions
    under test are exercised.
    """

    # Minimal stubs so that the top-level imports in serve.py succeed without
    # real torch / fastapi / eole infrastructure being available.
    for mod_name, stub in [
        ("torch", type(sys)("torch")),
        ("uvicorn", type(sys)("uvicorn")),
        ("yaml", type(sys)("yaml")),
        ("eole", type(sys)("eole")),
        ("eole.inference_engine", type(sys)("eole.inference_engine")),
        ("eole.config", type(sys)("eole.config")),
        ("eole.config.run", type(sys)("eole.config.run")),
        ("eole.config.inference", type(sys)("eole.config.inference")),
        ("eole.bin", type(sys)("eole.bin")),
        ("eole.utils", type(sys)("eole.utils")),
        ("eole.utils.logging", type(sys)("eole.utils.logging")),
        ("eole.constants", type(sys)("eole.constants")),
    ]:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = stub

    # Minimal attribute fakes required by serve.py module-level code
    eole_stub = sys.modules["eole"]
    eole_stub.__version__ = "0.0.0-test"

    torch_stub = sys.modules["torch"]
    torch_stub.cuda = type(sys)("torch.cuda")
    torch_stub.cuda.empty_cache = lambda: None

    constants_stub = sys.modules["eole.constants"]

    class _DefaultTokens:
        SEP = "<sep>"

    constants_stub.DefaultTokens = _DefaultTokens

    logging_stub = sys.modules["eole.utils.logging"]
    logging_stub.logger = type(
        "_Logger", (), {"info": lambda *a, **k: None, "error": lambda *a, **k: None}
    )()

    # DecodingConfig stub (base class for TextRequest / ChatRequest)
    config_inference_stub = sys.modules["eole.config.inference"]

    class _DecodingConfig:
        pass

    config_inference_stub.DecodingConfig = _DecodingConfig

    # PredictConfig stub
    config_run_stub = sys.modules["eole.config.run"]

    class _PredictConfig:
        def __init__(self, *a, **k):
            self.chat_template = None

    config_run_stub.PredictConfig = _PredictConfig

    # BaseBin / register_bin stubs
    bin_stub = sys.modules["eole.bin"]
    bin_stub.register_bin = lambda name: (lambda cls: cls)
    bin_stub.BaseBin = object

    # InferenceEnginePY stub
    ie_stub = sys.modules["eole.inference_engine"]
    ie_stub.InferenceEnginePY = object

    serve_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "bin",
        "run",
        "serve.py",
    )
    spec = importlib.util.spec_from_file_location("eole.bin.run.serve", serve_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_serve = _load_serve_module()

_parse_anthropic_response_content = _serve._parse_anthropic_response_content
_anthropic_messages_to_openai = _serve._anthropic_messages_to_openai
_anthropic_tools_to_openai = _serve._anthropic_tools_to_openai
_map_anthropic_to_eole_settings = _serve._map_anthropic_to_eole_settings
AnthropicMessagesRequest = _serve.AnthropicMessagesRequest
AnthropicTool = _serve.AnthropicTool
AnthropicToolChoice = _serve.AnthropicToolChoice
AnthropicInputMessage = _serve.AnthropicInputMessage
AnthropicUsage = _serve.AnthropicUsage
AnthropicMessagesResponse = _serve.AnthropicMessagesResponse


# ---------------------------------------------------------------------------
# Tests for _parse_anthropic_response_content
# ---------------------------------------------------------------------------


class TestParseAnthropicResponseContent(unittest.TestCase):
    def test_plain_text_returns_single_text_block(self):
        text = "Hello, world!"
        blocks, stop_reason = _parse_anthropic_response_content(text)
        self.assertEqual(stop_reason, "end_turn")
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["type"], "text")
        self.assertEqual(blocks[0]["text"], text)

    def test_empty_string_returns_text_block(self):
        blocks, stop_reason = _parse_anthropic_response_content("")
        self.assertEqual(stop_reason, "end_turn")
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["type"], "text")

    def test_single_tool_use_block(self):
        text = '<tool_use id="toolu_abc" name="get_weather">{"location": "Paris"}</tool_use>'
        blocks, stop_reason = _parse_anthropic_response_content(text)
        self.assertEqual(stop_reason, "tool_use")
        self.assertEqual(len(blocks), 1)
        b = blocks[0]
        self.assertEqual(b["type"], "tool_use")
        self.assertEqual(b["id"], "toolu_abc")
        self.assertEqual(b["name"], "get_weather")
        self.assertEqual(b["input"], {"location": "Paris"})

    def test_text_before_and_after_tool_use(self):
        text = (
            "Sure, let me check.\n"
            '<tool_use id="toolu_1" name="lookup">{"q": "python"}</tool_use>\n'
            "Done."
        )
        blocks, stop_reason = _parse_anthropic_response_content(text)
        self.assertEqual(stop_reason, "tool_use")
        types = [b["type"] for b in blocks]
        self.assertIn("text", types)
        self.assertIn("tool_use", types)

    def test_multiple_tool_use_blocks(self):
        text = (
            '<tool_use id="t1" name="search">{"q": "a"}</tool_use>'
            '<tool_use id="t2" name="fetch">{"url": "b"}</tool_use>'
        )
        blocks, stop_reason = _parse_anthropic_response_content(text)
        self.assertEqual(stop_reason, "tool_use")
        tool_blocks = [b for b in blocks if b["type"] == "tool_use"]
        self.assertEqual(len(tool_blocks), 2)
        self.assertEqual(tool_blocks[0]["name"], "search")
        self.assertEqual(tool_blocks[1]["name"], "fetch")

    def test_tool_use_without_id_generates_one(self):
        text = '<tool_use name="calc">{"expr": "1+1"}</tool_use>'
        blocks, _ = _parse_anthropic_response_content(text)
        b = next(b for b in blocks if b["type"] == "tool_use")
        self.assertTrue(b["id"].startswith("toolu_"))
        self.assertEqual(b["name"], "calc")

    def test_malformed_json_in_tool_use_stored_as_raw(self):
        text = '<tool_use id="x" name="fn">not valid json</tool_use>'
        blocks, stop_reason = _parse_anthropic_response_content(text)
        self.assertEqual(stop_reason, "tool_use")
        b = next(b for b in blocks if b["type"] == "tool_use")
        self.assertEqual(b["input"].get("raw"), "not valid json")


# ---------------------------------------------------------------------------
# Tests for _anthropic_messages_to_openai
# ---------------------------------------------------------------------------


class TestAnthropicMessagesToOpenai(unittest.TestCase):
    def _make_msg(self, role, content):
        return AnthropicInputMessage(role=role, content=content)

    def test_simple_string_messages_no_system(self):
        msgs = [self._make_msg("user", "Hi"), self._make_msg("assistant", "Hello")]
        result = _anthropic_messages_to_openai(msgs)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], {"role": "user", "content": "Hi"})
        self.assertEqual(result[1], {"role": "assistant", "content": "Hello"})

    def test_system_string_prepended(self):
        msgs = [self._make_msg("user", "Hi")]
        result = _anthropic_messages_to_openai(msgs, system="Be helpful.")
        self.assertEqual(result[0]["role"], "system")
        self.assertEqual(result[0]["content"], "Be helpful.")
        self.assertEqual(len(result), 2)

    def test_system_block_list_prepended(self):
        system = [{"type": "text", "text": "You are a bot."}]
        msgs = [self._make_msg("user", "Go")]
        result = _anthropic_messages_to_openai(msgs, system=system)
        self.assertEqual(result[0]["role"], "system")
        self.assertEqual(result[0]["content"], "You are a bot.")

    def test_text_content_blocks_concatenated(self):
        content = [
            {"type": "text", "text": "Hello"},
            {"type": "text", "text": " world"},
        ]
        msgs = [self._make_msg("user", content)]
        result = _anthropic_messages_to_openai(msgs)
        # Multiple text blocks are joined with '\n' as the separator.
        self.assertEqual(result[0]["content"], "Hello\n world")

    def test_tool_use_block_becomes_tool_calls_on_assistant_message(self):
        content = [
            {"type": "text", "text": "Calling tool."},
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "my_tool",
                "input": {"key": "val"},
            },
        ]
        msgs = [self._make_msg("assistant", content)]
        result = _anthropic_messages_to_openai(msgs)
        self.assertEqual(len(result), 1)
        msg = result[0]
        # Text prefix preserved as content
        self.assertEqual(msg["content"], "Calling tool.")
        # Tool call in proper OpenAI tool_calls format
        self.assertIn("tool_calls", msg)
        tc = msg["tool_calls"][0]
        self.assertEqual(tc["id"], "toolu_1")
        self.assertEqual(tc["type"], "function")
        self.assertEqual(tc["function"]["name"], "my_tool")
        self.assertEqual(json.loads(tc["function"]["arguments"]), {"key": "val"})

    def test_tool_use_only_assistant_message_content_is_none(self):
        content = [
            {
                "type": "tool_use",
                "id": "toolu_2",
                "name": "search",
                "input": {"q": "python"},
            },
        ]
        msgs = [self._make_msg("assistant", content)]
        result = _anthropic_messages_to_openai(msgs)
        self.assertEqual(len(result), 1)
        msg = result[0]
        # No text blocks → content should be None
        self.assertIsNone(msg["content"])
        self.assertEqual(len(msg["tool_calls"]), 1)

    def test_tool_result_becomes_tool_role_message(self):
        content = [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_1",
                "content": "42°C",
            }
        ]
        msgs = [self._make_msg("user", content)]
        result = _anthropic_messages_to_openai(msgs)
        tool_msgs = [m for m in result if m["role"] == "tool"]
        self.assertEqual(len(tool_msgs), 1)
        self.assertEqual(tool_msgs[0]["tool_call_id"], "toolu_1")
        self.assertEqual(tool_msgs[0]["content"], "42°C")

    def test_tool_result_with_list_content(self):
        content = [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_2",
                "content": [{"type": "text", "text": "answer"}],
            }
        ]
        msgs = [self._make_msg("user", content)]
        result = _anthropic_messages_to_openai(msgs)
        tool_msgs = [m for m in result if m["role"] == "tool"]
        self.assertEqual(tool_msgs[0]["content"], "answer")


# ---------------------------------------------------------------------------
# Tests for _anthropic_tools_to_openai
# ---------------------------------------------------------------------------


class TestAnthropicToolsToOpenai(unittest.TestCase):
    def test_basic_conversion(self):
        tools = [
            AnthropicTool(
                name="get_weather",
                description="Get the weather",
                input_schema={"type": "object", "properties": {"location": {"type": "string"}}},
            )
        ]
        result = _anthropic_tools_to_openai(tools)
        self.assertEqual(len(result), 1)
        t = result[0]
        self.assertEqual(t["type"], "function")
        self.assertEqual(t["function"]["name"], "get_weather")
        self.assertEqual(t["function"]["description"], "Get the weather")
        self.assertIn("location", t["function"]["parameters"]["properties"])

    def test_multiple_tools(self):
        tools = [
            AnthropicTool(name="a", input_schema={}),
            AnthropicTool(name="b", input_schema={}),
        ]
        result = _anthropic_tools_to_openai(tools)
        self.assertEqual([t["function"]["name"] for t in result], ["a", "b"])

    def test_missing_description_becomes_empty_string(self):
        tools = [AnthropicTool(name="no_desc", input_schema={})]
        result = _anthropic_tools_to_openai(tools)
        self.assertEqual(result[0]["function"]["description"], "")


# ---------------------------------------------------------------------------
# Tests for _map_anthropic_to_eole_settings
# ---------------------------------------------------------------------------


class TestMapAnthropicToEoleSettings(unittest.TestCase):
    def _make_req(self, **kwargs):
        base = {"model": "claude-3", "messages": [{"role": "user", "content": "hi"}]}
        base.update(kwargs)
        return AnthropicMessagesRequest(**base)

    def test_temperature_mapped(self):
        req = self._make_req(temperature=0.5)
        s = _map_anthropic_to_eole_settings(req)
        self.assertAlmostEqual(s["temperature"], 0.5)

    def test_top_p_mapped(self):
        req = self._make_req(top_p=0.9)
        s = _map_anthropic_to_eole_settings(req)
        self.assertAlmostEqual(s["top_p"], 0.9)

    def test_max_tokens_mapped_to_max_length(self):
        req = self._make_req(max_tokens=256)
        s = _map_anthropic_to_eole_settings(req)
        self.assertEqual(s["max_length"], 256)

    def test_stop_sequences_mapped(self):
        req = self._make_req(stop_sequences=["<|end|>"])
        s = _map_anthropic_to_eole_settings(req)
        self.assertEqual(s["stop"], ["<|end|>"])

    def test_no_stop_sequences_absent_from_settings(self):
        req = self._make_req()
        s = _map_anthropic_to_eole_settings(req)
        self.assertNotIn("stop", s)


# ---------------------------------------------------------------------------
# Tests for Pydantic model validation
# ---------------------------------------------------------------------------


class TestAnthropicMessagesRequest(unittest.TestCase):
    def _base_payload(self):
        return {
            "model": "claude-3-5-sonnet",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 512,
        }

    def test_minimal_valid_request(self):
        req = AnthropicMessagesRequest(**self._base_payload())
        self.assertEqual(req.model, "claude-3-5-sonnet")
        self.assertEqual(len(req.messages), 1)
        self.assertEqual(req.max_tokens, 512)
        self.assertFalse(req.stream)

    def test_defaults(self):
        req = AnthropicMessagesRequest(**self._base_payload())
        self.assertIsNone(req.system)
        self.assertIsNone(req.tools)
        self.assertIsNone(req.tool_choice)
        self.assertAlmostEqual(req.temperature, 1.0)

    def test_with_system(self):
        payload = self._base_payload()
        payload["system"] = "You are helpful."
        req = AnthropicMessagesRequest(**payload)
        self.assertEqual(req.system, "You are helpful.")

    def test_with_tools(self):
        payload = self._base_payload()
        payload["tools"] = [{"name": "fn", "input_schema": {}}]
        req = AnthropicMessagesRequest(**payload)
        self.assertEqual(len(req.tools), 1)
        self.assertEqual(req.tools[0].name, "fn")

    def test_tool_choice_auto(self):
        payload = self._base_payload()
        payload["tool_choice"] = {"type": "auto"}
        req = AnthropicMessagesRequest(**payload)
        self.assertEqual(req.tool_choice.type, "auto")

    def test_tool_choice_specific_tool(self):
        payload = self._base_payload()
        payload["tool_choice"] = {"type": "tool", "name": "my_fn"}
        req = AnthropicMessagesRequest(**payload)
        self.assertEqual(req.tool_choice.type, "tool")
        self.assertEqual(req.tool_choice.name, "my_fn")

    def test_extra_fields_ignored(self):
        payload = self._base_payload()
        payload["unknown_field"] = "should be ignored"
        req = AnthropicMessagesRequest(**payload)
        self.assertFalse(hasattr(req, "unknown_field"))

    def test_stream_false_by_default(self):
        req = AnthropicMessagesRequest(**self._base_payload())
        self.assertFalse(req.stream)

    def test_stream_true(self):
        payload = self._base_payload()
        payload["stream"] = True
        req = AnthropicMessagesRequest(**payload)
        self.assertTrue(req.stream)


class TestAnthropicMessagesResponse(unittest.TestCase):
    def _make_response(self, **kwargs):
        base = {
            "id": "msg_test123",
            "content": [{"type": "text", "text": "Hi there"}],
            "model": "claude-3-5-sonnet",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        base.update(kwargs)
        return AnthropicMessagesResponse(**base)

    def test_defaults(self):
        r = self._make_response()
        self.assertEqual(r.type, "message")
        self.assertEqual(r.role, "assistant")
        self.assertIsNone(r.stop_reason)
        self.assertIsNone(r.stop_sequence)

    def test_stop_reason_end_turn(self):
        r = self._make_response(stop_reason="end_turn")
        self.assertEqual(r.stop_reason, "end_turn")

    def test_stop_reason_tool_use(self):
        r = self._make_response(stop_reason="tool_use")
        self.assertEqual(r.stop_reason, "tool_use")


# ---------------------------------------------------------------------------
# Round-trip integration: multi-turn conversation with tool calling
# ---------------------------------------------------------------------------


class TestRoundTripToolCall(unittest.TestCase):
    """
    Simulate a full tool-call round trip:
      1. User asks a question
      2. Assistant responds with a tool_use block
      3. User sends tool_result
      4. Verify conversion to OpenAI messages is correct
      5. Verify that model output with <tool_use> tags is parsed correctly
    """

    def test_full_tool_call_round_trip(self):
        # Step 1-2: assistant turn with tool_use block
        assistant_content = [
            {"type": "text", "text": "Let me look that up."},
            {
                "type": "tool_use",
                "id": "toolu_abc123",
                "name": "get_weather",
                "input": {"location": "London"},
            },
        ]
        # Step 3: user sends tool_result
        user_content = [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_abc123",
                "content": "15°C, partly cloudy",
            }
        ]
        messages = [
            AnthropicInputMessage(role="user", content="What's the weather in London?"),
            AnthropicInputMessage(role="assistant", content=assistant_content),
            AnthropicInputMessage(role="user", content=user_content),
        ]
        openai_msgs = _anthropic_messages_to_openai(messages, system="You are helpful.")

        # System message first
        self.assertEqual(openai_msgs[0]["role"], "system")

        # User question
        self.assertEqual(openai_msgs[1]["role"], "user")
        self.assertEqual(openai_msgs[1]["content"], "What's the weather in London?")

        # Assistant turn: text preserved, tool call in tool_calls array
        asst_msg = openai_msgs[2]
        self.assertEqual(asst_msg["role"], "assistant")
        self.assertEqual(asst_msg["content"], "Let me look that up.")
        self.assertIn("tool_calls", asst_msg)
        tc = asst_msg["tool_calls"][0]
        self.assertEqual(tc["id"], "toolu_abc123")
        self.assertEqual(tc["function"]["name"], "get_weather")
        self.assertEqual(json.loads(tc["function"]["arguments"]), {"location": "London"})

        # Tool result message
        tool_msg = openai_msgs[3]
        self.assertEqual(tool_msg["role"], "tool")
        self.assertEqual(tool_msg["tool_call_id"], "toolu_abc123")
        self.assertEqual(tool_msg["content"], "15°C, partly cloudy")

        # Step 5: parse model output that uses <tool_call> format (most common
        # for open-source tool-capable models like Hermes/Qwen/etc.)
        model_output_tool_call = (
            "Sure!\n"
            "<tool_call>\n"
            '{"name": "get_weather", "arguments": {"location": "Paris"}}\n'
            "</tool_call>"
        )
        blocks, stop_reason = _parse_anthropic_response_content(model_output_tool_call)
        self.assertEqual(stop_reason, "tool_use")
        tool_block = next(b for b in blocks if b["type"] == "tool_use")
        self.assertEqual(tool_block["name"], "get_weather")
        self.assertEqual(tool_block["input"]["location"], "Paris")

        # Also verify the <tool_use id="…" name="…"> format still works —
        # this is the round-trip format we emit when serialising prior Anthropic
        # tool_use turns back into the conversation history for the next request.
        model_output_tool_use = (
            "Sure!\n"
            '<tool_use id="toolu_xyz" name="get_weather">{"location": "Paris"}</tool_use>'
        )
        blocks2, stop_reason2 = _parse_anthropic_response_content(model_output_tool_use)
        self.assertEqual(stop_reason2, "tool_use")
        tool_block2 = next(b for b in blocks2 if b["type"] == "tool_use")
        self.assertEqual(tool_block2["name"], "get_weather")
        self.assertEqual(tool_block2["input"]["location"], "Paris")


# ---------------------------------------------------------------------------
# Tests for _parse_anthropic_response_content: <tool_call> format
# ---------------------------------------------------------------------------


class TestParseToolCallFormat(unittest.TestCase):
    """Test parsing of <tool_call> tags used by Hermes/NousResearch models."""

    def test_single_tool_call_block(self):
        text = '<tool_call>\n{"name": "search", "arguments": {"q": "test"}}\n</tool_call>'
        blocks, stop_reason = _parse_anthropic_response_content(text)
        self.assertEqual(stop_reason, "tool_use")
        self.assertEqual(len(blocks), 1)
        b = blocks[0]
        self.assertEqual(b["type"], "tool_use")
        self.assertEqual(b["name"], "search")
        self.assertEqual(b["input"]["q"], "test")

    def test_tool_call_with_parameters_key(self):
        text = '<tool_call>{"name": "fn", "parameters": {"x": 1}}</tool_call>'
        blocks, stop_reason = _parse_anthropic_response_content(text)
        self.assertEqual(stop_reason, "tool_use")
        b = blocks[0]
        self.assertEqual(b["name"], "fn")
        self.assertEqual(b["input"]["x"], 1)

    def test_tool_call_with_string_arguments(self):
        text = '<tool_call>{"name": "fn", "arguments": "{\\"k\\": \\"v\\"}"}</tool_call>'
        blocks, stop_reason = _parse_anthropic_response_content(text)
        self.assertEqual(stop_reason, "tool_use")
        self.assertEqual(blocks[0]["input"]["k"], "v")

    def test_text_before_tool_call(self):
        text = "I will call the tool.\n<tool_call>{\"name\": \"fn\", \"arguments\": {}}</tool_call>"
        blocks, stop_reason = _parse_anthropic_response_content(text)
        self.assertEqual(stop_reason, "tool_use")
        types = [b["type"] for b in blocks]
        self.assertIn("text", types)
        self.assertIn("tool_use", types)

    def test_multiple_tool_call_blocks(self):
        text = (
            '<tool_call>{"name": "a", "arguments": {}}</tool_call>'
            '<tool_call>{"name": "b", "arguments": {}}</tool_call>'
        )
        blocks, stop_reason = _parse_anthropic_response_content(text)
        self.assertEqual(stop_reason, "tool_use")
        tool_blocks = [b for b in blocks if b["type"] == "tool_use"]
        self.assertEqual(len(tool_blocks), 2)
        self.assertEqual(tool_blocks[0]["name"], "a")
        self.assertEqual(tool_blocks[1]["name"], "b")

    def test_malformed_json_in_tool_call_stored_as_raw(self):
        text = "<tool_call>not json</tool_call>"
        blocks, stop_reason = _parse_anthropic_response_content(text)
        self.assertEqual(stop_reason, "tool_use")
        b = blocks[0]
        self.assertEqual(b["name"], "")
        self.assertIn("raw", b["input"])

    def test_tool_call_takes_precedence_over_tool_use_when_both_present(self):
        # When <tool_call> is present, the <tool_call> parser is used.
        text = '<tool_call>{"name": "fn1", "arguments": {}}</tool_call>'
        blocks, _ = _parse_anthropic_response_content(text)
        self.assertEqual(blocks[0]["name"], "fn1")


# ---------------------------------------------------------------------------
# Tests for _log_json_payload
# ---------------------------------------------------------------------------


class TestLogJsonPayload(unittest.TestCase):
    def test_dict_payload_logged_without_error(self):
        captured = []

        class _FakeLogger:
            def info(self, msg):
                captured.append(msg)

        # Monkey-patch the logger on the module
        original = _serve.logger
        _serve.logger = _FakeLogger()
        try:
            _serve._log_json_payload("TEST LABEL", {"key": "value"})
        finally:
            _serve.logger = original

        self.assertEqual(len(captured), 1)
        self.assertIn("TEST LABEL", captured[0])
        self.assertIn('"key"', captured[0])

    def test_string_payload_logged_without_error(self):
        captured = []

        class _FakeLogger:
            def info(self, msg):
                captured.append(msg)

        original = _serve.logger
        _serve.logger = _FakeLogger()
        try:
            _serve._log_json_payload("PROMPT", "Hello world prompt text")
        finally:
            _serve.logger = original

        self.assertEqual(len(captured), 1)
        self.assertIn("PROMPT", captured[0])
        self.assertIn("Hello world prompt text", captured[0])

    def test_list_payload_logged(self):
        captured = []

        class _FakeLogger:
            def info(self, msg):
                captured.append(msg)

        original = _serve.logger
        _serve.logger = _FakeLogger()
        try:
            _serve._log_json_payload("LIST", [1, 2, 3])
        finally:
            _serve.logger = original

        self.assertIn("[", captured[0])


if __name__ == "__main__":
    unittest.main()
