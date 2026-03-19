#!/usr/bin/env python

import os
import time
import gc
import yaml
import json
import uuid
from typing import Any, List, Union, Optional, Literal

import torch
import uvicorn

from fastapi import FastAPI, Request, Body
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator
from jinja2.exceptions import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment

import asyncio
from dataclasses import dataclass

import eole
from eole.inference_engine import InferenceEnginePY
from eole.config.run import PredictConfig
from eole.config.inference import DecodingConfig
from eole.bin import register_bin, BaseBin
from eole.utils.logging import logger
from eole.constants import DefaultTokens

CLAUDE_HAIKU_MODEL_ALIAS = "claude-haiku-4-5-20251001"

STATUS_OK = "ok"
STATUS_ERROR = "error"


class TextRequest(DecodingConfig):
    """
    Standard text "completion" request
    (as well as encoder/decoder models e.g. translation).
    """

    model: int | str = Field(description="Model identifier from server configuration.")
    inputs: Union[str, List[str]] = Field(
        description="List of inputs to run inference on. "
        "A single string will be automatically cast to a single item list."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "model": "llama3-8b-instruct",
                "inputs": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a funny guy.<|eot_id|><|start_header_id|>user<|end_header_id|>Tell me a joke :)<|eot_id|><|start_header_id|>assistant<|end_header_id|>",  # noqa: E501
            }
        }


class TextResponse(BaseModel):
    """
    Response of TextRequest.
    """

    predictions: List[List[str]] = Field(description="List of prediction(s) for each input(s).")
    scores: List[List[float]] = Field(description="Pred scores from the model for each prediction.")

    class Config:
        json_schema_extra = {
            "example": {
                "predictions": [
                    [
                        "\n\nHere's one:\n\nWhy couldn't the bicycle stand up by itself?\n\n(wait for it...)\n\nBecause it was two-tired!\n\nHope that made you laugh!"  # noqa: E501
                    ]
                ],
                "scores": [[-0.040771484375]],
            }
        }

    @model_validator(mode="after")
    def _validate_response(self):
        """
        Automatically apply some formatting to the provided text response.
        This logic might be moved elsewhere at some point.
        """
        self.predictions = [[pred.replace(DefaultTokens.SEP, "\n") for pred in preds] for preds in self.predictions]
        return self


class ChatRequest(DecodingConfig):
    """
    Request format for chat-based interactions.
    """

    model: int | str = Field(description="Model identifier from server configuration.")
    messages: List[dict] = Field(description="List of message dictionaries with 'role' and 'content' keys.")

    class Config:
        json_schema_extra = {
            "example": {
                "model": "llama3-8b-instruct",
                "messages": [
                    {"role": "system", "content": "You are a funny guy."},
                    {"role": "user", "content": "Tell me a joke :)"},
                ],
            }
        }


# TODO: specific response model for chat mode?
# class ChatResponse(BaseModel):
#     choices: List[dict]


class OpenAIMessage(BaseModel):
    # Allow any extra fields (e.g. name, tool_call_id, tool_calls) sent by
    # OpenAI-compatible clients without triggering a 422.
    model_config = ConfigDict(extra="ignore")

    role: Literal["system", "user", "assistant", "tool"]
    # Per the OpenAI API spec, content can be a string, an array of content
    # parts (multimodal), or null (when the message only carries tool_calls).
    content: Optional[Union[str, List[Any]]] = None


class OpenAIChatRequest(BaseModel):
    # Silently drop any OpenAI-compatible fields not explicitly declared here
    # (e.g. top_k, tools, tool_choice, response_format, seed, …) so that
    # clients that send them don't receive a 422.
    model_config = ConfigDict(
        extra="ignore",
        json_schema_extra={
            "example": {
                "model": "llama3-8b-instruct",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello!"},
                ],
                "temperature": 0.7,
                "max_tokens": 100,
            }
        },
    )

    model: str
    messages: List[OpenAIMessage]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 0.95
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[dict] = None
    user: Optional[str] = None
    tools: Optional[List[dict]] = None
    tool_choice: Optional[Union[str, dict]] = None


class ClaudeContentBlock(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: str = "text"
    text: Optional[str] = None


class ClaudeMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: Literal["user", "assistant"]
    content: Union[str, List[ClaudeContentBlock]]


class ClaudeMessagesRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str
    messages: List[ClaudeMessage]
    system: Optional[Union[str, List[ClaudeContentBlock]]] = None
    max_tokens: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = None
    stream: Optional[bool] = False
    tools: Optional[List[dict]] = None
    tool_choice: Optional[Union[str, dict]] = None


class OpenAIUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class OpenAIChoice(BaseModel):
    index: int
    message: OpenAIMessage
    finish_reason: Literal["stop", "length", "content_filter", "null"]


class OpenAIChatResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: List[OpenAIChoice]
    usage: OpenAIUsage

    class Config:
        json_schema_extra = {
            "example": {
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1677652288,
                "model": "llama3-8b-instruct",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello! How can I assist you today?"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
            }
        }


# ---------------------------------------------------------------------------
# SSE streaming models (OpenAI chat.completion.chunk format)
# ---------------------------------------------------------------------------


class OpenAIStreamDelta(BaseModel):
    """Delta object inside a streaming chunk choice."""

    role: Optional[Literal["assistant"]] = None
    content: Optional[str] = None


class OpenAIStreamChoice(BaseModel):
    """Choice object inside a streaming chunk."""

    index: int
    delta: OpenAIStreamDelta
    finish_reason: Optional[Literal["stop", "length"]] = None


class OpenAIStreamChunk(BaseModel):
    """A single Server-Sent Events chunk in OpenAI streaming format."""

    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: List[OpenAIStreamChoice]


class ClaudeUsage(BaseModel):
    input_tokens: int
    output_tokens: int


class ClaudeResponseContent(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ClaudeMessageResponse(BaseModel):
    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: List[ClaudeResponseContent]
    model: str
    stop_reason: Optional[Literal["end_turn", "max_tokens", "stop_sequence"]] = "end_turn"
    stop_sequence: Optional[str] = None
    usage: ClaudeUsage


def map_openai_to_eole_settings(openai_request: OpenAIChatRequest) -> dict:
    """
    Map OpenAI parameters to Eole settings.
    """
    settings = {}

    if openai_request.temperature is not None:
        settings["temperature"] = openai_request.temperature

    if openai_request.top_p is not None:
        settings["top_p"] = openai_request.top_p

    if openai_request.max_tokens is not None:
        settings["max_length"] = openai_request.max_tokens

    if openai_request.stop is not None:
        if isinstance(openai_request.stop, str):
            settings["stop"] = [openai_request.stop]
        else:
            settings["stop"] = openai_request.stop

    # Note: presence_penalty, frequency_penalty, logit_bias
    # may not have direct equivalents in your engine
    # You can add custom mappings if your engine supports similar features

    return settings


def map_claude_to_eole_settings(claude_request: ClaudeMessagesRequest) -> dict:
    """
    Map Claude parameters to Eole settings.
    """
    settings = {}

    if claude_request.temperature is not None:
        settings["temperature"] = claude_request.temperature

    if claude_request.top_p is not None:
        settings["top_p"] = claude_request.top_p

    if claude_request.max_tokens is not None:
        settings["max_length"] = claude_request.max_tokens

    if claude_request.stop_sequences:
        settings["stop"] = claude_request.stop_sequences

    return settings


def _content_to_text(content: Optional[Union[str, List[Any]]]) -> str:
    """
    Normalize API message content payloads into plain text.
    """
    if content is None:
        return ""
    if isinstance(content, list):
        return " ".join(p.get("text", "") if isinstance(p, dict) else getattr(p, "text", str(p)) for p in content)
    return str(content)


def _normalize_generated_text(text: str) -> str:
    """
    Normalize model-generated text for API responses.
    """
    return text.replace(DefaultTokens.SEP, "\n")


def _resolve_model_id(server, requested_model_id: str):
    """
    Resolve an incoming model id to a configured model id.
    """
    if requested_model_id in server.models:
        return requested_model_id, requested_model_id

    if requested_model_id == CLAUDE_HAIKU_MODEL_ALIAS and server.models:
        fallback_model_id = next(iter(server.models))
        return fallback_model_id, requested_model_id

    return None, requested_model_id


def estimate_tokens(text: str) -> int:
    """
    Rough token estimation (about 4 chars per token).
    For production, use a proper tokenizer.
    """
    return len(text) // 4


class Server(object):
    """
    Main server class to manage configuration, models and corresponding constraints.
    """

    def __init__(self):
        self.start_time = time.time()
        self.models = {}
        self.models_root = None

    def start(self, server_config_path):
        """
        Initialize the server with the given configuration.
        """
        with open(server_config_path) as f:
            server_config = yaml.safe_load(f)
        self.models_root = server_config["models_root"]
        for model in server_config["models"]:
            # instantiate models
            model_id = model["id"]
            model_path = model["path"]
            self.models[model_id] = Model(
                model_id=model_id,
                model_path=model_path,
                models_root=self.models_root,
                model_type=model.get("model_type", "default"),
                pre_config=model.get("config", {}),
            )
            if model.get("preload", False):
                self.models[model_id].load()

    def available_models(self):
        """
        Return a list of available models.
        """
        models = []
        for model_id, model in self.models.items():
            models.append({"id": model_id})
        return models

    async def maybe_load_model(self, model_id_to_load):
        """
        Very naive method to ensure a single model is loaded for now.
        """
        for model_id, model in self.models.items():
            if model_id != model_id_to_load:
                model.unload()


@dataclass
class QueuedRequest:
    inputs: any
    settings: dict
    is_chat: bool
    chat_template_kwargs: Optional[dict]
    future: asyncio.Future
    timestamp: float


class Model(object):
    """
    Represents a single model in the server.
    """

    def __init__(
        self,
        model_id=None,
        model_path=None,
        preload=False,
        models_root=None,
        model_type=False,
        pre_config={},
    ):
        self.loaded = False
        self.engine = None
        self.model_id = model_id
        self.preload = preload
        self.models_root = models_root
        self.model_path = model_path
        self.local_path = None
        self.model_type = model_type
        self.pre_config = pre_config
        self.request_queue = asyncio.Queue()
        self.batch_size = pre_config.get("batch_size", 8)
        self.batch_timeout = pre_config.get("batch_timeout", 0.1)
        self.processing_task = None

    def get_config(self):
        """
        Instanciate the configuration for the model.
        """
        # transforms and inference settings are retrieved from the model config for now
        self.config = PredictConfig(
            src="dummy",
            model_path=self.local_path,
            # TODO improve this
            gpu_ranks=[0],
            world_size=1,
            **self.pre_config,
        )

    async def _process_batch(self, batch):
        """
        Process a batch of requests together.
        """
        try:
            # Helper function to make settings hashable via JSON
            def settings_key(settings, is_chat):
                """Create a hashable key from settings."""
                return (json.dumps(settings, sort_keys=True), is_chat)

            # Group by settings and is_chat flag (only batch compatible requests)
            groups = {}
            for req in batch:
                key = settings_key(req.settings, req.is_chat)
                if key not in groups:
                    groups[key] = []
                groups[key].append(req)

            # Process each group
            for (settings_json, is_chat), reqs in groups.items():
                settings = json.loads(settings_json)

                # Collect all inputs
                all_inputs = []
                request_boundaries = []  # Track (start_idx, end_idx) for each request

                for req in reqs:
                    start_idx = len(all_inputs)

                    if is_chat:
                        # Chat mode: single input per request
                        all_inputs.append(
                            self.apply_chat_template(
                                req.inputs,
                                **(req.chat_template_kwargs or {}),
                            )
                        )
                        end_idx = len(all_inputs)
                    elif isinstance(req.inputs, str):
                        # Single string input
                        all_inputs.append(req.inputs)
                        end_idx = len(all_inputs)
                    elif isinstance(req.inputs, list):
                        # Multiple inputs in a single request
                        all_inputs.extend(req.inputs)
                        end_idx = len(all_inputs)
                    else:
                        # Fallback: treat as single input
                        all_inputs.append(req.inputs)
                        end_idx = len(all_inputs)

                    request_boundaries.append((start_idx, end_idx))

                # Run batched inference in thread pool to avoid blocking event loop
                scores, _, preds = await asyncio.get_event_loop().run_in_executor(
                    None, lambda s=settings: self.engine.infer_list(all_inputs, settings=s)
                )

                # Distribute results back to individual requests using boundaries
                for req, (start_idx, end_idx) in zip(reqs, request_boundaries):
                    req_scores = scores[start_idx:end_idx]
                    req_preds = preds[start_idx:end_idx]
                    req.future.set_result((req_scores, req_preds))

        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            # Set exception for all requests in batch
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(e)

    async def batch_processor(self):
        """
        Continuously collect and process requests in batches.
        """
        while True:
            batch = []
            deadline = None

            # Get first request (blocking)
            try:
                req = await self.request_queue.get()
                batch.append(req)
                deadline = time.time() + self.batch_timeout
            except Exception:
                continue

            # Collect more requests until timeout or batch full
            while len(batch) < self.batch_size and time.time() < deadline:
                try:
                    timeout = max(0, deadline - time.time())
                    req = await asyncio.wait_for(self.request_queue.get(), timeout=timeout)
                    batch.append(req)
                except asyncio.TimeoutError:
                    break

            # Process batch
            await self._process_batch(batch)

    def maybe_retrieve_model(self):
        """
        Download the model if it's not available locally.
        """
        from huggingface_hub import HfApi, snapshot_download

        hf_api = HfApi()
        try:
            hf_api.model_info(self.model_path)
        except Exception:
            self.local_path = os.path.expandvars(self.model_path)
        else:
            self.local_path = os.path.expandvars(os.path.join(self.models_root, self.model_path))
            logger.info(f"Downloading {self.model_path} from huggingface, " f"to local directory {self.local_path}")
            snapshot_download(repo_id=self.model_path, local_dir=self.local_path)

    async def ensure_batch_processor(self):
        """
        Start batch processor if not already running.
        Must be called from async context.
        """
        if self.processing_task is None or self.processing_task.done():
            self.processing_task = asyncio.create_task(self.batch_processor())
            logger.info(f"Started batch processor for model {self.model_id}")

    def load(self):
        """
        Create the inference engine.
        """
        self.maybe_retrieve_model()
        self.get_config()
        self.engine = InferenceEnginePY(self.config)
        self.loaded = True
        logger.info(f"Loaded model {self.model_id} from: {self.model_path}")

    def unload(self):
        """
        Not super clean, we might want to do better some day...
        """
        # Cancel batch processor if running
        if self.processing_task is not None and not self.processing_task.done():
            self.processing_task.cancel()
            self.processing_task = None

        # Clear any pending requests
        while not self.request_queue.empty():
            try:
                req = self.request_queue.get_nowait()
                if not req.future.done():
                    req.future.set_exception(Exception("Model unloaded"))
            except Exception:
                break
        del self.engine
        gc.collect()
        torch.cuda.empty_cache()
        self.engine = None
        self.loaded = False
        logger.info(f"Unloaded model {self.model_id}")

    def apply_chat_template(self, inputs, tools=None, tool_choice=None):
        """
        Render the model input based on the model chat template
        and the request inputs.
        """

        def raise_exception(message):
            raise TemplateError(message)

        chat_template = self.config.chat_template
        if chat_template is None:
            # Fall back to a standalone chat_template.jinja file in the model
            # directory (used by some modern HF models that don't embed the
            # template in tokenizer_config.json or config.json).
            jinja_file = os.path.join(self.local_path, "chat_template.jinja")
            if os.path.exists(jinja_file):
                with open(jinja_file, encoding="utf-8") as f:
                    chat_template = f.read()
            else:
                raise TemplateError(
                    f"Model '{self.model_id}' has no chat_template configured. "
                    "Set chat_template in the model's inference config or provide "
                    "a chat_template.jinja file in the model directory."
                )
        # Modern HuggingFace models store chat_template as a list of named
        # templates, e.g. [{"name": "default", "template": "..."}, ...].
        # Extract the "default" entry, or fall back to the first entry.
        if isinstance(chat_template, list):
            # Use .get() to safely handle list items that may lack a "template" key.
            template_str = next(
                (t.get("template") for t in chat_template if isinstance(t, dict) and t.get("name") == "default"),
                None,
            )
            if template_str is None and chat_template:
                first = chat_template[0]
                template_str = (
                    first.get("template") if isinstance(first, dict) else (first if isinstance(first, str) else None)
                )
            if not isinstance(template_str, str):
                raise TemplateError(f"Model '{self.model_id}': chat_template list contains no usable template string.")
            chat_template = template_str

        # Guard against any remaining non-string value (e.g. dict or bytes from
        # unexpected config formats) to give a clear error instead of a cryptic
        # "Can't compile non template nodes" TypeError from Jinja2.
        if not isinstance(chat_template, str):
            raise TemplateError(
                f"Model '{self.model_id}': chat_template has unexpected type "
                f"'{type(chat_template).__name__}' — expected a string. "
                "Check the model's config.json or chat_template.jinja file."
            )

        jinja_env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        jinja_env.globals["raise_exception"] = raise_exception
        template = jinja_env.from_string(chat_template)
        rendered_output = template.render(
            **{
                "messages": inputs,
                "tools": tools,
                "tool_choice": tool_choice,
                "bos_token": "",  # handled in numericalize
                "add_generation_prompt": True,
            }
        )
        return rendered_output

    async def infer_async(self, inputs, settings={}, is_chat=False, chat_template_kwargs=None):
        """
        Queue inference request and wait for result.
        """
        # Ensure model is loaded (sync operation)
        if not self.loaded:
            self.load()

        # Ensure batch processor is running (async operation)
        await self.ensure_batch_processor()

        future = asyncio.Future()
        req = QueuedRequest(
            inputs=inputs,
            settings=settings,
            is_chat=is_chat,
            chat_template_kwargs=chat_template_kwargs,
            future=future,
            timestamp=time.time(),
        )
        await self.request_queue.put(req)
        return await future


def create_app(config_file):
    """
    Create and configure the FastAPI application.
    """
    app = FastAPI(
        title="Eole Inference Server",
        version=eole.__version__,
        summary="A simple inference server to expose various models.",
        description="",  # TODO
    )

    server = Server()
    server.start(config_file)

    @app.get("/")
    def root(request: Request):
        """
        Root endpoint returning HTML content to help users find the docs.
        """
        html_content = f"""
        <html>
            <head>
                <title>Eole Server</title>
            </head>
            <body>
                <h1>Eole Server</h1>
                <p>Probably not what you're looking for.</p>
                <p>API docs --> <a href="{request.url}docs">{request.url}docs</a>.</p>
            </body>
        </html>
        """
        return HTMLResponse(content=html_content, status_code=200)

    @app.get("/models")
    def models():
        """
        Return available models currently exposed.
        """
        models = server.available_models()
        out = {"models": models}
        return out

    @app.post("/unload_model")
    def unload_model(model_id):
        """
        Unload a specific model.
        """
        server.models[model_id].unload()

    @app.get("/health")
    def health():
        """
        Health check endpoint.
        """
        out = {}
        out["status"] = STATUS_OK
        return out

    @app.post("/infer", response_model=TextResponse)
    async def infer(
        request: Union[TextRequest, ChatRequest] = Body(
            openapi_examples={
                "text_request": {
                    "summary": "Text Request Example",
                    "description": "A sample text request",
                    "value": TextRequest.Config.json_schema_extra["example"],
                },
                "chat_request": {
                    "summary": "Chat Request Example",
                    "description": "A sample chat request",
                    "value": ChatRequest.Config.json_schema_extra["example"],
                },
            },
        ),
    ):
        """
        Run inference on the given input.
        """
        if isinstance(request, TextRequest):
            inputs = request.inputs if isinstance(request.inputs, list) else [request.inputs]
        else:  # ChatRequest
            inputs = request.messages
        model_id = request.model
        # automatically grab anything that is not model/inputs
        # (we could probably rely on pydantic model once properly implemented)
        non_settings_keys = ["inputs", "messages", "model"]
        settings = {k: v for k, v in request.model_dump().items() if k not in non_settings_keys}

        await server.maybe_load_model(model_id)
        scores, preds = await server.models[model_id].infer_async(
            inputs,
            settings=settings,
            is_chat=isinstance(request, ChatRequest),
        )
        response = {"predictions": preds, "scores": scores}
        return response

    @app.post("/v1/chat/completions", response_model=OpenAIChatResponse)
    @app.post("/openai/chat/completions", response_model=OpenAIChatResponse)  # Alternative path
    async def openai_chat(request: OpenAIChatRequest):
        """
        OpenAI-compatible chat completions endpoint.
        This allows the server to be used as a drop-in replacement
        for OpenAI or other LLM APIs.
        """
        try:
            # Check if n > 1 (multiple completions not supported in simple implementation)
            if request.n > 1:
                from fastapi.responses import JSONResponse

                return JSONResponse(
                    status_code=400,
                    content={
                        "error": {
                            "message": "Multiple completions (n > 1) not yet supported",
                            "type": "invalid_request_error",
                            "code": "multiple_completions_not_supported",
                        }
                    },
                )

            # Convert OpenAI messages to the format expected by your engine
            messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
            chat_template_kwargs = {"tools": request.tools, "tool_choice": request.tool_choice}

            # Map OpenAI parameters to Eole settings
            settings = map_openai_to_eole_settings(request)

            # Ensure model is loaded
            resolved_model_id, response_model_id = _resolve_model_id(server, request.model)
            if resolved_model_id is None:
                from fastapi.responses import JSONResponse

                return JSONResponse(
                    status_code=404,
                    content={
                        "error": {
                            "message": f"Model '{request.model}' not found",
                            "type": "invalid_request_error",
                            "code": "model_not_found",
                        }
                    },
                )

            await server.maybe_load_model(resolved_model_id)

            # ----------------------------------------------------------------
            # Streaming path
            # ----------------------------------------------------------------
            if request.stream:
                model_obj = server.models[resolved_model_id]
                if not model_obj.loaded:
                    model_obj.load()

                chat_input = model_obj.apply_chat_template(messages, **chat_template_kwargs)
                completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
                created_ts = int(time.time())

                async def _stream_sse():
                    """Async generator that yields SSE-formatted data lines."""
                    loop = asyncio.get_event_loop()

                    # Run the synchronous streaming generator in a thread-pool
                    # executor so it doesn't block the event loop.
                    import concurrent.futures

                    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

                    # The engine's infer_list_stream is a synchronous generator.
                    # We iterate it inside run_in_executor via a queue-bridging
                    # pattern to keep the event loop responsive.
                    chunk_queue: asyncio.Queue = asyncio.Queue()

                    def _produce():
                        try:
                            for chunk in model_obj.engine.infer_list_stream(chat_input, settings=settings):
                                loop.call_soon_threadsafe(chunk_queue.put_nowait, chunk)
                        except Exception as exc:  # noqa: BLE001
                            loop.call_soon_threadsafe(chunk_queue.put_nowait, exc)
                        finally:
                            loop.call_soon_threadsafe(chunk_queue.put_nowait, None)

                    executor.submit(_produce)

                    # First chunk: role announcement
                    first_chunk = OpenAIStreamChunk(
                        id=completion_id,
                        created=created_ts,
                        model=response_model_id,
                        choices=[
                            OpenAIStreamChoice(
                                index=0,
                                delta=OpenAIStreamDelta(role="assistant"),
                            )
                        ],
                    )
                    yield f"data: {first_chunk.model_dump_json()}\n\n"

                    while True:
                        item = await chunk_queue.get()
                        if item is None:
                            break
                        if isinstance(item, Exception):
                            raise item
                        item = _normalize_generated_text(item)
                        content_chunk = OpenAIStreamChunk(
                            id=completion_id,
                            created=created_ts,
                            model=response_model_id,
                            choices=[
                                OpenAIStreamChoice(
                                    index=0,
                                    delta=OpenAIStreamDelta(content=item),
                                )
                            ],
                        )
                        yield f"data: {content_chunk.model_dump_json()}\n\n"

                    # Final chunk with finish_reason
                    final_chunk = OpenAIStreamChunk(
                        id=completion_id,
                        created=created_ts,
                        model=response_model_id,
                        choices=[
                            OpenAIStreamChoice(
                                index=0,
                                delta=OpenAIStreamDelta(),
                                finish_reason="stop",
                            )
                        ],
                    )
                    yield f"data: {final_chunk.model_dump_json()}\n\n"
                    yield "data: [DONE]\n\n"

                return StreamingResponse(
                    _stream_sse(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "X-Accel-Buffering": "no",
                    },
                )

            # ----------------------------------------------------------------
            # Non-streaming path
            # ----------------------------------------------------------------
            # Run inference using chat mode
            scores, preds = await server.models[resolved_model_id].infer_async(
                inputs=messages,
                settings=settings,
                is_chat=True,
                chat_template_kwargs=chat_template_kwargs,
            )

            # Calculate token usage (rough estimation)
            # content can be None (tool-call-only message) or a list (multipart);
            # normalise to plain text for token counting.
            prompt_text = " ".join([_content_to_text(msg.content) for msg in request.messages])
            prompt_tokens = estimate_tokens(prompt_text)
            completion_text = preds[0][0] if preds and preds[0] else ""
            completion_text = _normalize_generated_text(completion_text)
            completion_tokens = estimate_tokens(completion_text)

            # Build OpenAI-compatible response
            response = OpenAIChatResponse(
                id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
                object="chat.completion",
                created=int(time.time()),
                model=response_model_id,
                choices=[
                    OpenAIChoice(
                        index=0,
                        message=OpenAIMessage(role="assistant", content=completion_text),
                        finish_reason="stop",  # You might want to detect "length" based on max_tokens
                    )
                ],
                usage=OpenAIUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                ),
            )

            return response

        except Exception as e:
            logger.error(f"Error in OpenAI chat endpoint: {e}")
            from fastapi.responses import JSONResponse

            return JSONResponse(
                status_code=500,
                content={"error": {"message": str(e), "type": "internal_error", "code": "internal_error"}},
            )

    @app.post("/v1/messages", response_model=ClaudeMessageResponse)
    @app.post("/anthropic/v1/messages", response_model=ClaudeMessageResponse)  # Alternative path
    async def claude_messages(request: ClaudeMessagesRequest):
        """
        Claude-compatible messages endpoint.
        """
        try:
            resolved_model_id, response_model_id = _resolve_model_id(server, request.model)
            if resolved_model_id is None:
                from fastapi.responses import JSONResponse

                return JSONResponse(
                    status_code=404,
                    content={
                        "type": "error",
                        "error": {"type": "not_found_error", "message": f"model '{request.model}' not found"},
                    },
                )

            messages = [{"role": msg.role, "content": _content_to_text(msg.content)} for msg in request.messages]
            if request.system is not None:
                messages = [{"role": "system", "content": _content_to_text(request.system)}] + messages
            chat_template_kwargs = {"tools": request.tools, "tool_choice": request.tool_choice}

            settings = map_claude_to_eole_settings(request)
            await server.maybe_load_model(resolved_model_id)

            if request.stream:
                model_obj = server.models[resolved_model_id]
                if not model_obj.loaded:
                    model_obj.load()

                chat_input = model_obj.apply_chat_template(messages, **chat_template_kwargs)
                message_id = f"msg_{uuid.uuid4().hex[:8]}"
                input_tokens = estimate_tokens(" ".join(_content_to_text(m["content"]) for m in messages))

                async def _stream_sse():
                    loop = asyncio.get_event_loop()
                    chunk_queue: asyncio.Queue = asyncio.Queue()

                    import concurrent.futures

                    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

                    def _produce():
                        try:
                            for chunk in model_obj.engine.infer_list_stream(chat_input, settings=settings):
                                loop.call_soon_threadsafe(chunk_queue.put_nowait, chunk)
                        except Exception as exc:  # noqa: BLE001
                            loop.call_soon_threadsafe(chunk_queue.put_nowait, exc)
                        finally:
                            loop.call_soon_threadsafe(chunk_queue.put_nowait, None)

                    executor.submit(_produce)

                    start_payload = {
                        "type": "message_start",
                        "message": {
                            "id": message_id,
                            "type": "message",
                            "role": "assistant",
                            "content": [],
                            "model": response_model_id,
                            "stop_reason": None,
                            "stop_sequence": None,
                            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
                        },
                    }
                    yield f"event: message_start\ndata: {json.dumps(start_payload)}\n\n"
                    yield (
                        "event: content_block_start\n"
                        'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n'
                    )

                    output_text = ""
                    while True:
                        item = await chunk_queue.get()
                        if item is None:
                            break
                        if isinstance(item, Exception):
                            raise item
                        item = _normalize_generated_text(item)
                        output_text += item
                        delta_payload = {
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": {"type": "text_delta", "text": item},
                        }
                        yield f"event: content_block_delta\ndata: {json.dumps(delta_payload)}\n\n"

                    output_tokens = estimate_tokens(output_text)
                    yield 'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n'
                    message_delta_payload = {
                        "type": "message_delta",
                        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                        "usage": {"output_tokens": output_tokens},
                    }
                    yield f"event: message_delta\ndata: {json.dumps(message_delta_payload)}\n\n"
                    yield 'event: message_stop\ndata: {"type":"message_stop"}\n\n'

                return StreamingResponse(
                    _stream_sse(),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                )

            scores, preds = await server.models[resolved_model_id].infer_async(
                inputs=messages,
                settings=settings,
                is_chat=True,
                chat_template_kwargs=chat_template_kwargs,
            )

            completion_text = preds[0][0] if preds and preds[0] else ""
            completion_text = _normalize_generated_text(completion_text)
            input_tokens = estimate_tokens(" ".join(_content_to_text(m["content"]) for m in messages))
            output_tokens = estimate_tokens(completion_text)

            return ClaudeMessageResponse(
                id=f"msg_{uuid.uuid4().hex[:8]}",
                type="message",
                role="assistant",
                content=[ClaudeResponseContent(type="text", text=completion_text)],
                model=response_model_id,
                stop_reason="end_turn",
                stop_sequence=None,
                usage=ClaudeUsage(input_tokens=input_tokens, output_tokens=output_tokens),
            )

        except Exception as e:
            logger.error(f"Error in Claude messages endpoint: {e}")
            from fastapi.responses import JSONResponse

            return JSONResponse(
                status_code=500,
                content={
                    "type": "error",
                    "error": {"type": "api_error", "message": str(e)},
                },
            )

    return app


@register_bin(name="serve")
class Serve(BaseBin):
    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            "--config",
            "-config",
            "-c",
            default="./server_conf.yaml",
            help="Path of server YAML config file.",
        )
        parser.add_argument("--host", type=str, default="0.0.0.0")
        parser.add_argument("--port", type=int, default="5000")

    @classmethod
    def run(cls, args):
        app = create_app(args.config)
        uvicorn.run(app=app, host=args.host, port=args.port, log_level="info")
