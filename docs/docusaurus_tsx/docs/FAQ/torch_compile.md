# How to use torch.compile for fast inference?

EOLE supports [torch.compile](https://pytorch.org/docs/stable/torch.compile.html) for significantly accelerated inference, reaching speeds comparable to vLLM on GPU.

## Enabling torch.compile

Set the `EOLE_TORCH_COMPILE` environment variable before running inference:

```bash
export EOLE_TORCH_COMPILE=1
```

You can also control the compile mode with `EOLE_COMPILE_MODE`:

| Value | Description |
|-------|-------------|
| `0`   | Default compile (no CUDA graphs) |
| `1`   | Compile with `reduce-overhead` |
| `2`   | Compile + CUDA graph capture (fastest, requires fixed batch shapes) |
| `3`   | Compile + CUDA graph capture + persistent cache |

```bash
export EOLE_TORCH_COMPILE=1
export EOLE_COMPILE_MODE=2
eole predict --config inference.yaml --src input.txt --output output.txt
```

**Note**: The first run will take 60-90 seconds to compile. Subsequent runs will be significantly faster.

## InferenceEngine

EOLE provides an `InferenceEnginePY` class that manages the inference lifecycle, including:

- A single dedicated inference thread (`eole-inference`) for serialized GPU access
- Support for streaming token-by-token generation via `GenerationStreamer`
- Continuous batching via `ContinuousBatchingManager`

### Example usage (Python API)

```python
from eole.config.run import PredictConfig
from eole.inference_engine import InferenceEngine

config = PredictConfig(model_path="/path/to/model", gpu=0, compute_dtype="fp16")
engine = InferenceEngine(config)

results = engine.infer_list(["Hello, world!"])
for result in results:
    print(result)

engine.terminate()
```

### Streaming

```python
for token in engine.infer_list_stream(["Hello, world!"]):
    print(token, end="", flush=True)
```

## Benchmarks

See [benchmarks/genai/README.md](https://github.com/eole-nlp/eole/blob/main/benchmarks/genai/README.md) for detailed performance comparisons.
