# flake8: noqa
# Qwen3-8B-int4 inference test (autoround quantized model).
# Usage:
#   EOLE_MODEL_DIR=/path/to/models EOLE_TORCH_COMPILE=0 python recipes/qwen3/test_inference2.py
#
# quant_type, quant_layers, autoround_packing_format and autoround_sym are loaded
# automatically from the converted model metadata — do not set them manually here.
import os
from rich import print


def build_config():
    from eole.config.run import PredictConfig

    mydir = os.getenv("EOLE_MODEL_DIR")
    if mydir is None:
        raise RuntimeError("EOLE_MODEL_DIR environment variable is not set")

    config = PredictConfig(
        model_path=os.path.join(mydir, "qwen3-8B-int4"),
        src="dummy",
        max_length=4096,
        gpu_ranks=[0],
        # quant_type / quant_layers / autoround_packing_format / autoround_sym
        # are all read from the model's metadata at load time.
        compute_dtype="bf16",
        top_p=0.8,
        temperature=0.6,
        beam_size=1,
        seed=42,
        batch_size=2,
        batch_type="sents",
        report_time=True,
    )
    return config


def build_test_inputs():
    # Qwen3 thinking-mode chat format
    think_on = "<|im_start|>system\nYou are a helpful thinking assistant.<|im_end|>\n"
    think_off = "<|im_start|>system\nYou are a helpful assistant. /no_think<|im_end|>\n"
    user_tmpl = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    return [
        think_off + user_tmpl.format(prompt="What is the capital of France?"),
        think_off + user_tmpl.format(prompt="Explain what quantization is in one sentence."),
        think_on + user_tmpl.format(prompt="What is 17 * 23?"),
    ]


def postprocess_and_print(pred, test_input):
    for i in range(len(test_input)):
        print(f'{"#" * 40} example {i} {"#" * 40}')
        text = pred[i][0]
        print(text.replace("｟newline｠", "\n"))


def main():
    from eole.inference_engine import InferenceEnginePY

    config = build_config()
    engine = InferenceEnginePY(config)

    try:
        test_input = build_test_inputs()
        _, _, pred = engine.infer_list(test_input)

        postprocess_and_print(pred, test_input)

    finally:
        engine.terminate()


if __name__ == "__main__":
    main()
