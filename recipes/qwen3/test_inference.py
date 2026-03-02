# flake8: noqa
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
        max_length=2048,
        gpu_ranks=[0],
        # quant_type and quant_layers are loaded automatically from the converted model metadata.
        # autoround_packing_format and autoround_sym are also loaded from metadata.
        compute_dtype="bf16",
        top_p=0.8,
        temperature=0.6,
        beam_size=1,
        seed=42,
        batch_size=1,
        batch_type="sents",
        report_time=True,
    )
    return config


def build_test_inputs():
    return [
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nExplain what quantization is in one sentence.<|im_end|>\n<|im_start|>assistant\n",
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
