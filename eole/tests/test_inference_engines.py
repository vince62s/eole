import json
import time
import yaml
from eole.config.run import PredictConfig
from argparse import ArgumentParser
from eole.utils.misc import use_gpu, set_random_seed


def evaluate(config, engine, input_file, out, method, model_type=None):
    print("# input file", input_file)
    run_results = {}
    # Build the translator (along with the model)
    if engine == "eole":
        print("Inference with py ...")
        from eole.inference_engine import InferenceEnginePY

        enginerunner = InferenceEnginePY(config)
    elif engine == "ct2":
        print("Inference with ct2 ...")
        from eole.inference_engine import InferenceEngineCT2

        config.src_subword_vocab = config.get_model_path() + "/vocabulary.json"
        enginerunner = InferenceEngineCT2(config, model_type=model_type)
    start = time.time()
    if method == "file":
        enginerunner.config.src = input_file
        scores, _, preds = enginerunner.infer_file()
    elif method == "list":
        src = open(input_file, ("r")).readlines()
        scores, _, preds = enginerunner.infer_list(src)
    enginerunner.terminate()
    dur = time.time() - start
    print(f"Time to generate {len(preds)} answers: {dur}s")
    if engine == "eole":
        scores = [[_score.cpu().numpy().tolist() for _score in _scores] for _scores in scores]
    run_results = {"pred_answers": preds, "score": scores, "duration": dur}
    output_filename = out + f"_{method}.json"
    with open(output_filename, "w") as f:
        json.dump(run_results, f, ensure_ascii=False, indent=2)


def main():
    # Required arguments
    parser = ArgumentParser()
    parser.add_argument("-model", help="Path to model.", required=True, type=str)
    parser.add_argument(
        "-model_type",
        help="Model task.",
        required=True,
        type=str,
        choices=["decoder", "encoder_decoder"],
    )
    parser.add_argument("-inference_config_file", help="Inference config file", required=True, type=str)
    parser.add_argument(
        "-engine",
        help="Engine",
        default="eole",
        required=False,
        type=str,
        choices=["eole", "ct2"],
    )
    parser.add_argument(
        "-input_file",
        help="File with formatted input examples.",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-out",
        help="Output filename.",
        required=True,
        type=str,
    )
    args = parser.parse_args()
    model = args.model
    inference_config_file = args.inference_config_file
    with open(inference_config_file) as f:
        config = yaml.safe_load(f.read())
    config = PredictConfig(**config)
    set_random_seed(config.seed, use_gpu(config))
    config.model_path = [model]

    evaluate(
        config,
        engine=args.engine,
        input_file=args.input_file,
        out=args.out,
        method="file",
        model_type=args.model_type,
    )
    evaluate(
        config,
        engine=args.engine,
        input_file=args.input_file,
        out=args.out,
        method="list",
        model_type=args.model_type,
    )


if __name__ == "__main__":
    main()
