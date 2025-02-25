""" Modules for prediction """

from eole.predict.translator import Translator
from eole.predict.generator import GeneratorLM
from eole.predict.encoder import Encoder

from eole.predict.beam_search import GNMTGlobalScorer
from eole.decoders.ensemble import load_test_model as ensemble_load_test_model
from eole.models.model import BaseModel


def get_infer_class(model_config):
    # might have more cases later
    if model_config.decoder is None:
        return Encoder
    elif model_config.encoder is None:
        return GeneratorLM
    elif model_config.encoder.encoder_type == "vision":
        return GeneratorLM
    else:
        return Translator


def build_predictor(config, device_id=0, report_score=True, logger=None):
    # right now config is a full (non nested) PredictConfig

    load_test_model = ensemble_load_test_model if len(config.model_path) > 1 else BaseModel.load_test_model

    vocabs, model, model_config = load_test_model(config, device_id)
    config.update(model=model_config)

    scorer = GNMTGlobalScorer.from_config(config)

    infer_class = get_infer_class(model_config)
    if infer_class is not None:
        translator = infer_class.from_config(
            model,
            vocabs,
            config,
            model_config,
            device_id=device_id,
            global_scorer=scorer,
            report_align=config.report_align,
            report_score=report_score,
            logger=logger,
        )
    else:
        raise NotImplementedError(f"No infer_class found for {model_config.__class__}")
    return translator
