import logging
import os

from .scorer import Scorer
from eole.scorers import register_scorer

logger = logging.getLogger(__name__)


class _CometBaseScorer(Scorer):
    """Base class for COMET-based scorers.

    At scoring time the training model is offloaded to CPU by the trainer
    (see :meth:`~eole.trainer.Trainer._eval_handler`), the COMET model is
    loaded onto the GPU, scoring is performed, and then the COMET model is
    explicitly deleted to free GPU memory before the training model is
    restored.
    """

    uses_gpu = True
    comet_model_name = None  # overridden by subclasses
    needs_ref = True  # False for reference-free (QE) models

    def __init__(self, config):
        super().__init__(config)
        # Allow overriding the model name/path via config
        self.model_name = getattr(config, "comet_model", None) or self.comet_model_name

    def compute_score(self, preds, texts_refs, texts_srcs=None):
        """Compute COMET score.

        Args:
            preds (list[str]): Model predictions (MT hypotheses).
            texts_refs (list[str]): Reference translations (unused for QE models).
            texts_srcs (list[str]): Source sentences.

        Returns:
            float: System-level COMET score.
        """
        try:
            from comet import download_model, load_from_checkpoint
        except ImportError:
            raise ImportError(
                "unbabel-comet is required for COMET scoring: pip install -e .[comet]"
            )

        import torch

        if not preds:
            return 0

        if texts_srcs is None:
            texts_srcs = [""] * len(preds)

        data = []
        if self.needs_ref:
            if texts_refs is None:
                texts_refs = [""] * len(preds)
            for src, mt, ref in zip(texts_srcs, preds, texts_refs):
                data.append({"src": src, "mt": mt, "ref": ref})
        else:
            for src, mt in zip(texts_srcs, preds):
                data.append({"src": src, "mt": mt})

        logger.info("Loading COMET model: %s", self.model_name)
        if os.path.exists(self.model_name):
            # Local checkpoint file or directory — load directly without downloading
            comet_model = load_from_checkpoint(self.model_name)
        else:
            # HuggingFace repo ID — download (cached) then load
            model_path = download_model(self.model_name)
            comet_model = load_from_checkpoint(model_path)

        gpus = 1 if torch.cuda.is_available() else 0
        batch_size = getattr(self.config, "comet_batch_size", 64)

        try:
            result = comet_model.predict(
                data,
                batch_size=batch_size,
                gpus=gpus,
                num_workers=0,
                progress_bar=False,
            )
            score = result.system_score
        finally:
            # Unload COMET model to free GPU memory before returning
            del comet_model
            torch.cuda.empty_cache()

        return score


@register_scorer(metric="COMET")
class CometScorer(_CometBaseScorer):
    """COMET scorer using the wmt22-comet-da reference-based model."""

    comet_model_name = "Unbabel/wmt22-comet-da"
    needs_ref = True


@register_scorer(metric="COMET-KIWI")
class CometKiwiScorer(_CometBaseScorer):
    """COMET-KIWI scorer using the wmt22-cometkiwi-da reference-free (QE) model."""

    comet_model_name = "Unbabel/wmt22-cometkiwi-da"
    needs_ref = False
