import os
import json
import queue
import threading
from typing import List, Tuple, Optional, Dict, Any
from time import time
import torch

from eole.constants import CorpusTask, DefaultTokens, ModelType, InferenceConstants
from eole.inputters.dynamic_iterator import build_dynamic_dataset_iter
from eole.utils.logging import init_logger
from eole.utils.misc import get_device_type, configure_cuda_backends
from eole.transforms import get_transforms_cls, make_transforms, TransformPipe
from eole.predict import build_predictor  # moved to top-level (faster)


class InferenceEngine:
    """Wrapper Class to run Inference.

    Args:
        config: inference options
    """

    def __init__(self, config):
        self.config = config
        self.model_type = None
        configure_cuda_backends()

    def predict_batch(self, batch):
        """Predict a single batch. To be implemented by subclasses."""
        raise NotImplementedError

    def _predict(self, infer_iter, settings: Optional[Dict[str, Any]] = None):
        """Internal prediction method. To be implemented by subclasses."""
        raise NotImplementedError

    def _flatten_results(
        self, scores: List[List[float]], estims: Optional[List[List[float]]], preds: List[List[str]]
    ) -> Tuple[List[float], List[float], List[str]]:
        """Flatten nested prediction results."""
        flatten_preds = [text for sublist in preds for text in sublist]
        flatten_scores = [score for sublist in scores for score in sublist]

        if estims is not None:
            flatten_estims = [estim for sublist in estims for estim in sublist]
        else:
            flatten_estims = [InferenceConstants.DEFAULT_ESTIM_VALUE] * len(flatten_scores)

        return flatten_scores, flatten_estims, flatten_preds

    def _write_predictions_to_file(self, scores: List[float], estims: List[float], preds: List[str], output_path: str):
        """Write predictions to output file with optional scores."""
        import codecs

        with codecs.open(output_path, "w+", "utf-8") as out_file:
            if self.config.with_score:
                if len(scores) > 0:
                    lines = [
                        f"{pred}{InferenceConstants.OUTPUT_DELIMITER}{score}"
                        f"{InferenceConstants.OUTPUT_DELIMITER}{estim}"
                        for pred, score, estim in zip(preds, scores, estims)
                    ]
                else:
                    lines = [str(estim) for estim in estims]
                out_file.write("\n".join(lines) + "\n")
            else:
                out_file.write("\n".join(preds) + "\n")

    def infer_file(self) -> Tuple[List[List[float]], Optional[List[List[float]]], List[List[str]]]:
        """File inference. Source file must be the config.src argument."""
        if self.config.world_size <= 1:
            infer_iter = self._build_inference_iterator()
            scores, estims, preds = self._predict(infer_iter)
        else:
            scores, estims, preds = self.infer_file_parallel()

        flatten_scores, flatten_estims, flatten_preds = self._flatten_results(scores, estims, preds)
        self._write_predictions_to_file(flatten_scores, flatten_estims, flatten_preds, self.config.output)

        return scores, estims, preds

    def _build_inference_iterator(self, src: Optional[List[str]] = None, tgt: Optional[str] = None):
        """Build dynamic dataset iterator for inference."""
        return build_dynamic_dataset_iter(
            self.config,
            self.transforms,
            self.vocabs,
            task=CorpusTask.INFER,
            src=src,
            tgt=tgt,
            device_id=self.device_id,
            model_type=self.model_type,
        )

    def infer_list(
        self, src: List[str], settings: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[List[float]], Optional[List[List[float]]], List[List[str]]]:
        """List of strings inference."""
        settings = settings or {}

        if self.config.world_size <= 1:
            infer_iter = self._build_inference_iterator(src=src)
            scores, estims, preds = self._predict(infer_iter, settings=settings)
        else:
            scores, estims, preds = self.infer_list_parallel(src, settings=settings)

        return scores, estims, preds

    def infer_list_stream(self, src: str, settings: Optional[Dict[str, Any]] = None):
        """Stream inference results for a single input string.

        This is a generator that yields decoded text chunks as they are
        produced by the model, enabling a chatbot-style streaming interface
        instead of waiting for the full response.

        Only supported for decoder-only (LM) models (``GeneratorLM``).
        Streaming with encoder-decoder or encoder-only models falls back to
        returning the complete prediction as a single chunk at the end.

        Args:
            src (str): Single input string to run inference on.
            settings (dict, optional): Override inference settings
                (e.g. ``temperature``, ``max_length``).

        Yields:
            str: Decoded text chunks, one per generated token (or
            slightly larger chunks when the detokenizer defers output
            to avoid partial multi-byte / subword-piece artefacts).

        Raises:
            NotImplementedError: If world_size > 1 (parallel mode).

        Example::

            engine = InferenceEnginePY(config)
            for chunk in engine.infer_list_stream("Tell me a joke"):
                print(chunk, end="", flush=True)
        """
        raise NotImplementedError

    def infer_file_parallel(self, settings: Optional[Dict[str, Any]] = None):
        """File inference in multiprocessing with partitioned models."""
        raise NotImplementedError("The inference in multiprocessing with partitioned models is not implemented.")

    def infer_list_parallel(self, src: List[str], settings: Optional[Dict[str, Any]] = None):
        """List inference in multiprocessing with partitioned models."""
        raise NotImplementedError("The inference in multiprocessing with partitioned models is not implemented.")

    def _score(self, infer_iter, settings: Optional[Dict[str, Any]] = None):
        """Internal scoring method. To be implemented by subclasses."""
        raise NotImplementedError

    def score_file(self, settings: Optional[Dict[str, Any]] = None):
        """File scoring. Source file must be the config.src argument."""
        if self.config.world_size <= 1:
            infer_iter = self._build_inference_iterator(tgt=self.config.src)
            score_results = self._score(infer_iter, settings=settings)
        else:
            score_results = self.score_file_parallel(settings=settings)

        return score_results

    def score_list(self, src: List[str], settings: Optional[Dict[str, Any]] = None):
        """List of strings scoring."""
        if self.config.world_size <= 1:
            infer_iter = self._build_inference_iterator(src=src, tgt=src)
            score_results = self._score(infer_iter, settings=settings)
        else:
            score_results = self.score_list_parallel(src, settings=settings)

        return score_results

    def score_file_parallel(self, settings: Optional[Dict[str, Any]] = None):
        """File scoring in parallel. To be implemented by subclasses."""
        raise NotImplementedError

    def score_list_parallel(self, src: List[str], settings: Optional[Dict[str, Any]] = None):
        """List scoring in parallel. To be implemented by subclasses."""
        raise NotImplementedError

    def terminate(self):
        """Terminate the inference engine and cleanup resources."""
        pass


# Seconds to wait when joining the inference thread during terminate().
_INFERENCE_THREAD_SHUTDOWN_TIMEOUT_SECS = 5


class InferenceEnginePY(InferenceEngine):
    """Inference engine subclass to run inference with `predict.py`.

    Args:
        config: inference options
    """

    def __init__(self, config):
        super().__init__(config)
        self.logger = init_logger(config.log_file)
        t0 = time()

        if config.world_size > 1:
            self.logger.info("Init multi-process mode")
            self._initialize_multiprocessing()
        else:
            self.logger.info("Init single process mode")
            self._initialize_single_process()

        self.logger.info("Build and loading model took %.2f sec." % (time() - t0))

    def _initialize_multiprocessing(self):
        """Initialize multiprocessing components for parallel inference."""
        from eole.utils.distributed import ErrorHandler
        from eole.utils.distributed_workers import spawned_infer

        mp = torch.multiprocessing.get_context("spawn")

        # Create error handling
        self.error_queue = mp.SimpleQueue()
        self.error_handler = ErrorHandler(self.error_queue)

        # Create queues and processes
        self.queue_instruct = []
        self.queue_settings = []
        self.queue_result = []
        self.procs = []

        for device_id in range(self.config.world_size):
            self.queue_instruct.append(mp.Queue())
            self.queue_settings.append(mp.Queue())
            self.queue_result.append(mp.Queue())

            proc = mp.Process(
                target=spawned_infer,
                args=(
                    self.config,
                    device_id,
                    self.error_queue,
                    self.queue_instruct[device_id],
                    self.queue_result[device_id],
                ),
                daemon=False,
            )
            self.procs.append(proc)
            proc.start()
            self.error_handler.add_child(proc.pid)

    def _initialize_single_process(self):
        """Initialize components for single-process inference."""

        if len(self.config.gpu_ranks) > 0:
            self.device_id = self.config.gpu_ranks[0]
        else:
            self.device_id = InferenceConstants.DEFAULT_DEVICE_ID

        # Build predictor and transforms
        self.predictor = build_predictor(self.config, self.device_id, logger=self.logger, report_score=True)

        self.transforms_cls = get_transforms_cls(self.config._all_transform)
        self.vocabs = self.predictor.vocabs
        self.transforms = make_transforms(self.config, self.transforms_cls, self.vocabs)
        self.transform_pipe = TransformPipe.build_from(self.transforms.values())

        # Single dedicated inference thread to serialize all prediction calls.
        # This prevents concurrent access to the shared predictor / KV cache
        # and is required for torch.compile CUDA-graph compatibility (graphs
        # must be replayed on the same OS thread they were captured on).
        # The queue is intentionally unbounded: the server already throttles
        # concurrency via its own batch processor, and blocking callers here
        # would stall asyncio tasks rather than providing useful backpressure.
        self._infer_queue: queue.Queue = queue.Queue()
        self._infer_thread = threading.Thread(
            target=self._inference_loop,
            daemon=True,
            name="eole-inference",
        )
        self._infer_thread.start()

    def _inference_loop(self):
        """Main loop of the dedicated inference thread.

        Reads tasks submitted via :meth:`_submit_to_infer_thread` or
        :meth:`infer_list_stream` from the internal queue and executes them
        one at a time, guaranteeing that the underlying model and KV cache
        are never accessed concurrently.  A ``None`` sentinel stops the loop
        (sent by :meth:`terminate`).
        """
        while True:
            task = self._infer_queue.get()
            if task is None:
                # Shutdown sentinel received – exit the loop cleanly.
                break
            func, done_event, result_holder = task
            try:
                result = func()
                result_holder.append(result)
            except Exception as exc:  # noqa: BLE001
                # Broad catch is intentional: we must capture *any* exception
                # so it can be re-raised in the calling thread by
                # _submit_to_infer_thread (or, for streaming, via
                # exception_holder in infer_list_stream).
                result_holder.append(exc)
            finally:
                done_event.set()

    def _submit_to_infer_thread(self, func) -> Any:
        """Submit *func* to the inference thread and block until it finishes.

        Args:
            func: A zero-argument callable to execute on the inference thread.

        Returns:
            The return value of *func*.

        Raises:
            Exception: Any exception raised by *func* is re-raised in the
                calling thread.
        """
        done_event = threading.Event()
        result_holder: List[Any] = []
        self._infer_queue.put((func, done_event, result_holder))
        done_event.wait()
        if result_holder and isinstance(result_holder[0], Exception):
            raise result_holder[0]
        return result_holder[0] if result_holder else None

    @torch.inference_mode()
    def _predict_impl(
        self, infer_iter, settings: Optional[Dict[str, Any]] = None, streamer=None
    ) -> Tuple[List[List[float]], Optional[List[List[float]]], List[List[str]]]:
        """Run prediction on the inference iterator.

        Must be called on the dedicated inference thread (via
        :meth:`_submit_to_infer_thread` or from within the
        :meth:`infer_list_stream` task closure).
        """
        settings = settings or {}
        self.predictor.update_settings(**settings)

        scores, estims, preds = self.predictor._predict(
            infer_iter,
            infer_iter.transforms,
            self.config.attn_debug,
            self.config.align_debug,
            streamer=streamer,
        )

        return scores, estims, preds

    def _predict(
        self, infer_iter, settings: Optional[Dict[str, Any]] = None, streamer=None
    ) -> Tuple[List[List[float]], Optional[List[List[float]]], List[List[str]]]:
        """Submit a prediction task to the inference thread and wait for the result.

        All callers (``infer_list``, ``infer_file``, …) route through here so
        that the underlying model is never accessed from more than one thread
        at a time.
        """
        return self._submit_to_infer_thread(
            lambda: self._predict_impl(infer_iter, settings=settings, streamer=streamer)
        )

    def _score_impl(self, infer_iter, settings: Optional[Dict[str, Any]] = None):
        """Run scoring on the inference iterator.

        Must be called on the dedicated inference thread.
        """
        settings = settings or {}
        self.predictor.update_settings(**settings)

        self.predictor.with_scores = True
        self.predictor.return_gold_log_probs = True

        return self.predictor._score(infer_iter)

    def _score(self, infer_iter, settings: Optional[Dict[str, Any]] = None):
        """Submit a scoring task to the inference thread and wait for the result."""
        return self._submit_to_infer_thread(lambda: self._score_impl(infer_iter, settings=settings))

    def infer_list_stream(self, src: str, settings: Optional[Dict[str, Any]] = None):
        """Stream inference results for a single input string.

        Enqueues a generation task on the dedicated inference thread and
        yields decoded text chunks as they are produced token by token.
        Requests are processed in FIFO order; if another request is already
        being generated the caller will wait (without blocking the event loop
        when invoked from an ``async`` context via a thread-pool executor)
        until it reaches the front of the queue.

        The ``started`` event ensures the streamer's per-token timeout only
        begins once the inference thread actually starts processing this
        request, preventing spurious timeouts when many requests are queued.

        Only supported for single-process mode (``world_size <= 1``) and
        decoder-only (LM) models.

        Args:
            src (str): A single input string.
            settings (dict, optional): Override inference settings such as
                ``temperature``, ``max_length``, ``top_k``, ``top_p``.

        Yields:
            str: Decoded text chunks produced by the model.

        Raises:
            NotImplementedError: If called when ``world_size > 1``.

        Example::

            engine = InferenceEnginePY(config)
            for chunk in engine.infer_list_stream("Tell me a joke"):
                print(chunk, end="", flush=True)
            print()
        """
        from eole.predict.streamer import GenerationStreamer

        if self.config.world_size > 1:
            raise NotImplementedError("infer_list_stream is not supported in parallel (world_size > 1) mode.")

        settings = settings or {}

        streamer = GenerationStreamer(
            vocabs=self.vocabs,
            transform_pipe=self.transform_pipe,
        )

        # Event set by the inference thread as soon as it starts this task,
        # so the streamer's per-token timeout doesn't count time spent waiting
        # in the queue behind other requests.
        started = threading.Event()
        exception_holder: List[Exception] = []

        def _run():
            # Signal that we have dequeued this task and are about to generate.
            started.set()
            try:
                infer_iter = self._build_inference_iterator(src=[src])
                self._predict_impl(infer_iter, settings=settings, streamer=streamer)
            except Exception as exc:  # noqa: BLE001
                # Broad catch is intentional: end the streamer so the consumer
                # is unblocked, then surface the error via exception_holder
                # after yield-from completes.
                self.logger.error(f"Streaming inference failed: {exc}")
                exception_holder.append(exc)
                streamer.end()

        done_event = threading.Event()
        self._infer_queue.put((_run, done_event, []))

        # Wait until the inference thread actually starts our task before
        # entering the streamer iterator (avoids premature timeout).
        started.wait()

        yield from streamer

        # Ensure the inference thread has fully finished before returning so
        # that callers can rely on the generator being exhausted when done.
        done_event.wait()

        if exception_holder:
            raise exception_holder[0]

    def _distribute_parallel_task(self, task_name: str, task_arg: Any, settings: Optional[Dict[str, Any]] = None):
        """Distribute a task to all parallel workers.

        Args:
            task_name: Name of the task to execute
            task_arg: Argument to pass to the task
            settings: Settings dictionary to send to workers
        """
        assert self.config.world_size > 1, "World size must be greater than 1."

        settings = settings or {}

        for device_id in range(self.config.world_size):
            self.queue_instruct[device_id].put((task_name, task_arg))
            self.queue_settings[device_id].put(settings)

    def _collect_parallel_results(self, num_results_per_worker: int = 1) -> List[Any]:
        """Collect results from all parallel workers.

        Args:
            num_results_per_worker: Number of results to collect from each worker

        Returns:
            List of results from all workers
        """
        results = []
        for device_id in range(self.config.world_size):
            worker_results = [self.queue_result[device_id].get() for _ in range(num_results_per_worker)]
            results.append(worker_results if num_results_per_worker > 1 else worker_results[0])

        return results

    def _aggregate_inference_results(
        self, all_scores: List[List], all_estims: List[List], all_preds: List[List]
    ) -> Tuple[List, List, List]:
        """Aggregate inference results from parallel workers based on parallel mode.

        Args:
            all_scores: Scores from all workers
            all_estims: Estimations from all workers
            all_preds: Predictions from all workers

        Returns:
            Aggregated (scores, estims, preds) tuple
        """
        if self.config.parallel_mode == "data_parallel":
            # Flatten results from all workers
            scores = [item for worker_scores in all_scores for item in worker_scores]
            estims = [item for worker_estims in all_estims for item in worker_estims]
            preds = [item for worker_preds in all_preds for item in worker_preds]
            return scores, estims, preds
        else:
            # Tensor parallel: take only worker 0 results
            return all_scores[0], all_estims[0], all_preds[0]

    def _aggregate_score_results(self, results: List[Any]) -> Any:
        """Aggregate scoring results based on parallel mode.

        Args:
            results: Results from all workers

        Returns:
            Aggregated results
        """
        if self.config.parallel_mode == "data_parallel":
            # Flatten results from all workers
            return [item for worker_scores in results for item in worker_scores]
        else:
            # Tensor parallel: take only worker 0 results
            return results[0]

    def score_list_parallel(self, src: List[str], settings: Optional[Dict[str, Any]] = None):
        """Score a list of strings in parallel."""
        self._distribute_parallel_task(InferenceConstants.SCORE_LIST, src, settings)
        results = self._collect_parallel_results(num_results_per_worker=1)
        return self._aggregate_score_results(results)

    def score_file_parallel(self, settings: Optional[Dict[str, Any]] = None):
        """Score a file in parallel."""
        self._distribute_parallel_task(InferenceConstants.SCORE_FILE, self.config, settings)
        results = self._collect_parallel_results(num_results_per_worker=1)
        return self._aggregate_score_results(results)

    def infer_file_parallel(self, settings: Optional[Dict[str, Any]] = None):
        """Infer from file in parallel."""
        self._distribute_parallel_task(InferenceConstants.INFER_FILE, self.config, settings)
        results = self._collect_parallel_results(num_results_per_worker=3)

        # Unpack results (each worker returns 3 items: scores, estims, preds)
        scores = [worker_results[0] for worker_results in results]
        estims = [worker_results[1] for worker_results in results]
        preds = [worker_results[2] for worker_results in results]

        return self._aggregate_inference_results(scores, estims, preds)

    def infer_list_parallel(self, src: List[str], settings: Optional[Dict[str, Any]] = None):
        """Infer from list in parallel."""
        self._distribute_parallel_task(InferenceConstants.INFER_LIST, src, settings)
        results = self._collect_parallel_results(num_results_per_worker=3)

        # Unpack results (each worker returns 3 items: scores, estims, preds)
        scores = [worker_results[0] for worker_results in results]
        estims = [worker_results[1] for worker_results in results]
        preds = [worker_results[2] for worker_results in results]

        return self._aggregate_inference_results(scores, estims, preds)

    def terminate(self):
        """Terminate the inference thread and all worker processes."""
        if self.config.world_size <= 1:
            # Signal the single inference thread to stop and wait for it.
            if hasattr(self, "_infer_queue"):
                self._infer_queue.put(None)
            if hasattr(self, "_infer_thread"):
                self._infer_thread.join(timeout=_INFERENCE_THREAD_SHUTDOWN_TIMEOUT_SECS)
            return

        # Ask workers to stop
        for device_id in range(self.config.world_size):
            self.queue_instruct[device_id].put((InferenceConstants.STOP, ""))

        # Join workers
        for proc in self.procs:
            proc.join()


class InferenceEngineCT2(InferenceEngine):
    """Inference engine subclass to run inference with ctranslate2.

    Args:
        config: inference options
        model_type: Type of model (DECODER or ENCODER_DECODER)
    """

    def __init__(self, config, model_type=None):
        super().__init__(config)

        if model_type is None:
            raise ValueError("A model_type kwarg must be passed for CT2 models.")

        self.model_type = model_type
        self.logger = init_logger(config.log_file)

        self._validate_config()
        self._setup_device()
        self._initialize_transforms()
        self._load_predictor()

    def _validate_config(self):
        """Validate configuration for CT2 inference."""
        if self.config.world_size > 1:
            raise ValueError("CT2 inference does not support world_size > 1")

    def _setup_device(self):
        """Setup device configuration."""
        if len(self.config.gpu_ranks) > 0:
            self.device_id = self.config.gpu_ranks[0]
            self.device_index = self.config.gpu_ranks
            self.device = get_device_type()
        else:
            self.device_id = InferenceConstants.DEFAULT_DEVICE_ID
            self.device_index = 0
            self.device = "cpu"

    def _initialize_transforms(self):
        """Initialize transform classes."""
        self.transforms_cls = get_transforms_cls(self.config._all_transform)

    @property
    def ct2_model_path(self) -> str:
        """Get the ctranslate2 model path."""
        return os.path.join(self.config.get_model_path(), InferenceConstants.CT2_DIR)

    def _load_json(self, filename) -> Dict:
        """Load JSON file from path components."""
        file_path = os.path.join(self.ct2_model_path, filename)
        with open(file_path, "r") as f:
            return json.load(f)

    def _load_config_json(self) -> Dict:
        """Load CT2 config JSON."""
        return self._load_json("config.json")

    def _build_vocab_specials(self, ct2_json: Dict) -> Dict:
        """Build special tokens vocabulary from CT2 config."""
        return {
            "bos_token": ct2_json["bos_token"],
            "eos_token": ct2_json["eos_token"],
            "unk_token": ct2_json["unk_token"],
            "pad_token": ct2_json.get("pad_token", ct2_json["eos_token"]),
        }

    def _load_decoder_vocabs(self, vocabs: Dict):
        """Load vocabularies for decoder-only models."""
        import pyonmttok

        vocab = self._load_json("vocabulary.json")
        src_vocab = pyonmttok.build_vocab_from_tokens(vocab)

        self.config.share_vocab = True
        vocabs["src"] = src_vocab
        vocabs["tgt"] = src_vocab
        vocabs["decoder_start_token"] = ""

    def _load_encoder_decoder_vocabs(self, vocabs: Dict, ct2_json: Dict):
        """Load vocabularies for encoder-decoder models."""
        import pyonmttok

        vocabs["decoder_start_token"] = ct2_json["decoder_start_token"]

        if os.path.exists(os.path.join(self.ct2_model_path, "shared_vocabulary.json")):
            vocab = self._load_json("shared_vocabulary.json")
            src_vocab = pyonmttok.build_vocab_from_tokens(vocab)
            self.config.share_vocab = True
            vocabs["src"] = src_vocab
            vocabs["tgt"] = src_vocab
        else:
            vocab_src = self._load_json("source_vocabulary.json")
            vocab_tgt = self._load_json("target_vocabulary.json")

            src_vocab = pyonmttok.build_vocab_from_tokens(vocab_src)
            tgt_vocab = pyonmttok.build_vocab_from_tokens(vocab_tgt)

            self.config.share_vocab = False
            vocabs["src"] = src_vocab
            vocabs["tgt"] = tgt_vocab

    def _load_vocabs(self) -> Dict:
        """Load all vocabularies for CT2 model."""
        ct2_json = self._load_config_json()

        vocabs = {"specials": self._build_vocab_specials(ct2_json)}

        if self.model_type == ModelType.DECODER:
            self._load_decoder_vocabs(vocabs)
        else:
            self._load_encoder_decoder_vocabs(vocabs, ct2_json)

        return vocabs

    def _build_ct2_predictor(self):
        """Build CT2 generator or translator."""
        import ctranslate2

        if self.model_type == ModelType.DECODER:
            return ctranslate2.Generator(
                self.ct2_model_path,
                device=self.device,
                device_index=self.device_index,
            )
        else:
            return ctranslate2.Translator(
                self.ct2_model_path,
                device=self.device,
                device_index=self.device_index,
            )

    def _load_predictor(self):
        """Load CT2 predictor and vocabularies."""
        self.vocabs = self._load_vocabs()
        self.predictor = self._build_ct2_predictor()

        # Build transform pipe
        self.transforms = make_transforms(self.config, self.transforms_cls, self.vocabs)
        self.transforms_pipe = TransformPipe.build_from(self.transforms.values())

    def _prepare_input_tokens(self, batch) -> List[List[str]]:
        """Convert batch source tokens to predictor input format."""
        pad_token = self.vocabs["specials"].get("pad_token", DefaultTokens.PAD)
        pad_token_id = self.vocabs["src"].lookup_token(pad_token)

        input_tokens = []
        for i in range(batch["src"].size(0)):
            start_ids = batch["src"][i, :].cpu().numpy().tolist()
            tokens = [self.vocabs["src"].lookup_index(id) for id in start_ids if id != pad_token_id]
            input_tokens.append(tokens)

        return input_tokens

    def _get_generation_params(self, config) -> Dict[str, Any]:
        """Get common generation parameters for CT2."""
        return {
            "batch_type": "examples" if config.batch_type == "sents" else "tokens",
            "max_batch_size": config.batch_size,
            "beam_size": config.beam_size,
            "num_hypotheses": config.n_best,
            "return_scores": True,
            "sampling_topk": config.top_k,
            "sampling_topp": 1 if config.top_p == 0 else config.top_p,
            "sampling_temperature": config.temperature,
        }

    def _apply_reverse_transform(self, sequence: List[str]) -> str:
        """Apply reverse transform to a single predicted sequence.

        Args:
            sequence: Single sequence of tokens

        Returns:
            Transformed string
        """
        if self.transforms:
            return self.transforms_pipe.apply_reverse(sequence)
        else:
            return " ".join(sequence)

    def _predict_decoder(self, input_tokens: List[List[str]], config) -> Tuple:
        """Run prediction for decoder-only models."""
        params = self._get_generation_params(config)
        params["max_length"] = config.max_length
        params["include_prompt_in_result"] = False

        predicted_batch = self.predictor.generate_batch(start_tokens=input_tokens, **params)

        preds = [[self._apply_reverse_transform(nbest) for nbest in ex.sequences] for ex in predicted_batch]
        scores = [[nbest for nbest in ex.scores] for ex in predicted_batch]

        return scores, None, preds

    def _predict_encoder_decoder(self, input_tokens: List[List[str]], config) -> Tuple:
        """Run prediction for encoder-decoder models."""
        params = self._get_generation_params(config)
        params["max_decoding_length"] = config.max_length

        predicted_batch = self.predictor.translate_batch(input_tokens, **params)

        preds = [[self._apply_reverse_transform(nbest) for nbest in ex.hypotheses] for ex in predicted_batch]
        scores = [[nbest for nbest in ex.scores] for ex in predicted_batch]

        return scores, None, preds

    def predict_batch(self, batch) -> Tuple:
        """Predict a single batch using CT2."""
        input_tokens = self._prepare_input_tokens(batch)

        if self.model_type == ModelType.DECODER:
            return self._predict_decoder(input_tokens, self.config)
        elif self.model_type == ModelType.ENCODER_DECODER:
            return self._predict_encoder_decoder(input_tokens, self.config)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def _predict(
        self, infer_iter, settings: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[List[float]], Optional[List[List[float]]], List[List[str]]]:
        """Run prediction on inference iterator."""
        # TODO: convert settings to CT2 naming
        predictions = {"scores": [], "preds": [], "cid_line_number": []}

        for batch, bucket_idx in infer_iter:
            scores, _, preds = self.predict_batch(batch)
            predictions["scores"].extend(scores)
            predictions["preds"].extend(preds)
            predictions["cid_line_number"].extend(batch["cid_line_number"])

        # Sort by original line number
        sorted_data = sorted(
            zip(
                predictions["cid_line_number"],
                predictions["preds"],
                predictions["scores"],
            )
        )

        sorted_predictions = {
            "cid_line_number": [item[0] for item in sorted_data],
            "preds": [item[1] for item in sorted_data],
            "scores": [item[2] for item in sorted_data],
        }

        return sorted_predictions["scores"], None, sorted_predictions["preds"]

    def _score(self, infer_iter, settings: Optional[Dict[str, Any]] = None):
        """Scoring is not implemented for CT2."""
        raise NotImplementedError("The scoring with InferenceEngineCT2 is not implemented.")
