"""Module that contain iterator used for dynamic data."""

import torch
from itertools import cycle
from eole.constants import CorpusTask
from eole.inputters.text_corpus import get_corpora, build_corpora_iters
from eole.inputters.text_utils import (
    text_sort_key,
    transform_bucket,
    numericalize,
    tensorify,
)
from eole.config.data import Dataset
from eole.utils.logging import init_logger, logger
from eole.utils.misc import RandomShuffler, get_device
from torch.utils.data import DataLoader


class MixingStrategy(object):
    """Mixing strategy that should be used in Data Iterator."""

    def __init__(self, iterables, weights):
        """Initilize neccessary attr."""
        self._valid_iterable(iterables, weights)
        self.iterables = iterables
        self.weights = weights

    def _valid_iterable(self, iterables, weights):
        iter_keys = iterables.keys()
        weight_keys = weights.keys()
        if iter_keys != weight_keys:
            raise ValueError(f"keys in {iterables} & {weights} should be equal.")

    def __iter__(self):
        raise NotImplementedError


class SequentialMixer(MixingStrategy):
    """Generate data sequentially from `iterables` which is exhaustible."""

    def _iter_datasets(self):
        for ds_name, ds_weight in self.weights.items():
            for _ in range(ds_weight):
                yield ds_name

    def __iter__(self):
        for ds_name in self._iter_datasets():
            iterable = self.iterables[ds_name]
            yield from iterable


class WeightedMixer(MixingStrategy):
    """A mixing strategy that mix data weightedly and iterate infinitely."""

    def __init__(self, iterables, weights):
        super().__init__(iterables, weights)
        self._iterators = {}
        self._counts = {}
        for ds_name in self.iterables.keys():
            self._reset_iter(ds_name)

    def _logging(self):
        """Report corpora loading statistics."""
        msgs = []
        # patch to log stdout spawned processes of dataloader
        logger = init_logger()
        for ds_name, ds_count in self._counts.items():
            msgs.append(f"\t\t\t* {ds_name}: {ds_count}")
        logger.info("Weighted corpora loaded so far:\n" + "\n".join(msgs))

    def _reset_iter(self, ds_name):
        self._iterators[ds_name] = iter(self.iterables[ds_name])
        self._counts[ds_name] = self._counts.get(ds_name, 0) + 1
        self._logging()

    def _iter_datasets(self):
        for ds_name, ds_weight in self.weights.items():
            for _ in range(ds_weight):
                yield ds_name

    def __iter__(self):
        for ds_name in cycle(self._iter_datasets()):
            iterator = self._iterators[ds_name]
            try:
                item = next(iterator)
            except StopIteration:
                self._reset_iter(ds_name)
                iterator = self._iterators[ds_name]
                item = next(iterator)
            finally:
                yield item


class DynamicDatasetIter(torch.utils.data.IterableDataset):
    """Yield batch from (multiple) plain text corpus.

    Args:
        corpora (dict[str, ParallelCorpus]): collections of corpora to iterate;
        corpora_info (dict[str, dict]): corpora infos correspond to corpora;
        transforms (dict[str, Transform]): transforms may be used by corpora;
        vocabs (dict[str, Vocab]): vocab dict for convert corpora into Tensor;
        task (str): CorpusTask.TRAIN/VALID/INFER;
        batch_type (str): batching type to count on, choices=[tokens, sents];
        batch_size (int): numbers of examples in a batch;
        batch_size_multiple (int): make batch size multiply of this;
        data_type (str): input data type, currently only text;
        bucket_size (int): accum this number of examples in a dynamic dataset;
        bucket_size_init (int): initialize the bucket with this
        amount of examples;
        bucket_size_increment (int): increment the bucket
        size with this amount of examples;
        skip_empty_level (str): security level when encouter empty line;
        stride (int): iterate data files with this stride;
        offset (int): iterate data files with this offset.

    Attributes:
        sort_key (function): functions define how to sort examples;
        mixer (MixingStrategy): the strategy to iterate corpora.
    """

    def __init__(
        self,
        corpora,
        corpora_info,
        transforms,
        vocabs,
        task,
        batch_type,
        batch_size,
        batch_size_multiple,
        data_type="text",
        bucket_size=2048,
        bucket_size_init=-1,
        bucket_size_increment=0,
        device=torch.device("cpu"),
        skip_empty_level="warning",
        stride=1,
        offset=0,
        score_threshold=0,
        left_pad=False,
        model_type=None,
        image_patch_size=16,
        adapter=None,
    ):
        super(DynamicDatasetIter).__init__()
        self.corpora = corpora
        self.transforms = transforms
        self.vocabs = vocabs
        self.corpora_info = corpora_info
        self.task = task
        self.init_iterators = False
        self.batch_size = batch_size
        self.batch_type = batch_type
        self.batch_size_multiple = batch_size_multiple
        self.sort_key = text_sort_key
        self.bucket_size = bucket_size
        self.bucket_size_init = bucket_size_init
        self.bucket_size_increment = bucket_size_increment
        self.device = device
        if stride <= 0:
            raise ValueError(f"Invalid argument for stride={stride}.")
        self.stride = stride
        self.offset = offset
        self.score_threshold = score_threshold
        if skip_empty_level not in ["silent", "warning", "error"]:
            raise ValueError(f"Invalid argument skip_empty_level={skip_empty_level}")
        self.skip_empty_level = skip_empty_level
        self.random_shuffler = RandomShuffler()
        self.bucket_idx = 0
        # TODO: we might want to enable some hybrid mode (default left_pad True for LM, else False)
        if task != CorpusTask.TRAIN and left_pad:
            self.left_pad = True
        else:
            self.left_pad = False
        self.model_type = model_type
        self.image_patch_size = image_patch_size
        self.adapter = adapter

    @classmethod
    def from_config(
        cls,
        corpora,
        transforms,
        vocabs,
        config,
        task,
        device,
        stride=1,
        offset=0,
        model_type=None,
    ):
        """Initilize `DynamicDatasetIter` with options parsed from `opt`."""
        from eole.config.run import PredictConfig

        # patch for now, might do better later
        if isinstance(config, PredictConfig):
            running_config = config
        else:
            running_config = config.training

        corpora_info = {}
        batch_size = running_config.valid_batch_size if (task == CorpusTask.VALID) else running_config.batch_size
        # we might want to define proper infer/train classes instead of these conditions
        if task != CorpusTask.INFER:
            if running_config.batch_size_multiple is not None:
                batch_size_multiple = running_config.batch_size_multiple
            else:
                batch_size_multiple = 8 if running_config.compute_dtype == "fp16" else 1
            corpora_info = config.data
            bucket_size = running_config.bucket_size  # this should be in data rather than training?
            bucket_size_init = running_config.bucket_size_init
            bucket_size_increment = running_config.bucket_size_increment
            skip_empty_level = config.skip_empty_level
        else:
            batch_size_multiple = 1
            corpora_info[CorpusTask.INFER] = Dataset()
            corpora_info[CorpusTask.INFER].transforms = config.transforms
            corpora_info[CorpusTask.INFER].weight = 1
            # bucket_size = batch_size
            bucket_size = 16384
            bucket_size_init = -1
            bucket_size_increment = 0
            skip_empty_level = "warning"
        try:
            image_patch_size = config.model.encoder.patch_size * config.model.spatial_merge_size
        except AttributeError:
            image_patch_size = 16
        return cls(
            corpora,
            corpora_info,
            transforms,
            vocabs,
            task,
            running_config.batch_type,
            batch_size,
            batch_size_multiple,
            data_type=running_config.data_type,  # this is not super clear
            bucket_size=bucket_size,
            bucket_size_init=bucket_size_init,
            bucket_size_increment=bucket_size_increment,
            device=device,
            skip_empty_level=skip_empty_level,
            stride=stride,
            offset=offset,
            score_threshold=0 if isinstance(config, PredictConfig) else running_config.score_threshold,
            left_pad=getattr(config.model, "left_pad", False),
            model_type=model_type if model_type is not None else getattr(config.model, "model_type", None),
            image_patch_size=image_patch_size,
            adapter=getattr(config.model, "adapter", None),
        )

    def _init_datasets(self, worker_id):
        if self.num_workers > 0:
            stride = self.stride * self.num_workers
            offset = self.offset * self.num_workers + worker_id
        else:
            stride = self.stride
            offset = self.offset
        datasets_iterables = build_corpora_iters(
            self.corpora,
            self.transforms,
            self.corpora_info,
            skip_empty_level=self.skip_empty_level,
            stride=stride,
            offset=offset,
            image_patch_size=self.image_patch_size,
            adapter=self.adapter,
        )
        datasets_weights = {ds_name: int(self.corpora_info[ds_name].weight) for ds_name in datasets_iterables.keys()}
        if self.task == CorpusTask.TRAIN:
            self.mixer = WeightedMixer(datasets_iterables, datasets_weights)
        else:
            self.mixer = SequentialMixer(datasets_iterables, datasets_weights)
        self.init_iterators = True

    def _tuple_to_json_with_tokIDs(self, tuple_bucket):
        bucket = []
        tuple_bucket = transform_bucket(self.task, tuple_bucket, self.score_threshold)
        for example in tuple_bucket:
            if example is not None:
                bucket.append(numericalize(self.vocabs, example, model_type=self.model_type))

        return bucket

    def _add_indice(self, bucket):
        indice = 0
        indexed_bucket = []
        for ex in bucket:
            indexed_bucket.append((ex, indice))
            indice += 1
        return indexed_bucket

    def _bucketing(self):
        """
        Add up to bucket_size examples from the mixed corpora according
        to the above strategy. example tuple is converted to json and
        tokens numericalized.
        """
        bucket = []
        if self.bucket_size_init > 0:
            _bucket_size = self.bucket_size_init
        else:
            _bucket_size = self.bucket_size

        for ex in self.mixer:
            bucket.append(ex)
            if len(bucket) == _bucket_size:
                yield (self._tuple_to_json_with_tokIDs(bucket), self.bucket_idx)
                self.bucket_idx += 1
                bucket = []
                if _bucket_size < self.bucket_size:
                    _bucket_size += self.bucket_size_increment
                else:
                    _bucket_size = self.bucket_size
        if bucket:
            yield (self._tuple_to_json_with_tokIDs(bucket), self.bucket_idx)

    def batch_iter(self, data, batch_size, batch_type="sents", batch_size_multiple=1):
        """Yield elements from data in chunks of batch_size,
        where each chunk size is a multiple of batch_size_multiple.
        """

        def batch_size_fn(nbsents, maxlen):
            if batch_type == "sents":
                return nbsents
            elif batch_type == "tokens":
                return nbsents * maxlen
            else:
                raise ValueError(f"Invalid argument batch_type={batch_type}")

        def max_src_tgt(ex):
            """return the max tokens btw src and tgt in the sequence."""
            if ex.get("tgt", None) is not None:
                return max(len(ex["src"]["src_ids"]), len(ex["tgt"]["tgt_ids"]))
            return len(ex["src"]["src_ids"])

        minibatch, maxlen, size_so_far, seen = [], 0, 0, set()
        for ex, indice in data:
            src = ex["src"]["src"]
            if src not in seen or (self.task != CorpusTask.TRAIN):
                seen.add(src)
                minibatch.append((ex, indice))
                nbsents = len(minibatch)
                maxlen = max(max_src_tgt(ex), maxlen)
                size_so_far = batch_size_fn(nbsents, maxlen)
                if size_so_far >= batch_size:
                    overflowed = 1 if size_so_far > batch_size else 0
                    if batch_size_multiple > 1:
                        overflowed += (nbsents - overflowed) % batch_size_multiple
                    if overflowed == 0:
                        yield minibatch
                        minibatch, maxlen, size_so_far, seen = [], 0, 0, set()
                    else:
                        if overflowed == nbsents:
                            logger.warning(
                                "The batch will be filled until we reach"
                                " %d, its size may exceed %d tokens" % (batch_size_multiple, batch_size)
                            )
                        else:
                            yield minibatch[:-overflowed]
                            minibatch = minibatch[-overflowed:]
                            maxlen = max([max_src_tgt(ex) for ex, ind in minibatch])
                            size_so_far = batch_size_fn(len(minibatch), maxlen)
                            seen = set()

        if minibatch:
            yield minibatch

    def __iter__(self):
        for bucket, bucket_idx in self._bucketing():
            bucket = self._add_indice(bucket)
            bucket = sorted(bucket, key=lambda x: self.sort_key(x[0]))
            p_batch = list(
                self.batch_iter(
                    bucket,
                    self.batch_size,
                    batch_type=self.batch_type,
                    batch_size_multiple=self.batch_size_multiple,
                )
            )
            # For TRAIN we shuffle batches within the bucket
            # otherwise sequential
            if self.task == CorpusTask.TRAIN:
                p_batch = self.random_shuffler(p_batch)
            for i, minibatch in enumerate(p_batch):
                # for specific case of rnn_packed need to be sorted
                # within the batch
                if self.task == CorpusTask.TRAIN:
                    minibatch.sort(key=lambda x: self.sort_key(x[0]), reverse=True)
                tensor_batch = tensorify(self.vocabs, minibatch, self.device, self.left_pad)
                yield (tensor_batch, bucket_idx)


class OnDeviceDatasetIter:
    def __init__(self, data_iter, device):
        self.data_iter = data_iter
        self.device = device

    def __iter__(self):
        for tensor_batch, bucket_idx in self.data_iter:
            for key in tensor_batch.keys():
                if key not in [
                    "cid",
                    "ind_in_bucket",
                    "cid_line_number",
                    "left_pad",
                ]:
                    if isinstance(tensor_batch[key], list):
                        tensor_batch[key] = [t.to(self.device) for t in tensor_batch[key]]
                    else:
                        tensor_batch[key] = tensor_batch[key].to(self.device)
            yield (tensor_batch, bucket_idx)


def build_dynamic_dataset_iter(
    config,
    transforms,
    vocabs,
    task=CorpusTask.TRAIN,
    stride=1,
    offset=0,
    src=None,
    tgt=None,
    align=None,
    device_id=-1,
    model_type=None,
):
    """
    Build `DynamicDatasetIter` from opt.
    if src, tgt,align are passed then dataset is built from those lists
    instead of opt.[src, tgt, align]
    Typically this function is called for CorpusTask.[TRAIN,VALID,INFER]
    from the main train / predict scripts
    We disable automatic batching in the DataLoader.
    The custom optimized batching is performed by the
    custom class DynamicDatasetIter inherited from IterableDataset
    (and not by a custom collate function).
    We load opt.bucket_size examples, sort them and yield
    mini-batchs of size opt.batch_size.
    The bucket_size must be large enough to ensure homogeneous batches.
    Each worker will load opt.prefetch_factor mini-batches in
    advance to avoid the GPU waiting during the refilling of the bucket.
    """
    corpora = get_corpora(config, task, src=src, tgt=tgt, align=align)
    if corpora is None:
        assert task != CorpusTask.TRAIN, "only valid corpus is ignorable."
        return None
    device = get_device(device_id=device_id) if device_id >= 0 else torch.device("cpu")
    if hasattr(config, "training"):
        num_workers = getattr(config.training, "num_workers", 0)
    else:
        num_workers = 0
    if num_workers == 0 or task == CorpusTask.INFER:
        # single thread - create batch directly on GPU if device is gpu
        data_iter = DynamicDatasetIter.from_config(
            corpora,
            transforms,
            vocabs,
            config,
            task,
            stride=stride,
            offset=offset,
            device=device,
            model_type=model_type,
        )
        data_iter.num_workers = num_workers
        data_iter._init_datasets(0)  # when workers=0 init_fn not called
        return data_iter
    else:
        # multithread faster to create batch on CPU in each thread and then move it to gpu
        data_iter = DynamicDatasetIter.from_config(
            corpora,
            transforms,
            vocabs,
            config,
            task,
            stride=stride,
            offset=offset,
            device=torch.device("cpu"),
        )
        data_iter.num_workers = num_workers
        data_loader = DataLoader(
            data_iter,
            batch_size=None,
            pin_memory=True,
            multiprocessing_context="spawn",
            num_workers=data_iter.num_workers,
            worker_init_fn=data_iter._init_datasets,
            prefetch_factor=config.training.prefetch_factor,
        )
        # Move tensor_batch from cpu to device
        return OnDeviceDatasetIter(data_loader, device)
