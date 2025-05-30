import os
import torch
import json
import shutil
from collections import deque
from eole.utils.logging import logger
from eole.inputters.inputter import vocabs_to_dict
from eole.modules.lora import lora_state_dict
from eole.config import recursive_model_fields_set
from eole.config.run import TrainConfig
from eole.constants import DefaultTokens

try:
    from safetensors.torch import save_file
except ImportError:
    raise ImportError("run: pip install safetensors, to use safetensors")


def build_model_saver(config, model, vocabs, optim, device_id, transforms):

    model_saver = TrainingModelSaver(
        # config.training.save_model,
        config.training.model_path,
        model,
        config,
        vocabs,
        optim,
        config.training.keep_checkpoint,
        config.training.save_format,
        device_id,
        transforms,
    )
    return model_saver


def load_checkpoint(model_path):
    """
    Load checkpoint from `model_path` if any else return `None`.
    TODO: we need to clarify this compared to eole.models.BaseModel.load_checkpoint
    """
    checkpoint = None
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"{model_path} does not seem to exist.")
    elif os.path.isdir(model_path):
        os.environ["MODEL_PATH"] = model_path
        logger.info("Loading checkpoint from %s" % model_path)
        # checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
        checkpoint = {}
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                config_dict = json.loads(os.path.expandvars(f.read()))
                # drop data to prevent validation issues
                config_dict["data"] = {}
                _config = TrainConfig(**config_dict)
                checkpoint["config"] = _config
        else:
            raise FileNotFoundError(f"{model_path} does not contain config.json")
        vocab_path = os.path.join(model_path, "vocab.json")
        if os.path.exists(vocab_path):
            with open(vocab_path, encoding="utf-8") as f:
                checkpoint["vocab"] = json.load(f)
            # use default specials if not specified
            if "specials" not in checkpoint["vocab"].keys():
                checkpoint["vocab"]["specials"] = {
                    "bos_token": DefaultTokens.BOS,
                    "pad_token": DefaultTokens.PAD,
                    "eos_token": DefaultTokens.EOS,
                    "unk_token": DefaultTokens.UNK,
                }
        else:
            raise FileNotFoundError(f"{model_path} does not contain vocab.json")
        optim_path = os.path.join(model_path, "optimizer.pt")
        if os.path.exists(optim_path):
            checkpoint["optim"] = torch.load(optim_path, map_location=torch.device("cpu"), weights_only=True)
    else:
        raise FileNotFoundError(f"{model_path} is not a directory.")

    return checkpoint


class ModelSaverBase(object):
    """
    Base class for various model saving scenarii.
    Not really used for now, but will be when ConvertModelSaver is properly implemented.
    """

    def __init__(self, model_path, model, config, vocabs, device_id=0, transforms=None):
        self.model_path = model_path
        self.model = model
        self.config = config
        self.vocabs = vocabs
        self.device_id = device_id
        self.transforms = transforms

    def _save(self, *args, **kwargs):
        raise NotImplementedError()


# TODO: move saving and sharding logic from convert tools here
# class ConvertModelSaver(ModelSaverBase):
#     pass


class TrainingModelSaver(ModelSaverBase):
    """Simple model saver to filesystem"""

    def __init__(
        self,
        model_path,
        model,
        config,
        vocabs,
        optim=None,
        keep_checkpoint=-1,
        save_format="pytorch",
        device_id=0,
        transforms=None,
    ):
        self.model_path = model_path
        self.step_dir = None
        self.model = model
        self.config = config
        self.vocabs = vocabs
        self.optim = optim
        self.last_saved_step = None
        self.keep_checkpoint = keep_checkpoint
        self.save_format = save_format
        self.device_id = device_id
        self.transforms = transforms

        # we might raise some warning if model_path already exists
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path, exist_ok=True)

        if keep_checkpoint > 0:
            self.checkpoint_queue = deque([], maxlen=keep_checkpoint)

    def cleanup(self):
        if self.keep_checkpoint > 0:
            if len(self.checkpoint_queue) == self.checkpoint_queue.maxlen:
                step_dir_to_delete = os.path.join(self.model_path, self.checkpoint_queue.popleft())
                try:
                    shutil.rmtree(step_dir_to_delete)
                except FileNotFoundError:
                    pass
                except Exception:
                    raise

            self.checkpoint_queue.append(self.step_dir)

    def _maybe_lora(self, model):
        if (hasattr(self.config.training, "lora_layers") and len(self.config.training.lora_layers) > 0) or (
            hasattr(self.config.training, "lora_embedding") and self.config.training.lora_embedding
        ):
            model_state_dict = lora_state_dict(model, bias="lora_only")
            for k, v in model.state_dict().items():
                if "estimator" in k:
                    model_state_dict[k] = v
        else:
            model_state_dict = model.state_dict()
        return model_state_dict

    def _tensor_parallel_state_dict(self, model_state_dict, world_size):
        full_model = [None for _ in range(world_size)]
        for key, value in model_state_dict.items():
            model_state_dict[key] = value.cpu()
        torch.distributed.all_gather_object(full_model, model_state_dict)
        full_state_dict = {}
        for key in full_model[0].keys():
            key_2, key_1 = key.split(".")[-2:]
            averaged_params = {
                "linear_keys",
                "linear_values",
                "linear_query",
                "gate_up_proj",
                "up_proj",
            }
            cat_params = {"final_linear", "down_proj"}
            # we probably should try and improve this to rely on dimensions instead of names
            match key_1, key_2:
                case "lora_A", _ if key_2 in averaged_params:
                    full_state_dict[key] = sum([full_model[i][key].cpu() for i in range(world_size)]) / world_size
                case "lora_A", _ if key_2 in cat_params:
                    full_state_dict[key] = torch.cat([full_model[i][key].cpu() for i in range(world_size)], 1)
                case "lora_B", _ if key_2 in averaged_params:
                    full_state_dict[key] = torch.cat([full_model[i][key].cpu() for i in range(world_size)], 0)
                case "lora_B", _ if key_2 in cat_params:
                    full_state_dict[key] = torch.cat([full_model[i][key].cpu() for i in range(world_size)], 1)
                case _ if key_1 in averaged_params:
                    full_state_dict[key] = torch.cat([full_model[i][key].cpu() for i in range(world_size)], 0)
                case _ if key_1 in cat_params:
                    full_state_dict[key] = torch.cat([full_model[i][key].cpu() for i in range(world_size)], 1)
                case _ if key_2 in averaged_params:
                    full_state_dict[key] = torch.cat([full_model[i][key].cpu() for i in range(world_size)], 0)
                case _ if key_2 in cat_params:
                    full_state_dict[key] = torch.cat([full_model[i][key].cpu() for i in range(world_size)], 1)
                case _, _:
                    full_state_dict[key] = full_model[0][key]
        return full_state_dict

    def _save_tokenizer(self):
        pass

    def update_step_dir(self, step):
        self.step_dir = f"step_{step}"
        step_dir = os.path.join(self.model_path, self.step_dir)
        if not os.path.exists(step_dir):
            os.mkdir(step_dir)

    def _make_symlink(self, item):
        # symlink item from step_X dir to parent model_path
        symlink_src_path = os.path.join(self.step_dir, item)
        symlink_dst_path = os.path.join(self.model_path, item)
        if os.path.islink(symlink_dst_path):
            os.remove(symlink_dst_path)
        elif os.path.exists(symlink_dst_path):
            os.remove(symlink_dst_path)
        try:
            os.symlink(symlink_src_path, symlink_dst_path)
        except PermissionError:
            # If a symlink cannot be created, copy the file instead
            shutil.copy2(os.path.join(self.model_path, symlink_src_path), symlink_dst_path)

    def _save_optimizer(self):
        optim_data = self.optim.state_dict()
        optim_path = os.path.join(self.model_path, self.step_dir, "optimizer.pt")
        torch.save(optim_data, optim_path)
        self._make_symlink("optimizer.pt")

    def _save_weights(self, model_state_dict):
        # we might implement sharding later
        model_path = os.path.join(self.model_path, self.step_dir, "model.00.safetensors")
        if self.config.model.share_embeddings and "tgt_emb.embeddings.weight" in model_state_dict:
            model_state_dict.pop("tgt_emb.embeddings.weight")
        if self.config.model.share_decoder_embeddings and "generator.weight" in model_state_dict:
            model_state_dict.pop("generator.weight")
        if self.config.training.freeze_encoder:
            model_state_dict = {
                k: v for k, v in model_state_dict.items() if not (k.startswith("encoder") or k.startswith("adapter"))
            }
        if self.config.training.freeze_decoder:
            model_state_dict = {
                k: v
                for k, v in model_state_dict.items()
                if not (k.startswith("decoder") or k.startswith("generator") or k.startswith("tgt"))
            }
        save_file(model_state_dict, model_path)
        self._make_symlink("model.00.safetensors")

    def _save_vocab(self):
        vocab_data = vocabs_to_dict(self.vocabs)
        vocab_path = os.path.join(self.model_path, self.step_dir, "vocab.json")
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(vocab_data, f, indent=2, ensure_ascii=False)
        self._make_symlink("vocab.json")

    def _save_config(self):
        config_data = recursive_model_fields_set(self.config)
        config_path = os.path.join(self.model_path, self.step_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        self._make_symlink("config.json")

    def _save_transforms_artifacts(self):
        if self.transforms is not None:
            checkpoint_path = os.path.join(self.model_path, self.step_dir)
            for transform_name, transform in self.transforms.items():
                transform_save_config, artifacts = transform._save_artifacts(checkpoint_path)
                setattr(
                    self.config.transforms_configs,
                    transform_name,
                    transform_save_config,
                )
                for artifact in artifacts:
                    self._make_symlink(artifact)
                # we probably do not need to save transforms artifacts for each checkpoint
                # transform._save_artifacts(os.path.join(self.model_path, self.step_dir))

    # previously _st_save
    def _save(self, step):
        # convert state_dict to lora if needed
        model_state_dict = self._maybe_lora(self.model)

        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
        else:
            world_size = 1

        if world_size > 1:
            # gather model split across GPUs for tensor parallelism
            model_state_dict = self._tensor_parallel_state_dict(model_state_dict, world_size)

        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            self.update_step_dir(step)
            logger.info(f"Saving optimizer and weights to {self.step_dir}, and symlink to {self.model_path}")
            self._save_optimizer()
            self._save_weights(model_state_dict)
            logger.info("Saving transforms artifacts, if any, " f"to {os.path.join(self.model_path, self.step_dir)}")
            self._save_transforms_artifacts()
            logger.info(f"Saving config and vocab to {self.model_path}")
            self._save_vocab()
            self._save_config()
            self.cleanup()
            # we shall trigger optional saves from transforms here + some default inference config ?
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def save(self, step, moving_average=None):
        """Main entry point for model saver

        It wraps the `_save` method with checks and apply `keep_checkpoint`
        related logic
        """

        if self.keep_checkpoint == 0 or step == self.last_saved_step:
            return

        if moving_average:
            model_params_data = []
            for avg, param in zip(moving_average, self.model.parameters()):
                model_params_data.append(param.data)
                param.data = avg.data

        self._save(step)

        self.last_saved_step = step

        if moving_average:
            for param_data, param in zip(model_params_data, self.model.parameters()):
                param.data = param_data
