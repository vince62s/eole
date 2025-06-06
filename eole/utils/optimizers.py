""" Optimizers class """

import torch
from torch.amp import GradScaler
from torch.nn.utils import clip_grad_norm_
import operator
import functools
from copy import copy
from math import sqrt, cos, pi
import os

try:
    import optimi

    optimi_available = True
except ImportError:
    optimi_available = False
    pass


def build_torch_optimizer(model, config):
    """Builds the PyTorch optimizer.

    We use the default parameters for Adam that are suggested by
    the original paper https://arxiv.org/pdf/1412.6980.pdf
    These values are also used by other established implementations,
    e.g. https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer
    https://keras.io/optimizers/
    Recently there are slightly different values used in the paper
    "Attention is all you need"
    https://arxiv.org/pdf/1706.03762.pdf, particularly the value beta2=0.98
    was used there however, beta2=0.999 is still arguably the more
    established value, so we use that here as well

    Args:
      model: The model to optimize.
      opt. The dictionary of options.

    Returns:
      A ``torch.optim.Optimizer`` instance.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    betas = [config.adam_beta1, config.adam_beta2]

    if config.use_amp or not optimi_available:
        optim = torch.optim
    else:
        optim = optimi
    # optimi supports only sgd / adam / adamw for us
    # hence we use directly torch.optim for others
    if config.optim == "sgd":
        optimizer = optim.SGD(params, lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optim == "adagrad":
        optimizer = torch.optim.Adagrad(
            params,
            lr=config.learning_rate,
            initial_accumulator_value=config.adagrad_accumulator_init,
            weight_decay=config.weight_decay,
        )
    elif config.optim == "adadelta":
        optimizer = torch.optim.Adadelta(params, lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optim == "adafactor":
        optimizer = AdaFactor(
            params,
            non_constant_decay=True,
            enable_factorization=True,
            weight_decay=config.weight_decay,
        )
    elif config.optim == "adam":
        optimizer = optim.Adam(
            params,
            lr=config.learning_rate,
            betas=betas,
            eps=config.adam_eps,
            weight_decay=config.weight_decay,
        )
    elif config.optim == "adamw":
        optimizer = optim.AdamW(
            params,
            lr=config.learning_rate,
            betas=betas,
            eps=config.adam_eps,
            weight_decay=config.weight_decay,
            foreach=False,
        )
    elif config.optim == "sparseadam":
        dense = []
        sparse = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # TODO: Find a better way to check for sparse gradients.
            if "embed" in name:
                sparse.append(param)
            else:
                dense.append(param)
        optimizer = MultipleOptimizer(
            [
                optim.Adam(dense, lr=config.learning_rate, betas=betas, eps=config.adam_eps),
                torch.optim.SparseAdam(sparse, lr=config.learning_rate, betas=betas, eps=config.adam_eps),
            ]
        )
    elif config.optim in ["adamw8bit", "pagedadamw8bit", "pagedadamw32bit"]:
        try:
            os.environ["BITSANDBYTES_NOWELCOME"] = "1"
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("Install bitsandbytes to use bnb optimizers")
        if config.optim == "adamw8bit":
            optimizer = bnb.optim.AdamW8bit(
                params,
                lr=config.learning_rate,
                betas=betas,
                eps=config.adam_eps,
                weight_decay=config.weight_decay,
                amsgrad=False,
                optim_bits=8,
                args=None,
                min_8bit_size=1024,
                percentile_clipping=100,
                block_wise=True,
                is_paged=False,
            )
        elif config.optim == "pagedadamw8bit":
            optimizer = bnb.optim.PagedAdamW8bit(
                params,
                lr=config.learning_rate,
                betas=betas,
                eps=config.adam_eps,
                weight_decay=config.weight_decay,
                amsgrad=False,
                optim_bits=8,
                args=None,
                min_8bit_size=4096,
                percentile_clipping=100,
                block_wise=True,
            )
        elif config.optim == "pagedadamw32bit":
            optimizer = bnb.optim.PagedAdamW32bit(
                params,
                lr=config.learning_rate,
                betas=betas,
                eps=config.adam_eps,
                weight_decay=config.weight_decay,
                amsgrad=False,
                optim_bits=32,
                args=None,
                min_8bit_size=4096,
                percentile_clipping=100,
                block_wise=True,
            )
        else:
            raise ValueError("Invalid optimizer type: " + config.optim)
    else:
        raise ValueError("Invalid optimizer type: " + config.optim)

    return optimizer


def make_learning_rate_decay_fn(config):
    """Returns the learning decay function from options."""
    model_config = config.model
    running_config = config.training
    if running_config.decay_method == "noam":
        return functools.partial(
            noam_decay,
            warmup_steps=running_config.warmup_steps,
            model_size=model_config.hidden_size,
        )
    elif running_config.decay_method == "noamwd":
        return functools.partial(
            noamwd_decay,
            warmup_steps=running_config.warmup_steps,
            model_size=model_config.hidden_size,
            rate=running_config.learning_rate_decay,
            decay_steps=running_config.decay_steps,
            start_step=running_config.start_decay_steps,
        )
    elif running_config.decay_method == "cosine":
        return functools.partial(
            cosine_decay,
            warmup_steps=running_config.warmup_steps,
            train_steps=running_config.train_steps,
        )
    elif running_config.decay_method == "rsqrt":
        return functools.partial(rsqrt_decay, warmup_steps=running_config.warmup_steps)
    elif running_config.start_decay_steps is not None:
        return functools.partial(
            exponential_decay,
            rate=running_config.learning_rate_decay,
            decay_steps=running_config.decay_steps,
            start_step=running_config.start_decay_steps,
        )


def noam_decay(step, warmup_steps, model_size):
    """Learning rate schedule described in
    https://arxiv.org/pdf/1706.03762.pdf.
    """
    return model_size ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))


def noamwd_decay(step, warmup_steps, model_size, rate, decay_steps, start_step=0):
    """Learning rate schedule optimized for huge batches"""
    return (
        model_size ** (-0.5)
        * min(step ** (-0.5), step * warmup_steps ** (-1.5))
        * rate ** (max(step - start_step + decay_steps, 0) // decay_steps)
    )


def cosine_decay(step, warmup_steps, train_steps):
    if step < warmup_steps:
        return step / warmup_steps
    else:
        decay_ratio = (step - warmup_steps) / (train_steps - warmup_steps)
        return 0.5 * (1.0 + cos(pi * decay_ratio))


def exponential_decay(step, rate, decay_steps, start_step=0):
    """A standard exponential decay, scaling the learning rate by :obj:`rate`
    every :obj:`decay_steps` steps.
    """
    return rate ** (max(step - start_step + decay_steps, 0) // decay_steps)


def rsqrt_decay(step, warmup_steps):
    """Decay based on the reciprocal of the step square root."""
    return 1.0 / sqrt(max(step, warmup_steps))


class MultipleOptimizer(object):
    """Implement multiple optimizers needed for sparse adam"""

    def __init__(self, op):
        """?"""
        self.optimizers = op

    @property
    def param_groups(self):
        param_groups = []
        for optimizer in self.optimizers:
            param_groups.extend(optimizer.param_groups)
        return param_groups

    def zero_grad(self, set_to_none=True):
        """?"""
        for op in self.optimizers:
            op.zero_grad(set_to_none)

    def step(self):
        """?"""
        for op in self.optimizers:
            op.step()

    @property
    def state(self):
        """?"""
        return {k: v for op in self.optimizers for k, v in op.state.items()}

    def state_dict(self):
        """?"""
        return [op.state_dict() for op in self.optimizers]

    def load_state_dict(self, state_dicts):
        """?"""
        assert len(state_dicts) == len(self.optimizers)
        for i in range(len(state_dicts)):
            self.optimizers[i].load_state_dict(state_dicts[i])


class Optimizer(object):
    """
    Controller class for optimization. Mostly a thin
    wrapper for `optim`, but also useful for implementing
    rate scheduling beyond what is currently available.
    Also implements necessary methods for training RNNs such
    as grad manipulations.

    Args:
        optimizer: A ``torch.optim.Optimizer`` instance.
        learning_rate: The initial learning rate.
        learning_rate_decay_fn: An optional callable taking the current step
            as argument and return a learning rate scaling factor.
        max_grad_norm: Clip gradients to this global norm.
    """

    def __init__(self, optimizer, learning_rate, learning_rate_decay_fn=None, max_grad_norm=None, use_amp=True):
        self._optimizer = optimizer
        self._learning_rate = learning_rate
        self._learning_rate_decay_fn = learning_rate_decay_fn
        self._max_grad_norm = max_grad_norm or 0
        self._training_step = 1
        self._decay_step = 1
        self.use_amp = use_amp
        self._scaler = None

    @classmethod
    def from_config(cls, model, config, checkpoint=None):
        """Builds the optimizer from options.

        Args:
          cls: The ``Optimizer`` class to instantiate.
          model: The model to optimize.
          opt: The dict of user options.
          checkpoint: An optional checkpoint to load states from.

        Returns:
          An ``Optimizer`` instance.
        """
        # we could almost go with only training config here, except for noam schedule which requires hidden size (additional kwarg?) # noqa: E501
        running_config = config.training
        optim_state_dict = None

        if running_config.train_from and checkpoint is not None and "optim" in checkpoint.keys():
            optim = checkpoint["optim"]
            ckpt_config = checkpoint["config"].training
            ckpt_state_dict = {}
            if isinstance(optim, Optimizer):  # Backward compatibility.
                ckpt_state_dict["training_step"] = optim._step + 1
                ckpt_state_dict["decay_step"] = optim._step + 1
                ckpt_state_dict["optimizer"] = optim.optimizer.state_dict()
            else:
                ckpt_state_dict = optim

            # we might be able to simplify this with the new general config update
            if running_config.reset_optim == "none":
                # Load everything from the checkpoint.
                running_config = ckpt_config
                optim_state_dict = ckpt_state_dict
            elif running_config.reset_optim == "all":
                # Build everything from scratch.
                pass
            elif running_config.reset_optim == "states":
                # Reset optimizer, keep options.
                running_config = ckpt_config
                optim_state_dict = ckpt_state_dict
                del optim_state_dict["optimizer"]
            elif running_config.reset_optim == "keep_states":
                # Reset options, keep optimizer.
                optim_state_dict = ckpt_state_dict

        use_amp = running_config.use_amp and running_config.compute_dtype in [torch.float16, torch.bfloat16]
        optimizer = cls(
            build_torch_optimizer(model, running_config),
            running_config.learning_rate,
            learning_rate_decay_fn=make_learning_rate_decay_fn(config),
            max_grad_norm=running_config.max_grad_norm,
            use_amp=use_amp,
        )
        # if running_config.compute_dtype in [torch.float16, torch.bfloat16]:
        if use_amp:
            optimizer._scaler = GradScaler("cuda")

        if optim_state_dict:
            optimizer.load_state_dict(optim_state_dict)
        return optimizer

    @property
    def training_step(self):
        """The current training step."""
        return self._training_step

    @property
    def amp(self):
        """True if use torch amp mix precision training."""
        return self.use_amp

    def learning_rate(self, step=None):
        """Returns the current learning rate."""
        if step is None:
            step = self._decay_step
        if self._learning_rate_decay_fn is None:
            return self._learning_rate
        scale = self._learning_rate_decay_fn(step)
        return scale * self._learning_rate

    def state_dict(self):
        return {
            "training_step": self._training_step,
            "decay_step": self._decay_step,
            "optimizer": self._optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self._training_step = state_dict["training_step"]
        # State can be partially restored.
        if "decay_step" in state_dict:
            self._decay_step = state_dict["decay_step"]
        if "optimizer" in state_dict:
            self._optimizer.load_state_dict(state_dict["optimizer"])

    def zero_grad(self, set_to_none=True):
        """Zero the gradients of optimized parameters."""
        self._optimizer.zero_grad(set_to_none)

    def backward(self, loss):
        """Wrapper for backward pass. Some optimizer requires ownership of the
        backward pass."""
        if self._scaler is not None:
            self._scaler.scale(loss).backward()
        else:
            loss.backward()

    def step(self):
        """Update the model parameters based on current gradients.

        Optionally, will employ gradient modification or update learning
        rate.
        """
        learning_rate = self.learning_rate()

        if self._scaler is not None:
            self._scaler.unscale_(self._optimizer)

        for group in self._optimizer.param_groups:
            group["lr"] = learning_rate
            if self._max_grad_norm > 0:
                clip_grad_norm_(group["params"], self._max_grad_norm)

        if self._scaler is not None:
            # unscaled optimizer's gradients (already done therefore skip),
            # skips optimizer.step() if gradients contain infs/NaNs.
            self._scaler.step(self._optimizer)
            # Updates the scale for next iteration.
            self._scaler.update()
        else:
            self._optimizer.step()
        self._decay_step += 1
        self._training_step += 1


# Code below is an implementation of https://arxiv.org/pdf/1804.04235.pdf
# inspired but modified from https://github.com/DeadAt0m/adafactor-pytorch


class AdaFactor(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=None,
        beta1=0.9,
        beta2=0.999,
        eps1=1e-30,
        eps2=1e-3,
        cliping_threshold=1,
        non_constant_decay=True,
        enable_factorization=True,
        ams_grad=True,
        weight_decay=0,
    ):
        enable_momentum = beta1 != 0

        if non_constant_decay:
            ams_grad = False

        defaults = dict(
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            eps1=eps1,
            eps2=eps2,
            cliping_threshold=cliping_threshold,
            weight_decay=weight_decay,
            ams_grad=ams_grad,
            enable_factorization=enable_factorization,
            enable_momentum=enable_momentum,
            non_constant_decay=non_constant_decay,
        )

        super(AdaFactor, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdaFactor, self).__setstate__(state)

    def _experimental_reshape(self, shape):
        temp_shape = shape[2:]
        if len(temp_shape) == 1:
            new_shape = (shape[0], shape[1] * shape[2])
        else:
            tmp_div = len(temp_shape) // 2 + len(temp_shape) % 2
            new_shape = (
                shape[0] * functools.reduce(operator.mul, temp_shape[tmp_div:], 1),
                shape[1] * functools.reduce(operator.mul, temp_shape[:tmp_div], 1),
            )
        return new_shape, copy(shape)

    def _check_shape(self, shape):
        """
        output1 - True - algorithm for matrix, False - vector;
        output2 - need reshape
        """
        if len(shape) > 2:
            return True, True
        elif len(shape) == 2:
            return True, False
        elif len(shape) == 2 and (shape[0] == 1 or shape[1] == 1):
            return False, False
        else:
            return False, False

    def _rms(self, x):
        return sqrt(torch.mean(x.pow(2)))

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data

                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse \
                                       gradients, use SparseAdam instead"
                    )

                is_matrix, is_need_reshape = self._check_shape(grad.size())
                new_shape = p.data.size()
                if is_need_reshape and group["enable_factorization"]:
                    new_shape, old_shape = self._experimental_reshape(p.data.size())
                    grad = grad.view(new_shape)

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    if group["enable_momentum"]:
                        state["exp_avg"] = torch.zeros(new_shape, dtype=torch.float32, device=p.grad.device)

                    if is_matrix and group["enable_factorization"]:
                        state["exp_avg_sq_R"] = torch.zeros(
                            (1, new_shape[1]), dtype=torch.float32, device=p.grad.device
                        )
                        state["exp_avg_sq_C"] = torch.zeros(
                            (new_shape[0], 1), dtype=torch.float32, device=p.grad.device
                        )
                    else:
                        state["exp_avg_sq"] = torch.zeros(new_shape, dtype=torch.float32, device=p.grad.device)
                    if group["ams_grad"]:
                        state["exp_avg_sq_hat"] = torch.zeros(new_shape, dtype=torch.float32, device=p.grad.device)

                if group["enable_momentum"]:
                    exp_avg = state["exp_avg"]

                if is_matrix and group["enable_factorization"]:
                    exp_avg_sq_r = state["exp_avg_sq_R"]
                    exp_avg_sq_c = state["exp_avg_sq_C"]
                else:
                    exp_avg_sq = state["exp_avg_sq"]

                if group["ams_grad"]:
                    exp_avg_sq_hat = state["exp_avg_sq_hat"]

                state["step"] += 1
                lr_t = group["lr"]
                lr_t *= max(group["eps2"], self._rms(p.data))

                if group["enable_momentum"]:
                    if group["non_constant_decay"]:
                        beta1_t = (
                            group["beta1"]
                            * (1 - group["beta1"] ** (state["step"] - 1))
                            / (1 - group["beta1"] ** state["step"])
                        )
                    else:
                        beta1_t = group["beta1"]
                    exp_avg.mul_(beta1_t).add_(1 - beta1_t, grad)

                if group["non_constant_decay"]:
                    beta2_t = (
                        group["beta2"]
                        * (1 - group["beta2"] ** (state["step"] - 1))
                        / (1 - group["beta2"] ** state["step"])
                    )
                else:
                    beta2_t = group["beta2"]

                if is_matrix and group["enable_factorization"]:
                    exp_avg_sq_r.mul_(beta2_t).add_(
                        1 - beta2_t,
                        torch.sum(
                            torch.mul(grad, grad).add_(group["eps1"]),
                            dim=0,
                            keepdim=True,
                        ),
                    )
                    exp_avg_sq_c.mul_(beta2_t).add_(
                        1 - beta2_t,
                        torch.sum(
                            torch.mul(grad, grad).add_(group["eps1"]),
                            dim=1,
                            keepdim=True,
                        ),
                    )
                    v = torch.mul(exp_avg_sq_c, exp_avg_sq_r).div_(torch.sum(exp_avg_sq_r))
                else:
                    exp_avg_sq.mul_(beta2_t).addcmul_(1 - beta2_t, grad, grad).add_((1 - beta2_t) * group["eps1"])
                    v = exp_avg_sq

                g = grad
                if group["enable_momentum"]:
                    g = torch.div(exp_avg, 1 - beta1_t ** state["step"])

                if group["ams_grad"]:
                    torch.max(exp_avg_sq_hat, v, out=exp_avg_sq_hat)
                    v = exp_avg_sq_hat
                    u = torch.div(
                        g,
                        (torch.div(v, 1 - beta2_t ** state["step"])).sqrt().add_(group["eps1"]),
                    )
                else:
                    u = torch.div(g, v.sqrt())

                u.div_(max(1, self._rms(u) / group["cliping_threshold"]))
                p.data.add_(-lr_t * (u.view(old_shape) if is_need_reshape and group["enable_factorization"] else u))

                if group["weight_decay"] != 0:
                    p.data.add_(-group["weight_decay"] * lr_t, p.data)

        return loss
