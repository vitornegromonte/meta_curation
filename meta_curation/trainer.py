from __future__ import annotations

from typing import Callable, Dict

import torch
from torch import Tensor
from torch.func import functional_call
import torch.nn as nn

from .config import DataRaterConfig, ImplicitRaterConfig
from .meta_optim import MetaAdam
from .optim import _detach_state
from .types import ApplyFn, Batch, ParamDict, PerExampleLoss


class _InnerModel:
    """Book-keeping for one member of the inner population."""

    def __init__(self, module: nn.Module, optimizer):
        self.module = module
        self.optimizer = optimizer
        self.reset_state()

    def reset_state(self):
        # Parameters live as plain detached tensors (not nn.Parameter) because
        # we replace them by differentiable non-leaf tensors during unrolls.
        self.params: ParamDict = {k: v.detach().clone() for k, v in self.module.named_parameters()}
        self.buffers: ParamDict = {k: v.detach().clone() for k, v in self.module.named_buffers()}
        self.opt_state = self.optimizer.init_state(self.params)
        self.age = 0


class DataRaterTrainer:
    """Meta-learns a DataRater with meta-gradients (Algorithm 1).

    Typical use:

        trainer = DataRaterTrainer(rater, model_factory, inner_loss_fn,
                                   outer_loss_fn, inner_sampler, outer_sampler,
                                   config)
        for k in range(num_meta_steps):
            stats = trainer.meta_step()
        torch.save(rater.state_dict(), "datarater.pt")
    """

    def __init__(
        self,
        rater: nn.Module,
        model_factory: Callable[[], nn.Module],
        inner_loss_fn: PerExampleLoss,
        outer_loss_fn: PerExampleLoss,
        inner_sampler: Callable[[], Batch],
        outer_sampler: Callable[[], Batch],
        config: DataRaterConfig = DataRaterConfig(),
        device: str | torch.device = "cpu",
    ):
        assert 1 <= config.unroll_window <= config.inner_steps, (
            "need 1 <= unroll_window <= inner_steps"
        )
        self.rater = rater.to(device)
        self.cfg = config
        self.device = device
        self.model_factory = model_factory
        self.inner_loss_fn, self.outer_loss_fn = inner_loss_fn, outer_loss_fn
        self.inner_sampler, self.outer_sampler = inner_sampler, outer_sampler

        # Population of inner models, each with its own optimiser state.
        self.population = [
            _InnerModel(model_factory().to(device), config.inner_optimizer())
            for _ in range(config.num_inner_models)
        ]
        if config.reset_every and config.stagger_resets:
            n = config.num_inner_models
            for i, m in enumerate(self.population):
                m.age = (i * config.reset_every) // n  # evenly staggered ages

        # One MetaAdam per inner model (Alg. 1 line 10: H is per-model).
        eta = list(self.rater.parameters())
        self.meta_opts = [
            MetaAdam(
                eta,
                lr=config.meta_lr,
                weight_decay=config.meta_weight_decay,
                clip_norm=config.meta_grad_clip,
            )
            for _ in self.population
        ]
        self.meta_steps = 0

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _make_apply(m: _InnerModel, params: ParamDict) -> ApplyFn:
        """Return `apply(*a, **k)` that runs the inner model with `params`."""
        merged = {**params, **m.buffers}
        return lambda *a, **k: functional_call(m.module, merged, a, k)

    def _weights(self, batch: Batch, with_grad: bool) -> Tensor:
        """sigma_B(phi_eta(x)): softmax of DataRater scores over the batch."""
        if with_grad:
            scores = self.rater(batch)
        else:
            with torch.no_grad():
                scores = self.rater(batch)
        assert scores.ndim == 1, "rater must return one scalar score per example"
        return torch.softmax(scores, dim=0)  # weights sum to 1 within the batch

    # --------------------------------------------- Alg. 1 `UpdateInnerModel`
    def _update_inner_model(self, m: _InnerModel) -> ParamDict:
        """Run T inner steps on the weighted loss (Eq. 4).

        The first `T - W` steps are plain (non-differentiable w.r.t. eta);
        the last `W` steps keep the autograd graph so the outer loss can be
        back-propagated to eta ("truncated" meta-gradient window).

        Returns the final parameters (still attached to the graph w.r.t. eta).
        """
        T, W = self.cfg.inner_steps, self.cfg.unroll_window

        # Leaf copies of the current params so autograd.grad works w.r.t. them.
        params = {k: v.detach().requires_grad_(True) for k, v in m.params.items()}
        opt_state = _detach_state(m.opt_state)

        for t in range(T):
            tracked = t >= T - W  # inside the differentiable window?
            batch = self.inner_sampler()
            w = self._weights(batch, with_grad=tracked)
            apply = self._make_apply(m, params)
            per_ex = self.inner_loss_fn(apply, batch)  # [B]
            loss = (w * per_ex).sum()  # weighted loss
            # create_graph=True keeps dg/d(eta) and d g/d theta (2nd order).
            grads = torch.autograd.grad(loss, list(params.values()), create_graph=tracked)
            grads = dict(zip(params.keys(), grads))

            params, opt_state = m.optimizer.step(params, grads, opt_state)
            if not tracked:  # outside the window: cut the graph, re-leaf
                params = {k: v.detach().requires_grad_(True) for k, v in params.items()}
                opt_state = _detach_state(opt_state)

        m.opt_state = opt_state  # will be detached on commit below
        return params

    # ------------------------------------------------------------ one meta step
    def meta_step(self) -> Dict[str, float]:
        """One iteration of the outer loop (Alg. 1 lines 5-11)."""
        eta = list(self.rater.parameters())
        deltas_sum = [torch.zeros_like(p) for p in eta]
        outer_losses, gnorms = [], []

        for i, m in enumerate(self.population):
            # Lifetime management: periodically reset to a fresh random init.
            if self.cfg.reset_every and m.age >= self.cfg.reset_every:
                m.module = self.model_factory().to(self.device)
                m.reset_state()

            # (line 7) theta_{k+1} = UpdateInnerModel(theta_k, eta_k)
            new_params = self._update_inner_model(m)

            # (line 8) sample outer batch from held-out data
            outer_batch = self.outer_sampler()
            apply = self._make_apply(m, new_params)
            outer_loss = self.outer_loss_fn(apply, outer_batch)
            outer_loss = outer_loss.mean() if outer_loss.ndim else outer_loss

            # (line 9) meta-gradient: d L(theta_T(eta)) / d eta  via backprop
            #          through the unrolled inner updates.
            grads = torch.autograd.grad(outer_loss, eta, allow_unused=True)
            grads = [torch.zeros_like(p) if g is None else g for p, g in zip(eta, grads)]

            # (line 10) per-model meta-optimiser -> additive update
            d = self.meta_opts[i].delta(eta, grads)
            for acc, di in zip(deltas_sum, d):
                acc += di

            # Commit the inner model's new parameters (detached) -- the inner
            # model keeps training across outer steps (persistent population).
            m.params = {k: v.detach() for k, v in new_params.items()}
            m.opt_state = _detach_state(m.opt_state)
            m.age += 1

            outer_losses.append(float(outer_loss.detach()))
            gnorms.append(float(torch.sqrt(sum((g**2).sum() for g in grads))))

        # (line 11) eta_{k+1} = mean_i  eta-bar^i   (average of per-model updates)
        with torch.no_grad():
            n = len(self.population)
            for p, ds in zip(eta, deltas_sum):
                p.add_(ds / n)

        self.meta_steps += 1
        return {
            "outer_loss": sum(outer_losses) / n,
            "meta_grad_norm": sum(gnorms) / n,
        }


class ImplicitDataRaterTrainer(DataRaterTrainer):
    """iMAML-style DataRater training (see `implicit.py`).

    Same population / resets / per-model MetaAdam averaging as the explicit
    path, but the meta-gradient comes from the implicit theorem at a
    proximal stationary point instead of backprop through unrolled steps.
    `_InnerModel.opt_state` is unused here (kept only so resets work).
    """

    def __init__(
        self,
        rater: nn.Module,
        model_factory: Callable[[], nn.Module],
        inner_loss_fn: PerExampleLoss,
        outer_loss_fn: PerExampleLoss,
        inner_sampler: Callable[[], Batch],
        outer_sampler: Callable[[], Batch],
        config: ImplicitRaterConfig = ImplicitRaterConfig(),
        device: str | torch.device = "cpu",
    ):
        super().__init__(
            rater,
            model_factory,
            inner_loss_fn,
            outer_loss_fn,
            inner_sampler,
            outer_sampler,
            config=config,
            device=device,
        )
        from .implicit import ConjugateGradientSolver

        self.solver = ConjugateGradientSolver(
            iters=config.cg_iters, tol=config.cg_tol, damping=config.cg_damping
        )

    def meta_step(self) -> Dict[str, float]:
        """One implicit outer iteration."""
        from .implicit import implicit_update

        eta = list(self.rater.parameters())
        deltas_sum = [torch.zeros_like(p) for p in eta]
        outer_losses, gnorms = [], []
        cfg = self.cfg

        for i, m in enumerate(self.population):
            if cfg.reset_every and m.age >= cfg.reset_every:
                m.module = self.model_factory().to(self.device)
                m.reset_state()

            new_params, grads, outer_loss = implicit_update(
                self,
                m,
                eta,
                self.solver,
                cfg.inner_steps,
                cfg.inner_lr,
                cfg.proximal_lambda,
            )
            d = self.meta_opts[i].delta(eta, grads)
            for acc, di in zip(deltas_sum, d):
                acc += di

            m.params = {k: v.detach() for k, v in new_params.items()}
            m.age += 1
            outer_losses.append(outer_loss)
            gnorms.append(float(torch.sqrt(sum((g**2).sum() for g in grads))))

        with torch.no_grad():
            n = len(self.population)
            for p, ds in zip(eta, deltas_sum):
                p.add_(ds / n)

        self.meta_steps += 1
        return {
            "outer_loss": sum(outer_losses) / n,
            "meta_grad_norm": sum(gnorms) / n,
        }
