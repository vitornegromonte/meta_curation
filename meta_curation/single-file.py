"""
DataRater: Meta-Learned Dataset Curation -- a generic PyTorch skeleton
======================================================================

Reference: Calian, Farquhar, Kemaev, Zintgraf et al., "DataRater: Meta-Learned
Dataset Curation", NeurIPS 2025 (Google DeepMind).

This file is a *starting point*, not a reproduction of the paper's
infrastructure. It implements the algorithm (Algorithm 1 of the paper) in a
model-agnostic way, so you can plug in your own data, inner model, DataRater
architecture and losses.

-------------------------------------------------------------------------------
THE IDEA IN ONE PARAGRAPH
-------------------------------------------------------------------------------
A "DataRater" phi_eta(x) is a network that maps a data point x to a scalar
score. Inside a training batch B the scores are turned into weights with a
softmax, sigma_B(phi_eta(x)). An *inner model* with parameters theta is trained
on the weighted loss

    g_t = sum_{x in B_t}  sigma_{B_t}(phi_eta(x)) * grad_theta  l(x; theta_t)    (paper Eq. 4)

After T such inner updates we measure the loss L of the inner model on held-out
data (the *outer* loss) and back-propagate that loss THROUGH the unrolled inner
updates to get a meta-gradient dL/d(eta). Updating eta with this gradient makes
the DataRater up-weight data that makes the inner model learn faster on the
held-out data, and down-weight data that doesn't (noise, OCR garbage, ...).
Once trained, the DataRater is frozen and used to *filter*: oversample a
batch of size N/(1-rho), score it, and keep the top N (discard fraction rho).

-------------------------------------------------------------------------------
MAP OF THIS FILE
-------------------------------------------------------------------------------
 1. Differentiable inner optimisers  (SGD+momentum, Adam)  -- unrollable
 2. MetaAdam                         -- per-inner-model Adam on eta (paper §2)
 3. DataRaterConfig                  -- hyper-parameters
 4. DataRaterTrainer                 -- Algorithm 1 (meta-training)
 5. Filtering utilities              -- batch-level top-K and per-sample
                                        CDF-based acceptance probability
 6. Example DataRater networks       -- MLP and non-causal Transformer
 7. Toy demo (`python datarater.py`) -- reproduces the spirit of paper Fig. 3

-------------------------------------------------------------------------------
WHAT YOU MUST PROVIDE (the "generic" interface)
-------------------------------------------------------------------------------
* `model_factory()      -> nn.Module`
      Builds a fresh inner model (called at start and whenever one is reset).
* `rater: nn.Module`
      `rater(batch) -> Tensor[B]` of raw scores. `batch` can be any structure
      your samplers produce (tensor, tuple, dict ...), the trainer never
      looks inside it.
* `inner_loss_fn(apply, batch) -> Tensor[B]`
      PER-EXAMPLE inner loss l(x; theta). `apply(*args, **kwargs)` runs the
      inner model with the *current functional parameters*, e.g.
          lambda apply, batch: F.cross_entropy(apply(batch["x"]), batch["y"],
                                               reduction="none")
      NOTE: it must return one loss per example (no reduction), because the
      trainer applies the DataRater weights itself.
* `outer_loss_fn(apply, batch) -> Tensor[B]` (or a scalar)
      Held-out loss L(x; theta). In the paper it is the same functional form
      as the inner loss (next-token cross-entropy) evaluated on a held-out
      disjoint split of the *same* dataset.
* `inner_sampler() -> batch` and `outer_sampler() -> batch`
      Zero-argument callables returning a fresh batch from D_train / D_test.

-------------------------------------------------------------------------------
KNOWN GAPS VS. THE PAPER (things you'll likely want to add)
-------------------------------------------------------------------------------
* MixFlow-MG (Kemaev et al., 2025): the paper uses mixed-mode differentiation,
  block-level rematerialisation, etc. to make second-order meta-gradients fit
  in memory for 400M inner models. Here we use plain reverse-mode autograd with
  `create_graph=True`. For big models, look at `torch.utils.checkpoint` and
  `torch.func.jvp`/`jacfwd` for forward-over-reverse.
* Fused attention kernels (FlashAttention / SDPA flash backend) often do NOT
  support double-backward. If you use a Transformer inner model, force the math
  backend: `with torch.nn.attention.sdpa_kernel(SDPBackend.MATH): ...`.
* Distributed / data-parallel meta-training (the paper runs the population of
  inner models in parallel on TPUs). Here the population is a Python loop.
* Dropout/BatchNorm in the inner model: use `model.eval()`-style deterministic
  layers if you want reproducible meta-gradients. Buffers are treated as
  constants (not updated) in this implementation.

Tested with PyTorch >= 2.1 (needs `torch.func.functional_call`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor
from torch.func import functional_call
import torch.nn as nn
import torch.nn.functional as F

ParamDict = Dict[str, Tensor]
Batch = Any  # tensor / tuple / dict -- opaque to the trainer
ApplyFn = Callable[..., Any]
PerExampleLoss = Callable[[ApplyFn, Batch], Tensor]


# =============================================================================
# 1. Differentiable inner optimisers
# =============================================================================
# The inner update theta_{t+1} = Theta(theta_t, g_t) must be written with
# out-of-place tensor ops so that autograd can differentiate *through* it
# (this is what makes it a "meta-gradient"). torch.optim optimisers update
# in-place and under no_grad, so we can't use them here.
#
# Interface: `init_state(params) -> state` and
#            `step(params, grads, state) -> (new_params, new_state)`.
# Add your own (e.g. SGD without momentum, Lion, ...) by following the same
# interface.


class DifferentiableOptimizer:
    """Base class for functional, differentiable optimisers."""

    def init_state(self, params: ParamDict) -> Dict[str, Any]:
        raise NotImplementedError

    def step(
        self, params: ParamDict, grads: ParamDict, state: Dict[str, Any]
    ) -> Tuple[ParamDict, Dict[str, Any]]:
        raise NotImplementedError


class DiffSGD(DifferentiableOptimizer):
    """SGD with (optional) heavy-ball momentum."""

    def __init__(self, lr: float = 1e-2, momentum: float = 0.0):
        self.lr, self.momentum = lr, momentum

    def init_state(self, params):
        return {"buf": {k: torch.zeros_like(v) for k, v in params.items()}}

    def step(self, params, grads, state):
        new_buf, new_params = {}, {}
        for k, p in params.items():
            b = self.momentum * state["buf"][k] + grads[k]
            new_buf[k] = b
            new_params[k] = p - self.lr * b
        return new_params, {"buf": new_buf}


class DiffAdam(DifferentiableOptimizer):
    """Adam written functionally.

    Small deviation from torch.optim.Adam: we use sqrt(v_hat + eps^2) instead of
    sqrt(v_hat) + eps. The two are nearly identical but the former has a finite
    second derivative when v_hat == 0 (which is common for dead units) -- the
    latter would produce NaNs in the meta-gradient.
    """

    def __init__(self, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, betas[0], betas[1], eps

    def init_state(self, params):
        z = lambda: {k: torch.zeros_like(v) for k, v in params.items()}
        return {"m": z(), "v": z(), "t": 0}

    def step(self, params, grads, state):
        t = state["t"] + 1
        m, v, new_params = {}, {}, {}
        bc1, bc2 = 1 - self.b1**t, 1 - self.b2**t
        for k, p in params.items():
            g = grads[k]
            m[k] = self.b1 * state["m"][k] + (1 - self.b1) * g
            v[k] = self.b2 * state["v"][k] + (1 - self.b2) * g * g
            m_hat, v_hat = m[k] / bc1, v[k] / bc2
            new_params[k] = p - self.lr * m_hat / torch.sqrt(v_hat + self.eps**2)
        return new_params, {"m": m, "v": v, "t": t}


def _detach_state(state: Any) -> Any:
    """Recursively detach tensors in an optimiser state (cuts the meta-graph)."""
    if isinstance(state, Tensor):
        return state.detach()
    if isinstance(state, dict):
        return {k: _detach_state(v) for k, v in state.items()}
    return state


# =============================================================================
# 2. MetaAdam -- the meta-optimiser H(eta, g) of Algorithm 1
# =============================================================================
class MetaAdam:
    """A tiny stateful Adam that turns a meta-gradient into an *update delta*.

    The paper (Algorithm 1, lines 10-11; §2 "Implementation") passes the
    meta-gradient coming from EACH inner model through its OWN Adam instance and
    then AVERAGES the resulting updates. This stabilises meta-training because
    inner models at different stages of training produce meta-gradients with
    very different magnitudes; per-model normalisation equalises them.
    """

    def __init__(
        self,
        params: Sequence[Tensor],
        lr=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        clip_norm: Optional[float] = None,
    ):
        self.lr, self.b1, self.b2, self.eps = lr, betas[0], betas[1], eps
        self.wd, self.clip = weight_decay, clip_norm
        self.m = [torch.zeros_like(p) for p in params]
        self.v = [torch.zeros_like(p) for p in params]
        self.t = 0

    @torch.no_grad()
    def delta(self, params: Sequence[Tensor], grads: Sequence[Tensor]) -> List[Tensor]:
        """Return the additive update (to be *added* to params)."""
        if self.clip is not None:  # global-norm clipping
            total = torch.sqrt(sum((g**2).sum() for g in grads))
            scale = torch.clamp(self.clip / (total + 1e-12), max=1.0)
            grads = [g * scale for g in grads]
        self.t += 1
        out = []
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            m_hat = self.m[i] / (1 - self.b1**self.t)
            v_hat = self.v[i] / (1 - self.b2**self.t)
            d = -self.lr * m_hat / (v_hat.sqrt() + self.eps)
            d = d - self.lr * self.wd * p  # decoupled weight decay (AdamW style)
            out.append(d)
        return out


# =============================================================================
# 3. Configuration
# =============================================================================
@dataclass
class DataRaterConfig:
    # --- population of inner models (paper: 8 x 400M) ---
    num_inner_models: int = 4
    # --- inner loop ---
    inner_steps: int = 2  # T: inner updates per outer step (Alg.1 line 14)
    unroll_window: int = 2  # last W inner steps are differentiated through
    # (paper: 2; they found 1/2/4/8 similar).
    # Must satisfy 1 <= unroll_window <= inner_steps.
    inner_optimizer: Callable[[], DifferentiableOptimizer] = field(
        default_factory=lambda: DiffAdam(lr=1e-3)
    )
    # --- outer loop ---
    meta_lr: float = 1e-3
    meta_weight_decay: float = 0.0
    meta_grad_clip: Optional[float] = None
    # --- inner-model lifetimes ---
    # Re-initialise each inner model every `reset_every` meta-steps so the
    # DataRater sees ALL stages of training (not just one) and generalises to
    # fresh models. Initial ages are staggered so resets don't coincide
    # ("stratified lifetimes", paper Fig. 12).
    reset_every: Optional[int] = 1000
    stagger_resets: bool = True


# =============================================================================
# 4. Meta-training (Algorithm 1)
# =============================================================================
class _InnerModel:
    """Book-keeping for one member of the inner population."""

    def __init__(self, module: nn.Module, optimizer: DifferentiableOptimizer):
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
        return {"outer_loss": sum(outer_losses) / n, "meta_grad_norm": sum(gnorms) / n}


# =============================================================================
# 5. Filtering with a trained DataRater
# =============================================================================
def oversampled_batch_size(batch_size: int, discard_fraction: float) -> int:
    """Size N/(1-rho) of the batch to draw so that, after discarding the bottom
    `rho` fraction, exactly ~`batch_size` examples remain."""
    assert 0.0 <= discard_fraction < 1.0
    return math.ceil(batch_size / (1.0 - discard_fraction))


def select_from_batch(batch: Batch, idx: Tensor) -> Batch:
    """Index every tensor in a (nested) batch along dim 0."""
    if isinstance(batch, Tensor):
        return batch[idx.to(batch.device)]
    if isinstance(batch, dict):
        return {k: select_from_batch(v, idx) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(select_from_batch(v, idx) for v in batch)
    raise TypeError(f"Don't know how to index batch of type {type(batch)}")


@torch.no_grad()
def topk_filter_batch(rater: nn.Module, big_batch: Batch, keep: int) -> Tuple[Batch, Tensor]:
    """Batch-level top-K filtering (paper §2 "Data curation").

    Score an oversampled batch and keep the `keep` highest-rated examples.
    Returns (filtered_batch, kept_indices).
    """
    was_training = rater.training
    rater.eval()
    scores = rater(big_batch)
    rater.train(was_training)
    idx = torch.topk(scores, k=keep).indices
    return select_from_batch(big_batch, idx), idx


class ScoreCDF:
    """Empirical CDF F_Phi of DataRater scores over (a sample of) the dataset.

    Lets you make *independent, per-example* keep/discard decisions (e.g. in a
    massively parallel Apache-Beam-style pipeline) that match, in distribution,
    batch-level top-K filtering.
    """

    def __init__(self, scores: Tensor):
        self.sorted = torch.sort(scores.flatten().cpu()).values

    def __call__(self, s: Tensor) -> Tensor:
        """Fraction of reference scores <= s, in [0, 1]."""
        pos = torch.searchsorted(self.sorted, s.cpu().contiguous(), right=True)
        return pos.float() / len(self.sorted)


def acceptance_probability(p: Tensor, batch_size: int, keep: int) -> Tensor:
    """P(keep x) under batch-level top-K filtering, from its score quantile p.

    A point is kept iff at most K-1 of the other B-1 points in its batch beat
    it. Each other point beats it with prob (1-p), so

        P_accept = sum_{s=0}^{K-1} C(B-1, s) (1-p)^s p^(B-1-s)        (paper §2)

    Args:
        p: quantile(s) F_Phi(phi(x)) in [0, 1].
        batch_size: B -- the (oversampled) batch size.
        keep: K -- number of points kept per batch.
    """
    p = p.double().clamp(1e-12, 1 - 1e-12)
    Bm1 = batch_size - 1
    total = torch.zeros_like(p)
    for s in range(keep):
        log_comb = math.lgamma(Bm1 + 1) - math.lgamma(s + 1) - math.lgamma(Bm1 - s + 1)
        total += torch.exp(log_comb + s * torch.log1p(-p) + (Bm1 - s) * torch.log(p))
    return total.clamp(0, 1).float()


# =============================================================================
# 6. Example DataRater architectures (swap for your own)
# =============================================================================
class MLPDataRater(nn.Module):
    """DataRater for vector inputs. `batch` is a tuple/list of tensors that are
    flattened and concatenated per example (e.g. (x, y) for supervised data)."""

    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, batch) -> Tensor:
        if isinstance(batch, Tensor):
            batch = (batch,)
        z = torch.cat([b.flatten(1) for b in batch], dim=-1)
        return self.net(z).squeeze(-1)


class TransformerDataRater(nn.Module):
    """Non-causal Transformer DataRater for token sequences (as in the paper).

    `batch` is a LongTensor [B, S]. Tokens are embedded, passed through
    bidirectional self-attention layers, mean-pooled over (non-pad) positions
    and mapped to one scalar score. (The paper uses a 50M-parameter model; this
    is just a compact, readable stand-in.)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model=128,
        n_heads=4,
        n_layers=2,
        max_len=2048,
        pad_id: Optional[int] = None,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, 4 * d_model, dropout=0.0, batch_first=True, norm_first=True
        )
        self.enc = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.head = nn.Linear(d_model, 1)

    def forward(self, tokens: Tensor) -> Tensor:
        B, S = tokens.shape
        h = self.tok(tokens) + self.pos(torch.arange(S, device=tokens.device))
        pad = (tokens == self.pad_id) if self.pad_id is not None else None
        h = self.enc(h, src_key_padding_mask=pad)  # no causal mask => non-causal
        if pad is not None:
            keep = (~pad).unsqueeze(-1).float()
            h = (h * keep).sum(1) / keep.sum(1).clamp(min=1)
        else:
            h = h.mean(1)
        return self.head(h).squeeze(-1)


# =============================================================================
# 7. Toy demo -- DataRater learns to down-weight corrupted samples (cf. Fig. 3)
# =============================================================================
def _demo(meta_steps: int = 300, seed: int = 0):
    """Regression task where each training example's label is corrupted with a
    random amount of noise. The held-out set is clean. After meta-training, the
    DataRater's scores should be strongly *anti*-correlated with the noise level
    (it never sees the noise level, only (x, y))."""
    torch.manual_seed(seed)
    D = 8
    teacher = torch.randn(D, 1)

    def sample_train(n):
        x = torch.randn(n, D)
        noise_level = torch.rand(n)  # in [0, 1]; hidden
        y = x @ teacher + (3.0 * noise_level).unsqueeze(-1) * torch.randn(n, 1)
        return {"x": x, "y": y, "noise": noise_level}

    def sample_clean(n):
        x = torch.randn(n, D)
        return {"x": x, "y": x @ teacher, "noise": torch.zeros(n)}

    # --- plug-in pieces -----------------------------------------------------
    model_factory = lambda: nn.Sequential(nn.Linear(D, 32), nn.Tanh(), nn.Linear(32, 1))
    mse = lambda apply, b: (apply(b["x"]) - b["y"]).pow(2).squeeze(-1)  # [B]
    rater = MLPDataRater(in_dim=D + 1)
    rater_in = lambda b: (b["x"], b["y"])  # the rater must NOT see "noise"

    class _RaterWrap(nn.Module):  # adapt dict batches -> what the rater expects
        def __init__(self, r):
            super().__init__()
            self.r = r

        def forward(self, b):
            return self.r(rater_in(b))

    wrapped = _RaterWrap(rater)

    cfg = DataRaterConfig(
        num_inner_models=4,
        inner_steps=2,
        unroll_window=2,
        inner_optimizer=lambda: DiffAdam(lr=1e-2),
        meta_lr=3e-3,
        reset_every=200,
    )
    trainer = DataRaterTrainer(
        wrapped,
        model_factory,
        mse,
        mse,
        inner_sampler=lambda: sample_train(128),
        outer_sampler=lambda: sample_clean(128),
        config=cfg,
    )

    for k in range(meta_steps):
        s = trainer.meta_step()
        if k % 50 == 0 or k == meta_steps - 1:
            with torch.no_grad():
                b = sample_train(2000)
                sc = wrapped(b)
                corr = torch.corrcoef(torch.stack([sc, b["noise"]]))[0, 1].item()
            print(
                f"meta step {k:4d} | outer loss {s['outer_loss']:.4f} "
                f"| corr(score, noise) = {corr:+.3f}"
            )

    # --- use the trained DataRater to filter ---------------------------------
    rho, N = 0.5, 128
    big = sample_train(oversampled_batch_size(N, rho))
    kept, _ = topk_filter_batch(wrapped, big, keep=N)
    print(
        f"\nmean noise level  before filter: {big['noise'].mean():.3f}  "
        f"after discarding {rho:.0%}: {kept['noise'].mean():.3f}"
    )

    # per-example acceptance probabilities from the score CDF
    with torch.no_grad():
        ref = wrapped(sample_train(5000))
        cdf = ScoreCDF(ref)
        p_acc = acceptance_probability(cdf(wrapped(big)), batch_size=len(big["x"]), keep=N)
    print(f"mean acceptance prob (should be ~{N / len(big['x']):.2f}): {p_acc.mean():.3f}")


if __name__ == "__main__":
    _demo()
