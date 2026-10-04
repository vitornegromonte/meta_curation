from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from .optim import DiffAdam, DifferentiableOptimizer


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


@dataclass
class ImplicitRaterConfig(DataRaterConfig):
    """Config for the iMAML-style implicit path (`implicit.py`).

    Inherits population / meta-optimiser / lifetime fields. `unroll_window`
    and `inner_optimizer` are unused here (no unrolled graph; the proximal
    solve uses plain Adam at `inner_lr`). `inner_steps` now means proximal
    solve length (cheap, no graph — use 50-200, not 2).
    """

    inner_steps: int = 100
    inner_lr: float = 1e-2
    proximal_lambda: float = 1.0  # ponytail: guess; tune per task (big = stable but biased)
    # Bookkeeping-only factory for _InnerModel (opt_state unused on this path).
    # NOTE: base default is an instance, not a factory (latent upstream quirk);
    # here we need a real factory since population construction calls it.
    inner_optimizer: Callable[[], DifferentiableOptimizer] = field(
        default_factory=lambda: lambda: DiffAdam(lr=1e-2)
    )
    cg_iters: int = 20
    cg_tol: float = 1e-5
    cg_damping: float = 1e-3
