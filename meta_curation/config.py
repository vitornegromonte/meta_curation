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
