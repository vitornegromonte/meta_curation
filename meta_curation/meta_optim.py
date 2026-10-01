from __future__ import annotations

from typing import List, Optional, Sequence

import torch
from torch import Tensor


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
