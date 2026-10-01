from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
from torch import Tensor

from .types import ParamDict


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
