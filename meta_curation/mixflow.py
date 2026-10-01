"""
MixFlow-MG: mixed-mode differentiation for meta-gradients
"""

from __future__ import annotations

from typing import Callable, Tuple

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable
from torch.func import grad as func_grad
from torch.func import jvp as func_jvp

from .types import ParamDict

LossFn = Callable[[ParamDict, Tensor], Tensor]  # (params, eta) -> scalar loss


class _Spec:
    """Carries non-tensor state (the loss function and the param ordering)
    through `torch.autograd.Function.apply`, which only differentiates
    tensor arguments."""

    def __init__(self, loss_fn: LossFn, keys: Tuple[str, ...]):
        self.loss_fn = loss_fn
        self.keys = keys


class _FwdRevGrad(torch.autograd.Function):
    """y = dL/dtheta (as a tuple of tensors, one per parameter) with a custom
    VJP that uses forward-over-reverse instead of reverse-over-reverse."""

    @staticmethod
    def forward(ctx, spec: _Spec, eta: Tensor, *flat_params: Tensor):
        params = dict(zip(spec.keys, flat_params))
        ctx.spec = spec
        # Only the *inputs* are saved -- no activations of the inner model.
        ctx.save_for_backward(eta, *flat_params)
        # Plain reverse-mode gradient of the loss wrt params. torch.func.grad is
        # a function transform, so it works even though Function.forward runs
        # with autograd disabled.
        grads = func_grad(spec.loss_fn, argnums=0)(params, eta)
        return tuple(grads[k] for k in spec.keys)

    @staticmethod
    @once_differentiable
    def backward(ctx, *cts: Tensor):
        spec = ctx.spec
        eta, *flat_params = ctx.saved_tensors
        params = dict(zip(spec.keys, flat_params))
        ct = dict(zip(spec.keys, cts))  # cotangents, shaped like the params

        # f(theta, eta) = (dL/dtheta, dL/deta)
        grad_both = func_grad(spec.loss_fn, argnums=(0, 1))

        # One forward-mode JVP of f along the tangent (ct, 0):
        #     d f / d theta . ct = ( d2L/dtheta2 . ct ,  d2L/(deta dtheta) . ct )
        # By symmetry of the Hessian and of the mixed-derivative matrix these
        # are the transposed products we need:
        #     ct^T d2L/dtheta2          -> cotangent for theta
        #     ct^T d2L/(dtheta deta)    -> cotangent for eta
        _, (hvp, d_eta) = func_jvp(lambda p: grad_both(p, eta), (params,), (ct,))

        # Order: (spec, eta, *params)
        return (None, d_eta, *(hvp[k] for k in spec.keys))


def fwdrev_grad(loss_fn: LossFn, params: ParamDict, eta: Tensor) -> ParamDict:
    """Gradient of `loss_fn(params, eta)` wrt `params`, differentiable (once)
    wrt both `params` and `eta` using mixed-mode (forward-over-reverse) AD.

    Drop-in replacement for
        torch.autograd.grad(loss_fn(params, eta), params.values(), create_graph=True)

    Args:
        loss_fn: pure function (params_dict, eta) -> scalar loss. Anything else
            it needs (data batch, buffers, the module) should be closed over and
            must NOT require grad.
        params: dict name -> tensor (may be non-leaf tensors carrying a graph
            back to eta from earlier inner steps).
        eta: tensor differentiated through (e.g. DataRater scores, shape [B]).
    """
    keys = tuple(params.keys())
    out = _FwdRevGrad.apply(_Spec(loss_fn, keys), eta, *params.values())
    return dict(zip(keys, out))


# Self-test: mixed-mode meta-gradient must equal the default implementation.
# Run:  python -m meta_curation.mixflow
def _selftest(T: int = 3, B: int = 16, D: int = 5, H: int = 7, seed: int = 0):
    torch.manual_seed(seed)
    dt = torch.float64  # float64 so that the comparison is tight
    X = torch.randn(T, B, D, dtype=dt)
    Y = torch.randn(T, B, 1, dtype=dt)
    Xv, Yv = torch.randn(32, D, dtype=dt), torch.randn(32, 1, dtype=dt)
    theta0 = {
        "W1": torch.randn(D, H, dtype=dt) * 0.5,
        "b1": torch.zeros(H, dtype=dt),
        "W2": torch.randn(H, 1, dtype=dt) * 0.5,
        "b2": torch.zeros(1, dtype=dt),
    }
    eta_net = torch.nn.Linear(D + 1, 1).to(dt)  # a tiny "DataRater"

    def predict(p, x):
        return torch.tanh(x @ p["W1"] + p["b1"]) @ p["W2"] + p["b2"]

    def meta_grad(mode: str):
        eta_net.zero_grad()
        p = {k: v.clone().requires_grad_(True) for k, v in theta0.items()}
        mom = {k: torch.zeros_like(v) for k, v in p.items()}
        for t in range(T):
            scores = eta_net(torch.cat([X[t], Y[t]], -1)).squeeze(-1)  # [B]

            def loss_fn(params, s, t=t):
                per_ex = (predict(params, X[t]) - Y[t]).pow(2).squeeze(-1)
                return (torch.softmax(s, 0) * per_ex).sum()

            if mode == "reverse":
                gl = torch.autograd.grad(loss_fn(p, scores), list(p.values()), create_graph=True)
                g = dict(zip(p.keys(), gl))
            else:
                g = fwdrev_grad(loss_fn, p, scores)
            mom = {k: 0.9 * mom[k] + g[k] for k in p}  # SGD + momentum
            p = {k: p[k] - 0.1 * mom[k] for k in p}
        outer = (predict(p, Xv) - Yv).pow(2).mean()
        gs = torch.autograd.grad(outer, list(eta_net.parameters()))
        return torch.cat([x.flatten() for x in gs])

    g_ref, g_mix = meta_grad("reverse"), meta_grad("mixflow")
    err = (g_ref - g_mix).abs().max().item()
    rel = err / g_ref.abs().max().item()
    print(f"max |reverse - mixflow| = {err:.3e}  (relative {rel:.3e})")
    assert rel < 1e-8, "mixed-mode meta-gradient disagrees with the default one!"
    print("OK: MixFlow-MG meta-gradient matches reverse-over-reverse.")


if __name__ == "__main__":
    _selftest()
