"""Implicit meta-gradients for DataRater (iMAML-style, Rajeswaran et al. 2019).

The explicit path (`trainer.DataRaterTrainer`) back-propagates through T unrolled
inner steps (memory O(T), fragile double-backward). Here the inner problem is
instead solved to (near-)stationarity *without* a graph, under a proximal term:

    L(theta, eta) = sum_i w_i(eta) * l_i(theta) + (lam/2) * ||theta - theta0||^2

At a stationary point the implicit function theorem gives

    d theta*/d eta = -H^{-1} G,   H = d2L/dtheta2,  G = d2L/dtheta deta

so the meta-gradient is  dL_outer/deta = -g^T H^{-1} G  with g = dL_outer/dtheta.
We compute v = H^{-1} g with an iterative solver (CG, HVP-only, damping for
stability) and then a single first-order autograd of (v^T grad_theta_L) w.r.t.
eta. No unrolled graph is ever built: memory is O(1) in the inner steps.

DataRater specifics: eta enters only through the batch softmax weights, so the
weight graph w.r.t. eta is rebuilt once at theta* (one rater forward).
"""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

import torch
from torch import Tensor
from torch.func import functional_call

from .types import Batch, ParamDict, PerExampleLoss


# ---------------------------------------------------------------- flattening
def _shapes(params: Sequence[Tensor]) -> List[torch.Size]:
    return [p.shape for p in params]


def _flatten(tensors: Sequence[Tensor]) -> Tensor:
    return torch.cat([t.reshape(-1) for t in tensors])


def _unflatten(flat: Tensor, shapes: List[torch.Size]) -> List[Tensor]:
    out, pos = [], 0
    for s in shapes:
        n = 1
        for d in s:
            n *= d
        out.append(flat[pos : pos + n].reshape(s))
        pos += n
    return out


def _zeros_like(tensors: Sequence[Tensor]) -> List[Tensor]:
    return [torch.zeros_like(t) for t in tensors]


# --------------------------------------------------------------------- HVP
def hvp(grads: Sequence[Tensor], params: Sequence[Tensor], vec: Sequence[Tensor]) -> List[Tensor]:
    """Hessian-vector product d(grads)/d(params) . vec (one extra autograd layer).

    `grads` must come from an autograd call with create_graph=True. Nones
    (unused paths) are treated as zeros.
    """
    out = torch.autograd.grad(
        list(grads),
        list(params),
        grad_outputs=[v for v in vec],
        retain_graph=True,
        allow_unused=True,
    )
    return [torch.zeros_like(p) if o is None else o for p, o in zip(params, out)]


# ------------------------------------------------------- inverse-HVP solvers
class InverseHVPSolver:
    """Solves H . v = b given only HVP access. Subclass for algorithms.

    Contract is on flat vectors: `hvp_flat(x)` returns Hx (same shape as x).
    """

    def solve(self, hvp_flat: Callable[[Tensor], Tensor], b_flat: Tensor) -> Tensor:
        raise NotImplementedError


class ConjugateGradientSolver(InverseHVPSolver):
    """Conjugate gradients for (H + damping*I) . v = b.

    Damping keeps CG stable when H is only PSD (common for neural nets).
    """

    def __init__(self, iters: int = 20, tol: float = 1e-5, damping: float = 1e-3):
        self.iters, self.tol, self.damping = iters, tol, damping

    def solve(self, hvp_flat: Callable[[Tensor], Tensor], b_flat: Tensor) -> Tensor:
        x = torch.zeros_like(b_flat)
        r = b_flat - (hvp_flat(x) + self.damping * x)
        p = r.clone()
        rs = torch.dot(r, r)
        if rs.sqrt().item() < self.tol:
            return x
        for _ in range(self.iters):
            Ap = hvp_flat(p) + self.damping * p
            pAp = torch.dot(p, Ap)
            if pAp.abs().item() < 1e-12:
                break  # non-PD direction: stop rather than divide by ~0
            alpha = rs / pAp
            x = x + alpha * p
            r = r - alpha * Ap
            rs_new = torch.dot(r, r)
            if rs_new.sqrt().item() < self.tol:
                break
            p = r + (rs_new / rs) * p
            rs = rs_new
        return x


# ------------------------------------------------------- proximal inner solve
def proximal_inner_solve(
    module: torch.nn.Module,
    buffers: ParamDict,
    init: ParamDict,
    batch: Batch,
    weights: Tensor,
    inner_loss_fn: PerExampleLoss,
    steps: int,
    lr: float,
    lam: float,
    anchor: ParamDict,
) -> ParamDict:
    """Minimize sum(w*l) + (lam/2)||theta-anchor||^2 with plain Adam (no graph).

    `weights` are frozen (computed once, no grad). Returns detached theta*.
    """
    params = {k: v.detach().clone().requires_grad_(True) for k, v in init.items()}
    opt = torch.optim.Adam(list(params.values()), lr=lr)
    apply = lambda *a, **k: functional_call(module, {**params, **buffers}, a, k)  # noqa: E731
    for _ in range(steps):
        opt.zero_grad()
        per_ex = inner_loss_fn(apply, batch)
        assert per_ex.ndim == 1, f"inner loss must be [B], got {tuple(per_ex.shape)}"
        prox = (lam / 2) * sum(((params[k] - anchor[k]) ** 2).sum() for k in params)
        ((weights * per_ex).sum() + prox).backward()
        opt.step()
    return {k: v.detach().clone() for k, v in params.items()}


# ------------------------------------------------------- implicit meta-gradient
def implicit_update(
    trainer,  # DataRaterTrainer (duck-typed: needs _weights/_make_apply/samplers/losses)
    m,
    eta: Sequence[Tensor],
    solver: InverseHVPSolver,
    inner_steps: int,
    inner_lr: float,
    lam: float,
) -> Tuple[ParamDict, List[Tensor], float]:
    """One implicit bilevel update for a single inner model.

    Returns (theta_star detached, meta-gradients dL_outer/deta, outer loss).
    """
    inner_batch = trainer.inner_sampler()
    with torch.no_grad():
        w_frozen = trainer._weights(inner_batch, with_grad=False)
    anchor = {k: v.detach().clone() for k, v in m.params.items()}

    theta_star = proximal_inner_solve(
        m.module,
        m.buffers,
        m.params,
        inner_batch,
        w_frozen,
        trainer.inner_loss_fn,
        inner_steps,
        inner_lr,
        lam,
        anchor,
    )
    leaves = {k: v.detach().requires_grad_(True) for k, v in theta_star.items()}
    apply = trainer._make_apply(m, leaves)

    outer_batch = trainer.outer_sampler()
    outer_per = trainer.outer_loss_fn(apply, outer_batch)
    outer = outer_per.mean() if outer_per.ndim else outer_per
    g = torch.autograd.grad(outer, list(leaves.values()))

    # Rebuild the inner loss with eta-attached weights: this is the G path.
    w_eta = trainer._weights(inner_batch, with_grad=True)
    per_ex = trainer.inner_loss_fn(apply, inner_batch)
    prox = (lam / 2) * sum(((leaves[k] - anchor[k]) ** 2).sum() for k in leaves)
    L_inner = (w_eta * per_ex).sum() + prox
    grads_theta = torch.autograd.grad(L_inner, list(leaves.values()), create_graph=True)

    shapes = _shapes(list(leaves.values()))
    g_flat = _flatten([x.detach() for x in g])

    def hvp_flat(x_flat: Tensor) -> Tensor:
        vec = _unflatten(x_flat, shapes)
        return _flatten(hvp(grads_theta, list(leaves.values()), vec))

    v_flat = solver.solve(hvp_flat, g_flat)
    v = [x.detach() for x in _unflatten(v_flat, shapes)]
    s = sum((vi * gi).sum() for vi, gi in zip(v, grads_theta))
    meta = torch.autograd.grad(s, list(eta), allow_unused=True)
    meta = [torch.zeros_like(p) if x is None else -x for p, x in zip(eta, meta)]
    return theta_star, meta, float(outer.detach())


# -----------------------------------------------------------------------------
# Self-test: implicit meta-gradient must agree (directionally) with the
# explicit unrolled one on a toy. Run:  python -m meta_curation.implicit
# -----------------------------------------------------------------------------
def _selftest(seed: int = 0):
    import torch.nn as nn

    from .config import DataRaterConfig
    from .modeling.models import MLPDataRater
    from .optim import DiffAdam
    from .trainer import DataRaterTrainer

    torch.manual_seed(seed)
    D, H, B = 4, 8, 16
    teacher = torch.randn(D, 1)

    def sample(n):
        x = torch.randn(n, D)
        nl = torch.rand(n)
        y = x @ teacher + (2.0 * nl).unsqueeze(-1) * torch.randn(n, 1)
        return (x, y)

    factory = lambda: nn.Sequential(nn.Linear(D, H), nn.Tanh(), nn.Linear(H, 1))  # noqa: E731
    mse = lambda apply, b: (apply(b[0]).squeeze(-1) - b[1].squeeze(-1)).pow(2)  # noqa: E731
    rater = MLPDataRater(in_dim=D + 1)

    # Explicit reference: long unroll so it approximates the true bilevel gradient.
    cfg = DataRaterConfig(
        num_inner_models=1,
        inner_steps=50,
        unroll_window=50,
        inner_optimizer=lambda: DiffAdam(lr=1e-2),  # noqa: E731
        meta_lr=0.0,
        reset_every=None,
    )
    tr = DataRaterTrainer(
        rater,
        factory,
        mse,
        mse,
        inner_sampler=lambda: sample(B),
        outer_sampler=lambda: sample(B),  # noqa: E731
        config=cfg,
    )
    m = tr.population[0]
    new_params = tr._update_inner_model(m)
    outer = mse(tr._make_apply(m, new_params), tr.outer_sampler()).mean()
    eta = list(rater.parameters())
    g_exp = torch.autograd.grad(outer, eta)
    e_flat = _flatten([x.detach() for x in g_exp])

    # Implicit: proximal solve + CG.
    anchor = {k: v.detach().clone() for k, v in m.params.items()}
    batch = sample(B)
    with torch.no_grad():
        w0 = torch.softmax(rater(batch), dim=0)
    theta_star = proximal_inner_solve(
        m.module,
        m.buffers,
        m.params,
        batch,
        w0,
        mse,
        steps=200,
        lr=1e-2,
        lam=1.0,
        anchor=anchor,
    )
    leaves = {k: v.detach().requires_grad_(True) for k, v in theta_star.items()}
    apply = tr._make_apply(m, leaves)
    g_outer = torch.autograd.grad(mse(apply, tr.outer_sampler()).mean(), list(leaves.values()))
    w_eta = torch.softmax(rater(batch), dim=0)
    per = mse(apply, batch)
    prox = 0.5 * sum(((leaves[k] - anchor[k]) ** 2).sum() for k in leaves)
    Li = (w_eta * per).sum() + prox
    gt = torch.autograd.grad(Li, list(leaves.values()), create_graph=True)
    shapes = _shapes(list(leaves.values()))
    solver = ConjugateGradientSolver(iters=50, tol=1e-8, damping=1e-3)
    v_flat = solver.solve(
        lambda x: _flatten(hvp(gt, list(leaves.values()), _unflatten(x, shapes))),
        _flatten([x.detach() for x in g_outer]),
    )
    v = [x.detach() for x in _unflatten(v_flat, shapes)]
    s = sum((vi * gi).sum() for vi, gi in zip(v, gt))
    g_imp = torch.autograd.grad(s, eta, allow_unused=True)
    i_flat = _flatten([torch.zeros_like(p) if x is None else -x for p, x in zip(eta, g_imp)])

    cos = torch.dot(e_flat, i_flat) / (e_flat.norm() * i_flat.norm() + 1e-12)
    print(f"cos(explicit-T50, implicit) = {cos.item():+.3f}")
    assert cos.item() > 0.9, "implicit meta-gradient disagrees with explicit!"
    print("OK: implicit meta-gradient matches the unrolled one.")


if __name__ == "__main__":
    _selftest()
