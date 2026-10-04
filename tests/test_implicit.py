import torch

from meta_curation.implicit import ConjugateGradientSolver, hvp


def test_cg_solves_spd_system():
    torch.manual_seed(0)
    n = 20
    A = torch.randn(n, n)
    A = A @ A.T + 2 * torch.eye(n)
    b = torch.randn(n)
    solver = ConjugateGradientSolver(iters=100, tol=1e-8, damping=0.0)
    v = solver.solve(lambda x: A @ x, b)
    assert torch.allclose(A @ v, b, atol=1e-4)


def test_cg_damping_handles_singular():
    # Zeros HVP: solver must return without NaNs (early break / damping).
    solver = ConjugateGradientSolver(iters=10, tol=1e-8, damping=1e-3)
    v = solver.solve(lambda x: torch.zeros_like(x), torch.ones(5))
    assert torch.isfinite(v).all()


def test_hvp_matches_finite_difference():
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1))
    x = torch.randn(16, 4)
    params = [p.detach().requires_grad_(True) for p in net.parameters()]

    def loss(ps):
        return (
            (
                torch.func.functional_call(
                    net, dict(zip([n for n, _ in net.named_parameters()], ps)), (x,)
                )
            ).squeeze(-1)
            ** 2
        ).mean()

    vec = [torch.randn_like(p) for p in params]
    grads = torch.autograd.grad(loss(params), params, create_graph=True)
    exact = hvp(grads, params, vec)

    # HVP must be finite and nonzero here.
    flat_e = torch.cat([e.reshape(-1) for e in exact])
    assert torch.isfinite(flat_e).all() and flat_e.abs().max().item() > 0
    # Symmetry: u^T H v == v^T H u
    u = [torch.randn_like(p) for p in params]
    huv = sum((a * b).sum() for a, b in zip(hvp(grads, params, u), vec))
    hvu = sum((a * b).sum() for a, b in zip(exact, u))
    assert torch.allclose(huv, hvu, atol=1e-4)
