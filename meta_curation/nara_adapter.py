"""DataRater curation for NARA tabular data (parkinson first).

Phase 1: learn a DataRater on the real prepared train split, use it to filter
that same train pool, and compare downstream test MSE for
full-data vs curated vs random-subsample (same keep rate).

Run:  python -m meta_curation.nara_adapter --meta-steps 200 --keep 0.75
"""

from __future__ import annotations

import argparse
import csv
import os

import torch
import torch.nn as nn

from .config import DataRaterConfig
from .filtering import topk_filter_batch
from .modeling.models import MLPDataRater
from .optim import DiffAdam
from .trainer import DataRaterTrainer

DROP_COLS = {"", "subject."}  # row index + patient id (memorizing it would leak)
TARGET = "target"


def load_csv(path: str) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    """Return (feature_names, X, y) as float tensors, ID columns dropped."""
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    hdr, data = rows[0], rows[1:]
    keep = [i for i, h in enumerate(hdr) if h not in DROP_COLS and h != TARGET]
    ti = hdr.index(TARGET)
    feats = [hdr[i] for i in keep]
    X = torch.tensor([[float(r[i]) for i in keep] for r in data])
    y = torch.tensor([float(r[ti]) for r in data])
    assert torch.isfinite(X).all() and torch.isfinite(y).all(), f"non-finite in {path}"
    return feats, X, y


def split_inner_outer(X: torch.Tensor, y: torch.Tensor, frac: float = 0.8, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(X), generator=g)
    n = int(frac * len(X))
    return (X[perm[:n]], y[perm[:n]]), (X[perm[n:]], y[perm[n:]])


def mlp_factory(d_in: int):
    def _make():
        return nn.Sequential(
            nn.Linear(d_in, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    return _make


def train_eval_mlp(Xtr, ytr, Xte, yte, steps: int = 2000, bs: int = 512, seed: int = 0):
    """Plain supervised MLP (torch.optim) -> test MSE on raw target scale."""
    torch.manual_seed(seed)
    net = mlp_factory(Xtr.shape[1])()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    n = len(Xtr)
    for t in range(steps):
        idx = torch.randint(0, n, (min(bs, n),))
        opt.zero_grad()
        loss_fn(net(Xtr[idx]).squeeze(-1), ytr[idx]).backward()
        opt.step()
    with torch.no_grad():
        return loss_fn(net(Xte).squeeze(-1), yte).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="nara/datasets/prepared/parkinson")
    ap.add_argument("--meta-steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--keep", type=float, default=0.75)
    ap.add_argument("--inner-models", type=int, default=4)
    ap.add_argument("--inner-steps", type=int, default=2)
    ap.add_argument("--eval-seeds", type=int, default=3)
    ap.add_argument("--noise-frac", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="reports/nara_parkinson")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    _, Xtr_raw, ytr_raw = load_csv(os.path.join(args.data_root, "parkinson_train.csv"))
    _, Xte_raw, yte_raw = load_csv(os.path.join(args.data_root, "parkinson_test.csv"))
    D = Xtr_raw.shape[1]
    print(f"train {tuple(Xtr_raw.shape)} test {tuple(Xte_raw.shape)} D={D}")

    (Xi_raw, yi_raw), (Xo_raw, yo_raw) = split_inner_outer(Xtr_raw, ytr_raw, seed=args.seed)
    mi, mo = torch.zeros(len(yi_raw), dtype=torch.bool), torch.zeros(len(yo_raw), dtype=torch.bool)
    if args.noise_frac > 0:  # simulate a dirty pool (demo-style validation)
        mi = _corrupt(yi_raw, args.noise_frac, args.seed + 1)
        mo = _corrupt(yo_raw, args.noise_frac, args.seed + 2)
        print(f"corrupted {mi.sum().item() + mo.sum().item()}/{len(ytr_raw)} train labels")
    mu, sd = Xi_raw.mean(0), Xi_raw.std(0).clamp_min(1e-8)
    ymu, ysd = yi_raw.mean(), yi_raw.std().clamp_min(1e-8)
    Xi, Xo, Xte = (Xi_raw - mu) / sd, (Xo_raw - mu) / sd, (Xte_raw - mu) / sd
    yi, yo, yte = (yi_raw - ymu) / ysd, (yo_raw - ymu) / ysd, (yte_raw - ymu) / ysd

    B = args.batch
    # Fixed probe set for tracking corr(score, noisy) during training.
    Xall_probe = torch.cat([Xi, Xo])
    yall_probe = torch.cat([yi, yo])
    noisy_probe = torch.cat([mi, mo])
    inner_sampler = lambda: _batch(Xi, yi, B)  # noqa: E731
    outer_sampler = lambda: _batch(Xo, yo, B)  # noqa: E731
    mse = lambda apply, b: (apply(b[0]).squeeze(-1) - b[1]).pow(2)  # noqa: E731 ([B])

    rater = MLPDataRater(in_dim=D + 1)
    rater_in = lambda b: (b[0], b[1].unsqueeze(-1))  # noqa: E731

    class _Wrap(nn.Module):
        def __init__(self, r):
            super().__init__()
            self.r = r

        def forward(self, b):
            return self.r(rater_in(b))

    wrapped = _Wrap(rater)
    cfg = DataRaterConfig(
        num_inner_models=args.inner_models,
        inner_steps=args.inner_steps,
        unroll_window=min(2, args.inner_steps),
        inner_optimizer=lambda: DiffAdam(lr=1e-2),  # noqa: E731
        meta_lr=3e-3,
        reset_every=200,
    )
    trainer = DataRaterTrainer(
        wrapped,
        mlp_factory(D),
        mse,
        mse,
        inner_sampler=inner_sampler,
        outer_sampler=outer_sampler,
        config=cfg,
    )
    # Fail fast: per-example loss must be 1-D ([B,B] broadcast = silent garbage).
    probe = mse(
        trainer._make_apply(trainer.population[0], trainer.population[0].params), inner_sampler()
    )
    assert probe.ndim == 1, f"inner loss must be [B], got {tuple(probe.shape)}"
    for k in range(args.meta_steps):
        s = trainer.meta_step()
        if k % 50 == 0 or k == args.meta_steps - 1:
            msg = f"meta {k:4d} | outer {s['outer_loss']:.4f} | |g| {s['meta_grad_norm']:.4f}"
            if mi.any() or mo.any():
                with torch.no_grad():
                    sc = wrapped((Xall_probe, yall_probe))
                c = torch.corrcoef(torch.stack([sc, noisy_probe.float()]))[0, 1].item()
                msg += f" | corr(score, noisy) = {c:+.3f}"
            print(msg, flush=True)

    # Filter: rank UNIQUE rows (deduplicated pool keeps diversity; with-replacement
    # oversampling lets top-K collapse onto a few duplicated high-score rows).
    Xall = torch.cat([Xi, Xo])
    yall = torch.cat([yi, yo])
    keep_n = int(args.keep * len(Xall))
    (kept_x, kept_y), idx = topk_filter_batch(wrapped, (Xall, yall), keep=keep_n)
    print(f"kept {len(kept_x)}/{len(Xall)} (keep={args.keep})")
    is_noisy_all = torch.cat([mi, mo])
    if is_noisy_all.any():
        print(
            f"noisy frac all {is_noisy_all.float().mean():.3f} "
            f"kept {is_noisy_all[idx].float().mean():.3f} (want < all)"
        )

    with torch.no_grad():
        s_all = wrapped((Xall, yall))
        s_kept = wrapped((kept_x, kept_y))
    print(f"score mean all {s_all.mean():.3f} kept {s_kept.mean():.3f}")
    if is_noisy_all.any():
        c = torch.corrcoef(torch.stack([s_all, is_noisy_all.float()]))[0, 1].item()
        print(f"corr(score, is_noisy) = {c:+.3f} (want strongly negative)")
        print(
            f"score mean clean {s_all[~is_noisy_all].mean():.3f} "
            f"noisy {s_all[is_noisy_all].mean():.3f}"
        )

    # Downstream: fresh supervised MLPs on full / curated / random pools.
    g = torch.Generator().manual_seed(1)
    rand_idx = torch.randperm(len(Xall), generator=g)[:keep_n]
    rand_pool = (Xall[rand_idx], yall[rand_idx])
    res = {}
    for name, (X, y) in {
        "full": ((Xall, yall)),
        "curated": ((kept_x, kept_y)),
        "random": (rand_pool),
    }.items():
        mses = [train_eval_mlp(X, y, Xte, yte, seed=s) for s in range(args.eval_seeds)]
        res[name] = sum(mses) / len(mses)
        spread = torch.tensor(mses).std().item() if len(mses) > 1 else 0.0
        print(f"test MSE [{name:7s}] {res[name]:.4f} +- {spread:.4f}")
    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(
        {
            "scores": s_all,
            "is_noisy": is_noisy_all,
            "kept_idx": idx,
            "test_mse": res,
            "args": vars(args),
        },
        os.path.join(args.out_dir, "run.pt"),
    )
    print("saved", args.out_dir + "/run.pt")


def _corrupt(y: torch.Tensor, frac: float, seed: int) -> torch.Tensor:
    """Add 3-sigma label noise to a fraction of rows (raw units); return mask."""
    g = torch.Generator().manual_seed(seed)
    mask = torch.rand(len(y), generator=g) < frac
    y[mask] += torch.randn(mask.sum(), generator=g) * 3.0 * y.std()
    return mask


def _batch(X, y, n):
    idx = torch.randint(0, len(X), (n,))
    return (X[idx], y[idx])


if __name__ == "__main__":
    main()
