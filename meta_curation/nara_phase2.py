"""Phase 2: DataRater vs NARA Data-IQ head-to-head on a synthetic parkinson pool.

Stand-in for NARA's REaLTabFormer pool (which needs their GPU/Colab stack):
smoothed-bootstrap synthetic rows (real row + feature noise) mixed with junk
rows (shuffled targets / noise rows), with a known junk mask.

Arms (same keep count, same downstream eval on the REAL test split):
  datarater : inner sampler = synth pool, outer sampler = held-out REAL data
  dataiq-lr : faithful port of nara's sculpt_with_dataiq path
              (LinearRegression refit x epochs -> aleatoric/confidence ->
              stratify_samples percentile path; drops `hard`)
  dataiq-mlp: same, but warm-started MLPRegressor so aleatoric is non-degenerate
  random    : uniform subsample at the same keep count

NARA sources: profiling_synth_data/utils/{dataiqreg,sculping_data}.py
(DataIQ_SKLearn, fit_dataiq_sk, stratify_samples).

Run:  python -m meta_curation.nara_phase2 --meta-steps 600
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

from .config import DataRaterConfig
from .modeling.models import MLPDataRater
from .nara_adapter import _batch, load_csv, mlp_factory, split_inner_outer, train_eval_mlp
from .optim import DiffAdam
from .trainer import DataRaterTrainer


def make_synth_pool(X, y, n_synth: int, junk_frac: float, seed: int = 0):
    """Smoothed bootstrap + junk. X, y in RAW units. Returns (Xs, ys, is_junk)."""
    g = torch.Generator().manual_seed(seed)
    n = len(X)
    src = torch.randint(0, n, (n_synth,), generator=g)
    Xs = X[src].clone()
    ys = y[src].clone()
    is_junk = torch.zeros(n_synth, dtype=torch.bool)
    n_junk = int(junk_frac * n_synth)
    jidx = torch.randperm(n_synth, generator=g)[:n_junk]
    is_junk[jidx] = True
    # clean rows: small feature jitter; junk rows: shuffled target + big noise
    Xs += torch.randn_like(Xs) * 0.05 * X.std(0).clamp_min(1e-8)
    perm = torch.randperm(n_synth, generator=g)
    ys[jidx] = y[src[perm[:n_junk]]]
    Xs[jidx] += torch.randn_like(Xs[jidx]) * X.std(0).clamp_min(1e-8)
    return Xs, ys, is_junk


def dataiq_stratify(X_np, y_np, model_fn, epochs: int = 10, seed: int = 0):
    """Port of nara fit_dataiq_sk + stratify_samples (percentile path, p50).

    Returns (easy_idx, ambig_idx, hard_idx) with keep = easy + ambiguous.
    """
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_np)
    rng = np.random.RandomState(seed)
    n = len(Xs)
    tr = rng.choice(n, int(0.8 * n), replace=False)
    preds = []
    for _ in range(epochs):
        clf = model_fn()
        clf.fit(Xs[tr], y_np[tr])
        preds.append(clf.predict(Xs))
    P = np.stack(preds, axis=1)
    aleatoric = P.var(axis=1)
    confidence = P.mean(axis=1)
    below = aleatoric <= np.percentile(aleatoric, 50)
    hard = np.where((confidence <= 0.25) & below)[0]
    easy = np.where((confidence >= 0.75) & below)[0]
    ambig = np.setdiff1d(np.arange(n), np.concatenate([hard, easy]))
    return easy, ambig, hard


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="nara/datasets/prepared/parkinson")
    ap.add_argument("--dataset", default="parkinson")
    ap.add_argument("--n-synth", type=int, default=8000)
    ap.add_argument("--junk-frac", type=float, default=0.3)
    ap.add_argument("--meta-steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="reports/nara_phase2")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    _, Xtr_raw, ytr_raw = load_csv(os.path.join(args.data_root, f"{args.dataset}_train.csv"))
    _, Xte_raw, yte_raw = load_csv(os.path.join(args.data_root, f"{args.dataset}_test.csv"))
    D = Xtr_raw.shape[1]

    (Xi_raw, yi_raw), (Xo_raw, yo_raw) = split_inner_outer(Xtr_raw, ytr_raw, seed=args.seed)
    mu, sd = Xi_raw.mean(0), Xi_raw.std(0).clamp_min(1e-8)
    ymu, ysd = yi_raw.mean(), yi_raw.std().clamp_min(1e-8)
    Xo, yo = (Xo_raw - mu) / sd, (yo_raw - ymu) / ysd
    Xte, yte = (Xte_raw - mu) / sd, (yte_raw - ymu) / ysd

    # Synthetic pool built from the INNER real split only (outer stays clean).
    Xs_raw, ys_raw, is_junk = make_synth_pool(
        Xi_raw, yi_raw, args.n_synth, args.junk_frac, seed=args.seed
    )
    Xs, ys = (Xs_raw - mu) / sd, (ys_raw - ymu) / ysd
    print(f"synth pool {tuple(Xs.shape)} junk frac {is_junk.float().mean():.3f}")

    # ---- arm 1: DataRater (inner = synth, outer = held-out real) ----
    B = args.batch
    mse = lambda apply, b: (apply(b[0]).squeeze(-1) - b[1]).pow(2)  # noqa: E731 ([B])
    rater = MLPDataRater(in_dim=D + 1)

    class _Wrap(nn.Module):
        def __init__(self, r):
            super().__init__()
            self.r = r

        def forward(self, b):
            return self.r((b[0], b[1].unsqueeze(-1)))

    wrapped = _Wrap(rater)
    cfg = DataRaterConfig(
        num_inner_models=4,
        inner_steps=2,
        unroll_window=2,
        inner_optimizer=lambda: DiffAdam(lr=1e-2),  # noqa: E731
        meta_lr=3e-3,
        reset_every=200,
    )
    trainer = DataRaterTrainer(
        wrapped,
        mlp_factory(D),
        mse,
        mse,
        inner_sampler=lambda: _batch(Xs, ys, B),  # noqa: E731
        outer_sampler=lambda: _batch(Xo, yo, B),  # noqa: E731
        config=cfg,
    )
    probe = mse(
        trainer._make_apply(trainer.population[0], trainer.population[0].params),
        _batch(Xs, ys, B),
    )
    assert probe.ndim == 1, f"inner loss must be [B], got {tuple(probe.shape)}"
    for k in range(args.meta_steps):
        s = trainer.meta_step()
        if k % 100 == 0 or k == args.meta_steps - 1:
            with torch.no_grad():
                sc = wrapped((Xs, ys))
            c = torch.corrcoef(torch.stack([sc, is_junk.float()]))[0, 1].item()
            print(
                f"meta {k:4d} | outer {s['outer_loss']:.4f} | corr(score, junk) = {c:+.3f}",
                flush=True,
            )

    # ---- arm 2+3: Data-IQ baselines (CPU ports of nara's sculpt path) ----
    # NOTE: raw units, exactly as upstream (their confidence thresholds 0.25/0.75
    # operate on raw target scale).
    Xn, yn = Xs_raw.numpy(), ys_raw.numpy()
    arms: dict[str, np.ndarray] = {}
    for name, fn in {
        "dataiq-lr": lambda: LinearRegression(),  # noqa: E731 (nara's exact path)
        "dataiq-mlp": lambda: MLPRegressor(hidden_layer_sizes=(64,), max_iter=200, random_state=0),  # noqa: E731
    }.items():
        easy, ambig, hard = dataiq_stratify(Xn, yn, fn, epochs=10, seed=args.seed)
        keep = np.concatenate([easy, ambig])
        arms[name] = keep
        jrate = is_junk.numpy()[keep].mean() if len(keep) else float("nan")
        print(
            f"{name}: easy {len(easy)} ambig {len(ambig)} hard {len(hard)} "
            f"-> keep {len(keep)} junk-in-kept {jrate:.3f}"
        )

    n_keep = min(len(v) for v in arms.values())
    print(f"common keep count: {n_keep}")
    arms = {k: v[:n_keep] for k, v in arms.items()}  # easy-first truncation
    with torch.no_grad():
        scores = wrapped((Xs, ys))
    _, dr_idx = torch.topk(scores, k=n_keep)
    dr_idx = dr_idx.numpy()
    arms["datarater"] = dr_idx
    # Fixed aggressive budget pair (independent of Data-IQ's keep count).
    n_fix = int(0.7 * len(Xs))
    _, dr70 = torch.topk(scores, k=n_fix)
    arms["datarater70"] = dr70.numpy()
    g = torch.Generator().manual_seed(1)
    arms["random"] = torch.randperm(len(Xs), generator=g)[:n_keep].numpy()
    arms["random70"] = torch.randperm(len(Xs), generator=g)[:n_fix].numpy()
    for name, idx in arms.items():
        print(f"{name:10s} keep {len(idx):5d} junk-in-kept {is_junk.numpy()[idx].mean():.3f}")

    # ---- downstream: fresh MLPs on each kept set, tested on REAL test ----
    res = {}
    for name, idx in arms.items():
        mses = [train_eval_mlp(Xs[idx], ys[idx], Xte, yte, seed=s) for s in range(3)]
        res[name] = sum(mses) / len(mses)
        print(f"test MSE [{name:10s}] {res[name]:.4f} +- {np.std(mses):.4f}")
    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(
        {"scores": scores, "is_junk": is_junk, "arms": arms, "test_mse": res, "args": vars(args)},
        os.path.join(args.out_dir, "run.pt"),
    )
    print("saved", args.out_dir + "/run.pt")


if __name__ == "__main__":
    main()
