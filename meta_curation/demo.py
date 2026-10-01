from __future__ import annotations

import torch
import torch.nn as nn

from .config import DataRaterConfig
from .filtering import (
    ScoreCDF,
    acceptance_probability,
    oversampled_batch_size,
    topk_filter_batch,
)
from .modeling.models import MLPDataRater
from .optim import DiffAdam
from .trainer import DataRaterTrainer


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
