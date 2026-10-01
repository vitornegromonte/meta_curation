from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import Tensor
import torch.nn as nn

from .types import Batch


def oversampled_batch_size(batch_size: int, discard_fraction: float) -> int:
    """Size N/(1-rho) of the batch to draw so that, after discarding the bottom
    `rho` fraction, exactly ~`batch_size` examples remain."""
    assert 0.0 <= discard_fraction < 1.0
    return math.ceil(batch_size / (1.0 - discard_fraction))


def select_from_batch(batch: Batch, idx: Tensor) -> Batch:
    """Index every tensor in a (nested) batch along dim 0."""
    if isinstance(batch, Tensor):
        return batch[idx.to(batch.device)]
    if isinstance(batch, dict):
        return {k: select_from_batch(v, idx) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(select_from_batch(v, idx) for v in batch)
    raise TypeError(f"Don't know how to index batch of type {type(batch)}")


@torch.no_grad()
def topk_filter_batch(rater: nn.Module, big_batch: Batch, keep: int) -> Tuple[Batch, Tensor]:
    """Batch-level top-K filtering (paper §2 "Data curation").

    Score an oversampled batch and keep the `keep` highest-rated examples.
    Returns (filtered_batch, kept_indices).
    """
    was_training = rater.training
    rater.eval()
    scores = rater(big_batch)
    rater.train(was_training)
    idx = torch.topk(scores, k=keep).indices
    return select_from_batch(big_batch, idx), idx


class ScoreCDF:
    """Empirical CDF F_Phi of DataRater scores over (a sample of) the dataset.

    Lets you make *independent, per-example* keep/discard decisions (e.g. in a
    massively parallel Apache-Beam-style pipeline) that match, in distribution,
    batch-level top-K filtering.
    """

    def __init__(self, scores: Tensor):
        self.sorted = torch.sort(scores.flatten().cpu()).values

    def __call__(self, s: Tensor) -> Tensor:
        """Fraction of reference scores <= s, in [0, 1]."""
        pos = torch.searchsorted(self.sorted, s.cpu().contiguous(), right=True)
        return pos.float() / len(self.sorted)


def acceptance_probability(p: Tensor, batch_size: int, keep: int) -> Tensor:
    """P(keep x) under batch-level top-K filtering, from its score quantile p.

    A point is kept iff at most K-1 of the other B-1 points in its batch beat
    it. Each other point beats it with prob (1-p), so

        P_accept = sum_{s=0}^{K-1} C(B-1, s) (1-p)^s p^(B-1-s)        (paper §2)

    Args:
        p: quantile(s) F_Phi(phi(x)) in [0, 1].
        batch_size: B -- the (oversampled) batch size.
        keep: K -- number of points kept per batch.
    """
    p = p.double().clamp(1e-12, 1 - 1e-12)
    Bm1 = batch_size - 1
    total = torch.zeros_like(p)
    for s in range(keep):
        log_comb = math.lgamma(Bm1 + 1) - math.lgamma(s + 1) - math.lgamma(Bm1 - s + 1)
        total += torch.exp(log_comb + s * torch.log1p(-p) + (Bm1 - s) * torch.log(p))
    return total.clamp(0, 1).float()
