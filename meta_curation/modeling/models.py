from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor
import torch.nn as nn


class MLPDataRater(nn.Module):
    """DataRater for vector inputs. `batch` is a tuple/list of tensors that are
    flattened and concatenated per example (e.g. (x, y) for supervised data)."""

    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, batch) -> Tensor:
        if isinstance(batch, Tensor):
            batch = (batch,)
        z = torch.cat([b.flatten(1) for b in batch], dim=-1)
        return self.net(z).squeeze(-1)


class TransformerDataRater(nn.Module):
    """Non-causal Transformer DataRater for token sequences (as in the paper).

    `batch` is a LongTensor [B, S]. Tokens are embedded, passed through
    bidirectional self-attention layers, mean-pooled over (non-pad) positions
    and mapped to one scalar score. (The paper uses a 50M-parameter model; this
    is just a compact, readable stand-in.)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model=128,
        n_heads=4,
        n_layers=2,
        max_len=2048,
        pad_id: Optional[int] = None,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, 4 * d_model, dropout=0.0, batch_first=True, norm_first=True
        )
        self.enc = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.head = nn.Linear(d_model, 1)

    def forward(self, tokens: Tensor) -> Tensor:
        B, S = tokens.shape
        h = self.tok(tokens) + self.pos(torch.arange(S, device=tokens.device))
        pad = (tokens == self.pad_id) if self.pad_id is not None else None
        h = self.enc(h, src_key_padding_mask=pad)  # no causal mask => non-causal
        if pad is not None:
            keep = (~pad).unsqueeze(-1).float()
            h = (h * keep).sum(1) / keep.sum(1).clamp(min=1)
        else:
            h = h.mean(1)
        return self.head(h).squeeze(-1)
