from __future__ import annotations

from typing import Any, Callable, Dict

from torch import Tensor

ParamDict = Dict[str, Tensor]
Batch = Any  # tensor / tuple / dict -- opaque to the trainer
ApplyFn = Callable[..., Any]
PerExampleLoss = Callable[[ApplyFn, Batch], Tensor]
