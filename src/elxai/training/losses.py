"""Training objectives, including the paper's exponential penalty loss.

Paper S6.1 defines three objectives:

* ``MAE`` — ``mean|y - yhat|``, linear penalty, "统计意义更接近于估计条件中位数";
* ``MSE`` — ``mean(y - yhat)^2``, quadratic penalty;
* exponential penalty — ``mean(2^(alpha |y - yhat|) - 1)``, "误差越大的样本获得越高的
  梯度权重".

Two discrepancies in the original implementation are resolved here explicitly
rather than silently inherited:

1. The text says ``取β = 2`` while the prose formula writes ``2^(2|·|)``; the code
   used ``2**(4|·|)``. The exponent is therefore
   :attr:`ExpPenaltyLoss.alpha` (default 4.0, matching the code that produced the
   published table) with :attr:`ExpPenaltyLoss.beta` (2.0) as the base, and both
   are copied into the run directory.
2. The loss is defined on *standardised* errors. Since the dataset is
   standardised, an error of 1.0 already means one standard deviation, so
   ``alpha`` has a scale-free meaning — but only as long as the scaler is applied
   consistently. That is the coupling the paper's footnote about "对不同负荷尺度的
   敏感性" refers to.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

try:  # torch is an optional extra
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - exercised only without the extra
    torch = None  # type: ignore[assignment]
    nn = object  # type: ignore[assignment]


ArrayLike = Any


class ExpPenaltyLoss(nn.Module if torch is not None else object):  # type: ignore[misc]
    """``mean(beta^(alpha |y - yhat|) - 1)``.

    Strictly convex in the absolute error for ``beta > 1``, and grows far faster
    than MSE, so large deviations dominate the gradient. It is unnormalised on
    purpose: the exponential term is not a likelihood, so its absolute value is
    only meaningful relative to other runs with the same parameters.
    """

    def __init__(self, beta: float = 2.0, alpha: float = 4.0):
        if torch is not None:
            super().__init__()
        if beta <= 1.0:
            raise ValueError(f"beta must be > 1 for a convex penalty, got {beta}")
        if alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {alpha}")
        self.beta = float(beta)
        self.alpha = float(alpha)

    def forward(self, pred, target, *args, **kwargs):
        return torch.mean(self.beta ** (self.alpha * torch.abs(pred - target)) - 1.0)

    # Numpy twin so the loss can be unit-tested, and plotted, without torch.
    def numpy(self, pred: np.ndarray, target: np.ndarray) -> float:
        diff = np.abs(np.asarray(pred, dtype=np.float64) - np.asarray(target, dtype=np.float64))
        with np.errstate(over="ignore"):
            return float(np.mean(self.beta ** (self.alpha * diff) - 1.0))

    def __repr__(self) -> str:  # pragma: no cover
        return f"ExpPenaltyLoss(beta={self.beta}, alpha={self.alpha})"


def mse_loss(pred, target, *args, **kwargs):
    if torch is None:  # pragma: no cover
        raise ImportError("torch is required for the neural losses")
    return torch.mean((pred - target) ** 2)


def mae_loss(pred, target, *args, **kwargs):
    if torch is None:  # pragma: no cover
        raise ImportError("torch is required for the neural losses")
    return torch.mean(torch.abs(pred - target))


#: Registry mirrored by the YAML ``training.loss`` field.
LOSSES: dict[str, Callable[..., Any]] = {
    "mse": mse_loss,
    "mae": mae_loss,
}


def build_loss(name: str, **kwargs) -> Callable[..., Any]:
    """Return a callable ``(pred, target) -> scalar tensor``."""
    key = name.lower()
    if key == "exp":
        return ExpPenaltyLoss(
            beta=float(kwargs.get("beta", 2.0)), alpha=float(kwargs.get("alpha", 4.0))
        )
    try:
        return LOSSES[key]
    except KeyError:
        raise KeyError(f"unknown loss {name!r}; known: {sorted(LOSSES) + ['exp']}") from None


def loss_metadata(cfg) -> dict[str, Any]:
    """Everything needed to reinterpret a reported error, taken from a TrainingConfig."""
    meta: dict[str, Any] = {"name": cfg.loss}
    if cfg.loss == "exp":
        meta.update({"beta": cfg.exp_loss_beta, "alpha": cfg.exp_loss_alpha})
    return meta


def gradient_weight_ratio(alpha: float, beta: float, e_small: float, e_large: float) -> float:
    """Ratio of |d loss/d error| at two error magnitudes.

    Makes "指数惩罚更关注大误差" a number instead of an assertion: with the
    paper's parameters and errors of 0.1 and 1.0 standard deviations this is
    ``alpha*ln(beta)*beta**(alpha*e)`` evaluated at both points.
    """
    if beta <= 1:
        raise ValueError("beta must exceed 1")
    g = lambda e: alpha * math.log(beta) * beta ** (alpha * e)  # noqa: E731
    return g(e_large) / g(e_small)
