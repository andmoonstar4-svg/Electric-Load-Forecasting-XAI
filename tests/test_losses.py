"""Losses: the paper's three objectives, including the exponential penalty."""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from elxai.config import TrainingConfig  # noqa: E402
from elxai.training.losses import (  # noqa: E402
    ExpPenaltyLoss,
    build_loss,
    gradient_weight_ratio,
    loss_metadata,
)


def test_mse_and_mae_match_numpy_reference():
    pred = torch.tensor([[1.0, 2.0, 3.0]])
    true = torch.tensor([[1.5, 1.0, 4.0]])
    assert float(build_loss("mse")(pred, true)) == pytest.approx(
        float(np.mean((np.array([1.0, 2.0, 3.0]) - np.array([1.5, 1.0, 4.0])) ** 2))
    )
    assert float(build_loss("mae")(pred, true)) == pytest.approx(
        float(np.mean(np.abs(np.array([1.0, 2.0, 3.0]) - np.array([1.5, 1.0, 4.0]))))
    )


def test_loss_names_are_case_insensitive():
    assert build_loss("MSE") is build_loss("mse")


def test_unknown_loss_raises():
    with pytest.raises(KeyError, match="unknown loss"):
        build_loss("huber")


def test_exp_penalty_matches_the_paper_formula():
    """loss = mean(beta^(alpha|e|) - 1); zero error must give exactly zero."""
    loss = ExpPenaltyLoss(beta=2.0, alpha=4.0)
    pred = torch.tensor([[0.0, 1.0, 2.0]])
    true = torch.tensor([[0.0, 1.0, 2.0]])
    assert float(loss(pred, true)) == pytest.approx(0.0)

    err = torch.tensor([[0.25, 0.5]])
    expected = np.mean(2.0 ** (4.0 * np.abs(np.array([0.25, 0.5]))) - 1.0)
    assert float(loss(err, torch.zeros_like(err))) == pytest.approx(expected)


def test_exp_penalty_numpy_twin_agrees_with_torch():
    loss = ExpPenaltyLoss(beta=2.0, alpha=4.0)
    err = np.array([[0.1, 0.9], [0.3, 1.4]])
    assert loss.numpy(err, np.zeros_like(err)) == pytest.approx(
        float(loss(torch.tensor(err), torch.zeros_like(torch.tensor(err))))
    )


def test_exp_penalty_grows_faster_than_mse_for_large_errors():
    """The whole point of the objective: large deviations dominate the gradient."""
    err = torch.linspace(0.0, 2.0, 401, requires_grad=True)
    zeros = torch.zeros_like(err)

    exp_val = ExpPenaltyLoss(beta=2.0, alpha=4.0)(err, zeros)
    exp_val.backward()
    grad_exp = err.grad.detach().numpy().copy()

    err2 = torch.linspace(0.0, 2.0, 401, requires_grad=True)
    build_loss("mse")(err2, torch.zeros_like(err2)).backward()
    grad_mse = err2.grad.detach().numpy()

    # Normalise each at e = 0.1 and compare at e = 1.5. For MSE the ratio is exactly
    # 15; for the exponential penalty it is 2^(4 * 1.4) ~ 48.5.
    i_small, i_large = 20, 300
    ratio_exp = grad_exp[i_large] / grad_exp[i_small]
    ratio_mse = grad_mse[i_large] / grad_mse[i_small]
    assert ratio_mse == pytest.approx(15.0, rel=1e-6)
    assert ratio_exp == pytest.approx(2.0 ** (4.0 * 1.4), rel=1e-6)
    assert ratio_exp > 3 * ratio_mse


def test_gradient_weight_ratio_is_analytic():
    """Closed form: d/de [beta^(alpha e) - 1] = alpha ln(beta) beta^(alpha e)."""
    alpha, beta = 4.0, 2.0
    expected = beta ** (alpha * 1.0) / beta ** (alpha * 0.1)
    assert gradient_weight_ratio(alpha, beta, 0.1, 1.0) == pytest.approx(expected)


def test_exp_penalty_rejects_non_convex_base():
    with pytest.raises(ValueError, match="beta must be > 1"):
        ExpPenaltyLoss(beta=1.0, alpha=4.0)
    with pytest.raises(ValueError, match="alpha must be > 0"):
        ExpPenaltyLoss(beta=2.0, alpha=0.0)


def test_exp_penalty_is_differentiable_everywhere():
    """No in-place ops, so it can be used with autograd and AMP."""
    loss = ExpPenaltyLoss()
    pred = torch.randn(4, 24, requires_grad=True)
    target = torch.randn(4, 24)
    value = loss(pred, target)
    value.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_loss_metadata_records_the_exponent():
    """The paper says beta = 2 but the code used alpha = 4; both must be recorded."""
    meta = loss_metadata(TrainingConfig(loss="exp", exp_loss_beta=2.0, exp_loss_alpha=4.0))
    assert meta == {"name": "exp", "beta": 2.0, "alpha": 4.0}
    assert loss_metadata(TrainingConfig(loss="mse")) == {"name": "mse"}


def test_scaling_invariance_argument():
    """alpha is only scale-free while errors are standardised.

    This documents the coupling the paper flags ("对不同负荷尺度的敏感性"): on raw
    load (thousands of MW) an error of 1.0 saturates the exponential immediately.
    """
    loss = ExpPenaltyLoss(beta=2.0, alpha=4.0)
    standardised = loss.numpy(np.array([[1.0]]), np.array([[0.0]]))
    raw_scale = loss.numpy(np.array([[1000.0]]), np.array([[0.0]]))
    assert math.isinf(raw_scale) or raw_scale > 1e300
    assert standardised == pytest.approx(15.0)
