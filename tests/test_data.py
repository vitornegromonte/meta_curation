import torch

from meta_curation.config import DataRaterConfig
from meta_curation.filtering import acceptance_probability, oversampled_batch_size
from meta_curation.optim import DiffAdam, DiffSGD


def test_config_defaults():
    cfg = DataRaterConfig()
    assert 1 <= cfg.unroll_window <= cfg.inner_steps
    assert cfg.num_inner_models == 4


def test_diff_sgd_step():
    torch.manual_seed(0)
    opt = DiffSGD(lr=0.01, momentum=0.9)
    params = {"w": torch.randn(3, 3)}
    grads = {"w": torch.randn(3, 3)}
    new_params, state = opt.step(params, grads, opt.init_state(params))
    assert "buf" in state
    assert torch.allclose(new_params["w"], params["w"] - 0.01 * grads["w"])


def test_diff_adam_finite_second_order():
    # v_hat == 0 case must not produce NaNs in the meta-graph (see DiffAdam docstring)
    opt = DiffAdam()
    params = {"w": torch.zeros(4, requires_grad=True)}
    state = opt.init_state(params)
    new_params, _ = opt.step(params, {"w": torch.zeros(4)}, state)
    assert torch.isfinite(new_params["w"]).all()


def test_filtering_utils():
    assert oversampled_batch_size(128, 0.5) == 256
    p = torch.linspace(0.1, 0.9, 5)
    acc = acceptance_probability(p, batch_size=8, keep=4)
    assert ((acc >= 0) & (acc <= 1)).all()
    assert acc[0] < acc[-1]  # higher score quantile -> higher keep prob
