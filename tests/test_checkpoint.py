import pytest
import torch
from torch import nn

from cvfdpl.training import EMAModel


def test_ema_update_and_copy() -> None:
    model = nn.Linear(2, 1, bias=False)
    model.weight.data.fill_(1.0)
    ema = EMAModel(model, decay=0.5)
    model.weight.data.fill_(3.0)
    ema.update(model)
    assert torch.equal(ema.shadow["weight"], torch.full_like(model.weight, 2.0))

    model.weight.data.zero_()
    ema.copy_to(model)
    assert torch.equal(model.weight.detach(), torch.full_like(model.weight, 2.0))


def test_ema_rejects_invalid_decay() -> None:
    with pytest.raises(ValueError):
        EMAModel(nn.Linear(2, 1), decay=1.0)
