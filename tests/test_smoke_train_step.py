import torch

from cvfdpl import BrightnessAwareLoss, PerChannelLogFDPLoss, RRDBPlusLightRCAB


def test_one_cpu_optimization_step_is_finite() -> None:
    model = RRDBPlusLightRCAB(num_rrdb=2, num_rcab=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    noisy = torch.rand(1, 3, 32, 32)
    clean = noisy - 0.02
    prediction = model(noisy)
    mse = BrightnessAwareLoss()
    fdpl = PerChannelLogFDPLoss(
        torch.ones(3, 64, 64),
        torch.ones(3, 256, 256),
    )
    loss = mse(prediction, clean) + fdpl(prediction, clean, noisy)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
