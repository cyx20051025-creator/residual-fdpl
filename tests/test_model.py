import torch

from cvfdpl import RRDBPlusLightRCAB, count_parameters


def test_3rcab_parameter_count_and_forward() -> None:
    model = RRDBPlusLightRCAB(num_rrdb=4, num_rcab=3)
    assert count_parameters(model) == 197819

    noisy = torch.rand(1, 3, 32, 32)
    output = model(noisy)
    assert output.shape == noisy.shape
    assert torch.isfinite(output).all()
