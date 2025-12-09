import torch
from src.models.simple_model import SimpleModel
from src.utils.metrics import mse

def test_model_forward():
    model = SimpleModel()
    input_tensor = torch.randn(1, 10)
    output = model(input_tensor)
    assert output.shape == (1, 1)

def test_metrics():
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([1.0, 2.0])
    assert mse(a, b) == 0.0
