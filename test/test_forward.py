import torch
from src.andx.models.model_and import ModelCfg, build_model, ANDCriterion

def test_forward():
    model = build_model(ModelCfg())
    x = torch.randn(8, 3, 32, 32)
    _, z = model(x)
    loss = ANDCriterion(0.1)(z)
    assert z.shape[0] == 8 and loss.item() > 0
