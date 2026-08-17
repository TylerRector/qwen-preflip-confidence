import torch
from src.experiment import project

def test_projection_removes_rows():
    torch.manual_seed(9)
    v = torch.randn(18)
    rows = [torch.randn(18), torch.randn(18)]
    p = project(v, rows, ridge=1e-10)
    matrix = torch.stack(rows)
    assert float(torch.max(torch.abs(matrix @ p))) < 1e-5

def test_semantic_position_coordinates():
    original = torch.tensor(-2.0)
    swapped = torch.tensor(-4.0)
    position = 0.5 * (original + swapped)
    semantic = 0.5 * (original - swapped)
    assert torch.allclose(position + semantic, original)
    assert torch.allclose(position - semantic, swapped)
