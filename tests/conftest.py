import pytest
from pathlib import Path

needs_data = pytest.mark.skipif(
    not Path("data/Segmentation").exists(),
    reason="Dataset not available in this environment",
)

needs_model = pytest.mark.skipif(
    not Path("models/best_model.pth").exists(),
    reason="Model weights not available in this environment",
)