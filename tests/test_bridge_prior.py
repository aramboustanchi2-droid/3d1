"""
Unit Test: VAE Prior Bridge Effectiveness
Tests that using VAE prior initialization (prior_strength > 0) reduces MSE
compared to pure noise initialization (prior_strength = 0).

We run with ddim_steps=0 to bypass diffusion for speed and determinism, making
the final output equal to the prior/noise initialization.
"""

import numpy as np
import pytest
from pathlib import Path
import sys

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Optional heavy deps
cv2 = pytest.importorskip('cv2')
torch = pytest.importorskip('torch')

from cad3d.hybrid_vae_vit_diffusion import create_deep_hybrid_converter


@pytest.fixture
def synthetic_test_image(tmp_path):
    img = np.ones((256, 256, 3), dtype=np.uint8) * 255
    cv2.putText(img, 'TEST', (30,140), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,0), 3)
    img_path = tmp_path / "test_cad.png"
    cv2.imwrite(str(img_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return img_path


def _to_mm(points_arr, mm_range):
    pts = np.asarray(points_arr, dtype=np.float32)
    pmin = float(pts.min()); pmax = float(pts.max()); pr = pmax - pmin
    if pr > 1e-6:
        pts_n = 2.0 * (pts - pmin) / pr - 1.0
    else:
        pts_n = pts
    lo, hi = mm_range
    return ((pts_n * 0.5) + 0.5) * (hi - lo) + lo


def test_prior_reduces_mse_with_ddim_zero(synthetic_test_image, tmp_path):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    rng = (0.0, 1000.0)

    conv0 = create_deep_hybrid_converter(device=dev, prior_strength=0.0, normalize_range=rng, ddim_steps=0)
    res0 = conv0.convert(synthetic_test_image, tmp_path / 'no_prior.dxf')

    conv1 = create_deep_hybrid_converter(device=dev, prior_strength=0.7, normalize_range=rng, ddim_steps=0)
    res1 = conv1.convert(synthetic_test_image, tmp_path / 'with_prior.dxf')

    prior0 = np.array(res0['prior_points_data'], dtype=np.float32)
    prior1 = np.array(res1['prior_points_data'], dtype=np.float32)
    final0 = np.array(res0['final_points_data'], dtype=np.float32)
    final1 = np.array(res1['final_points_data'], dtype=np.float32)

    # Convert priors to mm to match final's units
    prior0_mm = _to_mm(prior0, rng)
    prior1_mm = _to_mm(prior1, rng)

    mse0 = float(np.mean((final0 - prior0_mm) ** 2))
    mse1 = float(np.mean((final1 - prior1_mm) ** 2))

    assert mse1 < mse0, f"Expected MSE with prior (0.7) < with pure noise (0.0); got {mse1} vs {mse0}"
