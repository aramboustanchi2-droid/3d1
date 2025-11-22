"""
Smoke Test: Deep Hybrid CLI
Quick integration test for deep-hybrid command with ddim_steps=0 for speed.
"""

import sys
from pathlib import Path
import numpy as np
import cv2

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cad3d.cli import main as cli_main


def test_deep_hybrid_smoke(tmp_path=None):
    """Quick smoke test for deep-hybrid CLI with minimal diffusion steps."""
    if tmp_path is None:
        tmp_path = Path('outputs/smoke_test')
    tmp_path.mkdir(parents=True, exist_ok=True)
    
    # Create synthetic image
    img = np.ones((256, 256, 3), dtype=np.uint8) * 255
    cv2.putText(img, 'SMOKE', (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    img_path = tmp_path / 'smoke_img.png'
    cv2.imwrite(str(img_path), img)
    print(f"✓ Created test image: {img_path}")
    
    # Run CLI
    out_dxf = tmp_path / 'smoke_out.dxf'
    args = [
        'deep-hybrid',
        '--input', str(img_path),
        '--output', str(out_dxf),
        '--prior-strength', '0.5',
        '--ddim-steps', '0',
        '--normalize-range', '0', '100'
    ]
    
    print(f"Running CLI: deep-hybrid {' '.join(args[1:])}")
    try:
        cli_main(args)
    except SystemExit:
        pass  # CLI may call sys.exit
    
    # Verify outputs
    assert out_dxf.exists(), f"DXF not created: {out_dxf}"
    print(f"✓ DXF created: {out_dxf}")
    
    metadata = out_dxf.with_suffix('.deep_hybrid.json')
    assert metadata.exists(), f"Metadata not created: {metadata}"
    print(f"✓ Metadata created: {metadata}")
    
    import json
    data = json.loads(metadata.read_text(encoding='utf-8'))
    assert 'dxf' in data
    assert 'prior_strength' in data
    assert data['prior_strength'] == 0.5
    assert data['ddim_steps'] == 0
    print(f"✓ Metadata valid: {len(data.get('final_points_data', []))} points")
    
    print("\n✅ Smoke test PASSED")


if __name__ == '__main__':
    print("="*70)
    print("DEEP-HYBRID CLI SMOKE TEST")
    print("="*70)
    test_deep_hybrid_smoke()
