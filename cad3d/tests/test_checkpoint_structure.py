"""
Comprehensive checkpoint and directory structure verification test.
Tests:
1. VAE trainer checkpoint creation
2. Diffusion trainer checkpoint creation
3. Dataset directory structure
4. Optional weight loading (graceful handling when missing)
"""

import sys
from pathlib import Path
import shutil
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
import cv2


def test_vae_checkpoint_structure():
    """Test VAE trainer creates proper directories and checkpoints."""
    print("\n" + "="*70)
    print("TEST 1: VAE Checkpoint Structure")
    print("="*70)
    
    from cad3d.vae_trainer import VAETrainer, CAD2D3DDataset
    from cad3d.diffusion_trainer import create_synthetic_training_data
    
    # Create test data
    data_dir = Path('outputs/test_vae_checkpoint/data')
    if not (data_dir / 'images').exists():
        print("Creating synthetic test data...")
        create_synthetic_training_data(data_dir, num_samples=5)
    
    # Create trainer with specific save_dir
    save_dir = Path('outputs/test_vae_checkpoint/models')
    if save_dir.exists():
        shutil.rmtree(save_dir)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = VAETrainer(
        device=device,
        latent_dim=128,
        num_points=512,
        save_dir=save_dir,
        use_voxel_loss=False
    )
    
    # Train for 2 epochs
    print("\nTraining VAE for 2 epochs...")
    trainer.train(data_dir=data_dir, epochs=2, batch_size=2)
    
    # Verify checkpoint structure
    print("\n✓ Checking checkpoint files...")
    
    expected_files = [
        save_dir / "vae_best.pth",
        save_dir / "vae_epoch_1.pth",
        save_dir / "vae_epoch_2.pth",
        save_dir / "vae_epoch_log.json",
        save_dir / "vae_training_report.json"
    ]
    
    missing = []
    for f in expected_files:
        if f.exists():
            print(f"  ✓ {f.name} exists ({f.stat().st_size} bytes)")
        else:
            print(f"  ✗ {f.name} MISSING")
            missing.append(f.name)
    
    # Verify checkpoint content
    print("\n✓ Checking checkpoint content...")
    ckpt = torch.load(save_dir / "vae_best.pth", map_location='cpu')
    required_keys = ['epoch', 'state_dict', 'opt', 'scheduler', 'val_loss', 'loss_history', 'kl_weight']
    
    for key in required_keys:
        if key in ckpt:
            print(f"  ✓ checkpoint['{key}'] present")
        else:
            print(f"  ✗ checkpoint['{key}'] MISSING")
            missing.append(key)
    
    # Verify epoch log
    print("\n✓ Checking epoch log...")
    with open(save_dir / "vae_epoch_log.json", 'r', encoding='utf-8') as f:
        logs = json.load(f)
    print(f"  ✓ {len(logs)} epoch entries logged")
    for entry in logs:
        assert 'epoch' in entry and 'train_loss' in entry and 'kl_weight' in entry
    
    if missing:
        print(f"\n✗ VAE TEST FAILED: {len(missing)} items missing")
        return False
    else:
        print("\n✅ VAE TEST PASSED: All checkpoints and logs created correctly")
        return True


def test_diffusion_checkpoint_structure():
    """Test Diffusion trainer creates proper directories and checkpoints."""
    print("\n" + "="*70)
    print("TEST 2: Diffusion Checkpoint Structure")
    print("="*70)
    
    from cad3d.diffusion_trainer import DiffusionTrainer, CAD2D3DDataset, create_synthetic_training_data
    from cad3d.diffusion_3d_model import create_diffusion_model
    
    # Create test data
    data_dir = Path('outputs/test_diffusion_checkpoint/data')
    if not (data_dir / 'images').exists():
        print("Creating synthetic test data...")
        create_synthetic_training_data(data_dir, num_samples=5)
    
    # Create dataset
    dataset = CAD2D3DDataset(data_dir=data_dir, image_size=256, num_points=1024, augment=False)
    
    # Create model and trainer
    save_dir = Path('outputs/test_diffusion_checkpoint/models')
    if save_dir.exists():
        shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = create_diffusion_model(num_points=1024, timesteps=100, device=device)
    trainer = DiffusionTrainer(model=model, device=device, save_dir=save_dir)
    
    # Train for 2 epochs (save_every=1 to get epoch checkpoints)
    print("\nTraining Diffusion for 2 epochs...")
    trainer.train(train_dataset=dataset, epochs=2, batch_size=2, save_every=1)
    
    # Verify checkpoint structure
    print("\n✓ Checking checkpoint files...")
    
    expected_files = [
        save_dir / "diffusion_best.pth",
        save_dir / "diffusion_epoch_1.pth",
        save_dir / "diffusion_epoch_2.pth",
        save_dir / "training_report.json"
    ]
    
    missing = []
    for f in expected_files:
        if f.exists():
            print(f"  ✓ {f.name} exists ({f.stat().st_size} bytes)")
        else:
            print(f"  ✗ {f.name} MISSING")
            missing.append(f.name)
    
    # Verify checkpoint content
    print("\n✓ Checking checkpoint content...")
    ckpt = torch.load(save_dir / "diffusion_best.pth", map_location='cpu')
    required_keys = ['epoch', 'global_step', 'image_encoder_state', 'unet_state', 
                     'optimizer_state', 'scheduler_state', 'loss', 'best_loss', 'loss_history']
    
    for key in required_keys:
        if key in ckpt:
            print(f"  ✓ checkpoint['{key}'] present")
        else:
            print(f"  ✗ checkpoint['{key}'] MISSING")
            missing.append(key)
    
    # Verify training report
    print("\n✓ Checking training report...")
    with open(save_dir / "training_report.json", 'r') as f:
        report = json.load(f)
    assert 'total_epochs' in report and 'best_loss' in report
    print(f"  ✓ Report: {report['total_epochs']} epochs, best_loss={report['best_loss']:.6f}")
    
    if missing:
        print(f"\n✗ DIFFUSION TEST FAILED: {len(missing)} items missing")
        return False
    else:
        print("\n✅ DIFFUSION TEST PASSED: All checkpoints and reports created correctly")
        return True


def test_dataset_directory_structure():
    """Test dataset creation and structure."""
    print("\n" + "="*70)
    print("TEST 3: Dataset Directory Structure")
    print("="*70)
    
    from cad3d.diffusion_trainer import create_synthetic_training_data
    
    data_dir = Path('outputs/test_dataset_structure')
    if data_dir.exists():
        shutil.rmtree(data_dir)
    
    print("Creating synthetic dataset...")
    create_synthetic_training_data(data_dir, num_samples=10)
    
    # Verify directory structure
    print("\n✓ Checking directory structure...")
    
    expected_dirs = [
        data_dir / 'images',
        data_dir / 'pointclouds'
    ]
    
    missing = []
    for d in expected_dirs:
        if d.exists() and d.is_dir():
            print(f"  ✓ {d.name}/ exists")
        else:
            print(f"  ✗ {d.name}/ MISSING")
            missing.append(d.name)
    
    # Verify file counts
    print("\n✓ Checking file counts...")
    images = list((data_dir / 'images').glob('*.png'))
    pointclouds = list((data_dir / 'pointclouds').glob('*.npy'))
    
    print(f"  ✓ {len(images)} images created")
    print(f"  ✓ {len(pointclouds)} point clouds created")
    
    if len(images) != 10 or len(pointclouds) != 10:
        print(f"  ✗ Expected 10 of each, got {len(images)} images and {len(pointclouds)} point clouds")
        missing.append("file_count_mismatch")
    
    # Verify pairing
    print("\n✓ Checking image-pointcloud pairing...")
    for img in images:
        pc = data_dir / 'pointclouds' / f"{img.stem}.npy"
        if not pc.exists():
            print(f"  ✗ Missing point cloud for {img.name}")
            missing.append(f"missing_pc_{img.stem}")
    
    if not missing:
        print("  ✓ All images have corresponding point clouds")
    
    if missing:
        print(f"\n✗ DATASET TEST FAILED: {len(missing)} issues found")
        return False
    else:
        print("\n✅ DATASET TEST PASSED: Directory structure correct")
        return True


def test_optional_weight_loading():
    """Test optional pretrained weight loading (graceful handling when missing)."""
    print("\n" + "="*70)
    print("TEST 4: Optional Weight Loading")
    print("="*70)
    
    from cad3d.vae_integration import VAEConverter
    from cad3d.hybrid_vit_diffusion import HybridCAD3DConverter
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test 1: VAEConverter with missing checkpoint
    print("\n✓ Test VAEConverter with missing checkpoint...")
    try:
        converter = VAEConverter(device=device, checkpoint=Path('nonexistent_checkpoint.pth'))
        print("  ✓ VAEConverter created successfully (checkpoint missing is OK)")
    except Exception as e:
        print(f"  ✗ VAEConverter failed with missing checkpoint: {e}")
        return False
    
    # Test 2: VAEConverter with None checkpoint
    print("\n✓ Test VAEConverter with None checkpoint...")
    try:
        converter = VAEConverter(device=device, checkpoint=None)
        print("  ✓ VAEConverter created successfully (no checkpoint)")
    except Exception as e:
        print(f"  ✗ VAEConverter failed with None checkpoint: {e}")
        return False
    
    # Test 3: HybridConverter with missing checkpoints
    print("\n✓ Test HybridConverter with missing checkpoints...")
    try:
        converter = HybridCAD3DConverter(
            device=device,
            vit_model_path=Path('nonexistent_vit.pth'),
            diffusion_model_path=Path('nonexistent_diffusion.pth'),
            enable_learning=False
        )
        print("  ✓ HybridConverter created successfully (checkpoints missing is OK)")
    except Exception as e:
        print(f"  ✗ HybridConverter failed with missing checkpoints: {e}")
        return False
    
    # Test 4: HybridConverter with None checkpoints
    print("\n✓ Test HybridConverter with None checkpoints...")
    try:
        converter = HybridCAD3DConverter(
            device=device,
            vit_model_path=None,
            diffusion_model_path=None,
            enable_learning=False
        )
        print("  ✓ HybridConverter created successfully (no checkpoints)")
    except Exception as e:
        print(f"  ✗ HybridConverter failed with None checkpoints: {e}")
        return False
    
    print("\n✅ OPTIONAL WEIGHT LOADING TEST PASSED: All converters handle missing weights gracefully")
    return True


def main():
    """Run all checkpoint structure tests."""
    print("\n" + "="*70)
    print("COMPREHENSIVE CHECKPOINT STRUCTURE VERIFICATION")
    print("="*70)
    
    results = {}
    
    try:
        results['vae_checkpoints'] = test_vae_checkpoint_structure()
    except Exception as e:
        print(f"\n✗ VAE TEST CRASHED: {e}")
        import traceback
        traceback.print_exc()
        results['vae_checkpoints'] = False
    
    try:
        results['diffusion_checkpoints'] = test_diffusion_checkpoint_structure()
    except Exception as e:
        print(f"\n✗ DIFFUSION TEST CRASHED: {e}")
        import traceback
        traceback.print_exc()
        results['diffusion_checkpoints'] = False
    
    try:
        results['dataset_structure'] = test_dataset_directory_structure()
    except Exception as e:
        print(f"\n✗ DATASET TEST CRASHED: {e}")
        import traceback
        traceback.print_exc()
        results['dataset_structure'] = False
    
    try:
        results['optional_weights'] = test_optional_weight_loading()
    except Exception as e:
        print(f"\n✗ OPTIONAL WEIGHTS TEST CRASHED: {e}")
        import traceback
        traceback.print_exc()
        results['optional_weights'] = False
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {test_name}")
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == '__main__':
    exit(main())
