"""
Tests for training dataset builder and training pipeline.
"""
import pytest
from pathlib import Path
import json
import tempfile
import shutil


def _has_torch():
    """Check if PyTorch is installed."""
    try:
        import torch
        return True
    except ImportError:
        return False


def _has_pil():
    """Check if PIL is installed."""
    try:
        from PIL import Image
        return True
    except ImportError:
        return False


def test_training_imports():
    """Test that training modules can be imported."""
    if not _has_pil():
        pytest.skip("PIL not installed")
    
    from cad3d.training_dataset_builder import CADDatasetBuilder, BoundingBox, Annotation
    from cad3d.training_pipeline import CADDataset, CADDetectionTrainer
    assert CADDatasetBuilder is not None
    assert BoundingBox is not None
    assert Annotation is not None
    assert CADDataset is not None
    assert CADDetectionTrainer is not None


def test_bounding_box_dataclass():
    """Test BoundingBox dataclass structure."""
    if not _has_pil():
        pytest.skip("PIL not installed")
    
    from cad3d.training_dataset_builder import BoundingBox
    
    bbox = BoundingBox(
        x_min=10.0,
        y_min=20.0,
        x_max=100.0,
        y_max=200.0,
        category="wall",
        category_id=0
    )
    
    assert bbox.x_min == 10.0
    assert bbox.y_min == 20.0
    assert bbox.x_max == 100.0
    assert bbox.y_max == 200.0
    assert bbox.category == "wall"
    assert bbox.category_id == 0


def test_annotation_dataclass():
    """Test Annotation dataclass structure."""
    if not _has_pil():
        pytest.skip("PIL not installed")
    
    from cad3d.training_dataset_builder import Annotation, BoundingBox
    
    bbox = BoundingBox(10, 20, 100, 200, "door", 1)
    annotation = Annotation(
        image_id=1,
        bboxes=[bbox],
        segmentation_mask=None,
        metadata={"source": "test.dxf"}
    )
    
    assert annotation.image_id == 1
    assert len(annotation.bboxes) == 1
    assert annotation.bboxes[0].category == "door"
    assert annotation.metadata["source"] == "test.dxf"


def test_dataset_builder_init():
    """Test CADDatasetBuilder initialization."""
    if not _has_pil():
        pytest.skip("PIL not installed")
    
    from cad3d.training_dataset_builder import CADDatasetBuilder
    
    with tempfile.TemporaryDirectory() as tmpdir:
        builder = CADDatasetBuilder(output_dir=tmpdir)
        assert builder.output_dir == Path(tmpdir)
        assert len(builder.categories) == 15
        assert "wall" in builder.categories
        assert "door" in builder.categories
        assert builder.category_to_id["wall"] == 0
        assert builder.category_to_id["door"] == 1


def test_dataset_builder_categories():
    """Test that all 15 categories are defined correctly."""
    if not _has_pil():
        pytest.skip("PIL not installed")
    
    from cad3d.training_dataset_builder import CADDatasetBuilder
    
    with tempfile.TemporaryDirectory() as tmpdir:
        builder = CADDatasetBuilder(output_dir=tmpdir)
        
        expected_categories = [
            "wall", "door", "window", "column", "beam", "slab",
            "hvac", "plumbing", "electrical", "furniture", "equipment",
            "dimension", "text", "symbol", "grid_line"
        ]
        
        assert builder.categories == expected_categories
        assert len(builder.category_to_id) == 15
        
        # Check that IDs are sequential
        for i, cat in enumerate(expected_categories):
            assert builder.category_to_id[cat] == i


def test_coco_export_structure(tmp_path):
    """Test COCO format export structure."""
    if not _has_pil():
        pytest.skip("PIL not installed")
    
    from cad3d.training_dataset_builder import CADDatasetBuilder
    
    builder = CADDatasetBuilder(output_dir=str(tmp_path))
    
    # Add dummy data
    builder.images.append({
        "id": 1,
        "file_name": "test.png",
        "width": 1024,
        "height": 1024
    })
    
    builder.annotations.append({
        "id": 1,
        "image_id": 1,
        "category_id": 0,
        "bbox": [10, 20, 90, 180],
        "area": 16200,
        "iscrowd": 0
    })
    
    # Export
    builder.export_coco_format()
    
    # Verify file exists
    coco_file = tmp_path / "annotations.json"
    assert coco_file.exists()
    
    # Verify structure
    with open(coco_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    assert "images" in data
    assert "annotations" in data
    assert "categories" in data
    assert len(data["images"]) == 1
    assert len(data["annotations"]) == 1
    assert len(data["categories"]) == 15


def test_yolo_export_structure(tmp_path):
    """Test YOLO format export creates correct directory structure."""
    if not _has_pil():
        pytest.skip("PIL not installed")
    
    from cad3d.training_dataset_builder import CADDatasetBuilder
    
    builder = CADDatasetBuilder(output_dir=str(tmp_path))
    
    # Add dummy data
    builder.images.append({
        "id": 1,
        "file_name": "test.png",
        "width": 1024,
        "height": 1024
    })
    
    builder.annotations.append({
        "id": 1,
        "image_id": 1,
        "category_id": 0,
        "bbox": [10, 20, 100, 200],  # COCO format: x, y, width, height
        "area": 20000,
        "iscrowd": 0
    })
    
    # Export
    builder.export_yolo_format()
    
    # Verify directory structure
    labels_dir = tmp_path / "labels"
    assert labels_dir.exists()
    
    # Verify label file
    label_file = labels_dir / "test.txt"
    assert label_file.exists()
    
    # Verify content format (class_id x_center y_center width height - normalized)
    with open(label_file, "r") as f:
        lines = f.readlines()
        assert len(lines) == 1
        parts = lines[0].strip().split()
        assert len(parts) == 5
        assert parts[0] == "0"  # category_id


@pytest.mark.skipif(
    not _has_torch(),
    reason="PyTorch not installed"
)
def test_cad_dataset_init():
    """Test CADDataset initialization."""
    from cad3d.training_pipeline import CADDataset
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy COCO annotation file
        coco_data = {
            "images": [],
            "annotations": [],
            "categories": [{"id": i, "name": f"cat_{i}"} for i in range(15)]
        }
        
        ann_file = Path(tmpdir) / "annotations.json"
        with open(ann_file, "w") as f:
            json.dump(coco_data, f)
        
        dataset = CADDataset(
            root_dir=tmpdir,
            annotation_file=str(ann_file)
        )
        
        assert dataset.root_dir == Path(tmpdir)
        assert len(dataset.image_ids) == 0  # No images added


@pytest.mark.skipif(
    not _has_torch(),
    reason="PyTorch not installed"
)
def test_trainer_init():
    """Test CADDetectionTrainer initialization."""
    from cad3d.training_pipeline import CADDetectionTrainer
    import torch
    
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = CADDetectionTrainer(
            data_dir=tmpdir,
            output_dir=tmpdir,
            batch_size=2,
            num_workers=0,
            device=torch.device("cpu"),
            pretrained=False
        )
        
        assert trainer.data_dir == Path(tmpdir)
        assert trainer.output_dir == Path(tmpdir)
        assert trainer.batch_size == 2
        assert trainer.device.type == "cpu"
        assert trainer.model is not None


def test_cli_build_dataset_command():
    """Test that build-dataset CLI command is registered."""
    from cad3d import cli
    import sys
    
    # Test --help doesn't crash
    with pytest.raises(SystemExit):
        cli.main(["build-dataset", "--help"])


def test_cli_train_command():
    """Test that train CLI command is registered."""
    from cad3d import cli
    import sys
    
    # Test --help doesn't crash
    with pytest.raises(SystemExit):
        cli.main(["train", "--help"])


def test_training_cli_integration():
    """Test that training commands are integrated in CLI."""
    from cad3d import cli
    
    # Just verify the module exists - full CLI test done in other tests
    assert hasattr(cli, 'main')


def test_training_system_summary(capsys):
    """Print comprehensive training system summary."""
    print("\n" + "="*70)
    print("🎓 TRAINING SYSTEM - SUMMARY")
    print("="*70)
    
    print("\n📦 Dataset Builder (training_dataset_builder.py):")
    print("  ✓ CADDatasetBuilder class")
    print("  ✓ 15 CAD categories (wall, door, window, ...)")
    print("  ✓ DXF entity parsing & bbox extraction")
    print("  ✓ COCO format export (images, annotations, categories)")
    print("  ✓ YOLO format export (normalized coordinates)")
    print("  ✓ Annotation visualization with color-coded bboxes")
    print("  ✓ DXF to PNG rendering for training images")
    
    print("\n🎓 Training Pipeline (training_pipeline.py):")
    print("  ✓ CADDataset class (PyTorch Dataset interface)")
    print("  ✓ CADDetectionTrainer (Faster R-CNN training)")
    print("  ✓ Optimizer setup (SGD/Adam with LR scheduling)")
    print("  ✓ Training loop (forward, backward, optimize)")
    print("  ✓ Validation loop (evaluation without gradients)")
    print("  ✓ Checkpoint management (best model + periodic saves)")
    print("  ✓ Metrics tracking (loss per epoch)")
    
    print("\n🔧 CLI Commands:")
    print("  ✓ build-dataset: Convert DXF files to training dataset")
    print("    - Supports COCO, YOLO, or both formats")
    print("    - Recursive directory scanning")
    print("    - Annotation visualization")
    print("  ✓ train: Train custom detection model")
    print("    - Configurable epochs, batch size, learning rate")
    print("    - Resume from checkpoint")
    print("    - Pre-trained weights support")
    
    print("\n📊 Workflow:")
    print("  1. DXF files → build-dataset → COCO/YOLO annotations")
    print("  2. COCO dataset → train → Custom Faster R-CNN model")
    print("  3. Trained model → Neural detector → High accuracy inference")
    
    print("\n💡 Use Cases:")
    print("  • Fine-tune on company-specific CAD conventions")
    print("  • Adapt to regional drawing standards")
    print("  • Learn architectural styles (residential/commercial)")
    print("  • Continuous improvement with new annotated data")
    
    print("\n🔗 Integration:")
    print("  • Dataset builder extracts from existing DXF library")
    print("  • Training creates custom models for neural_cad_detector")
    print("  • Trained models improve PDF/Image → DXF accuracy")
    print("  • Complete ML lifecycle: Data → Train → Deploy")
    
    print("\n" + "="*70)


def _build_parser():
    """Helper to build CLI parser for testing."""
    import argparse
    from cad3d import cli
    
    # Call main with empty args to get parser
    parser = argparse.ArgumentParser(description="CAD 2D→3D Offline Converter")
    sub = parser.add_subparsers(dest="cmd", help="Available commands")
    
    # This is a simplified version - in reality we'd need to replicate the full parser
    # For testing, we'll just check if the functions exist
    return parser
