"""
Tests for Construction Phasing & Project Staging detection
فاز ۱۰: مراحل پروژه و ساخت
"""
import pytest
import ezdxf
import tempfile
from pathlib import Path

from cad3d.architectural_analyzer import (
    ArchitecturalAnalyzer,
    ConstructionPhasingElementType,
)


@pytest.fixture
def temp_dxf():
    """Create a temporary DXF file"""
    with tempfile.TemporaryDirectory() as tmpdir:
        dxf_path = Path(tmpdir) / "test_phasing.dxf"
        yield dxf_path


def test_phase_markers_and_boundaries(temp_dxf):
    """تست تشخیص نشانگرهای فاز (PHASE 1، 2، 3) و مرزهای فاز"""
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()

    doc.layers.add("PHASE-1")
    doc.layers.add("PHASE-2")

    # Phase markers as text
    msp.add_text("PHASE 1", dxfattribs={"layer": "PHASE-1"}).set_placement((1000, 1000))
    msp.add_text("PHASE 2 - FOUNDATION", dxfattribs={"layer": "PHASE-2"}).set_placement((5000, 1000))
    msp.add_text("PHASE 3", dxfattribs={"layer": "PHASE-2"}).set_placement((10000, 1000))

    # Phase boundary as polygon
    msp.add_lwpolyline([(0, 0), (3000, 0), (3000, 3000), (0, 3000), (0, 0)], dxfattribs={"layer": "PHASE-1"})

    doc.saveas(temp_dxf)

    analyzer = ArchitecturalAnalyzer(str(temp_dxf))
    analysis = analyzer.analyze()

    phase_markers = [e for e in analysis.construction_phasing_elements if e.element_type == ConstructionPhasingElementType.PHASE_MARKER]
    phase_boundaries = [e for e in analysis.construction_phasing_elements if e.element_type == ConstructionPhasingElementType.PHASE_BOUNDARY]

    assert len(phase_markers) >= 3
    assert len(phase_boundaries) >= 1
    # Check phase numbers extracted
    phase_numbers = [m.phase_number for m in phase_markers if m.phase_number]
    assert 1 in phase_numbers
    assert 2 in phase_numbers
    assert 3 in phase_numbers


def test_demolition_zones(temp_dxf):
    """تست تشخیص مناطق تخریب، دیوارها و موارد خطرناک"""
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()

    doc.layers.add("DEMO")
    doc.layers.add("DEMOLITION")

    # Demolition zone as polygon
    msp.add_lwpolyline([(1000, 1000), (3000, 1000), (3000, 3000), (1000, 3000), (1000, 1000)], dxfattribs={"layer": "DEMO"})

    # Demolition wall as line
    msp.add_line((5000, 0), (5000, 3000), dxfattribs={"layer": "DEMO"})

    # Hazmat text
    msp.add_text("HAZMAT - Asbestos Removal", dxfattribs={"layer": "DEMOLITION"}).set_placement((7000, 2000))

    # Salvage item
    msp.add_text("SALVAGE ITEM", dxfattribs={"layer": "DEMOLITION"}).set_placement((8000, 2000))

    doc.saveas(temp_dxf)

    analyzer = ArchitecturalAnalyzer(str(temp_dxf))
    analysis = analyzer.analyze()

    demo_zones = [e for e in analysis.construction_phasing_elements if e.element_type == ConstructionPhasingElementType.DEMOLITION_ZONE]
    hazmat = [e for e in analysis.construction_phasing_elements if e.element_type == ConstructionPhasingElementType.HAZMAT_AREA]
    salvage = [e for e in analysis.construction_phasing_elements if e.element_type == ConstructionPhasingElementType.SALVAGE_ITEM]

    assert len(demo_zones) >= 1
    assert len(hazmat) >= 1
    assert len(salvage) >= 1


def test_temporary_structures(temp_dxf):
    """تست تشخیص سازه‌های موقت: دیوار، حصار، شمع‌کوبی، داربست"""
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()

    doc.layers.add("TEMP-WALL")
    doc.layers.add("TEMP-FENCE")
    doc.layers.add("SHORING")
    doc.layers.add("SCAFFOLD")

    # Temporary wall
    msp.add_line((0, 0), (5000, 0), dxfattribs={"layer": "TEMP-WALL"})

    # Temporary fence
    msp.add_lwpolyline([(1000, 1000), (2000, 1000), (2000, 2000)], dxfattribs={"layer": "TEMP-FENCE"})

    # Shoring
    msp.add_line((3000, 0), (3000, 3000), dxfattribs={"layer": "SHORING"})

    # Scaffolding
    msp.add_lwpolyline([(4000, 0), (5000, 0), (5000, 1000)], dxfattribs={"layer": "SCAFFOLD"})

    doc.saveas(temp_dxf)

    analyzer = ArchitecturalAnalyzer(str(temp_dxf))
    analysis = analyzer.analyze()

    temp_walls = [e for e in analysis.construction_phasing_elements if e.element_type == ConstructionPhasingElementType.TEMPORARY_WALL]
    temp_fences = [e for e in analysis.construction_phasing_elements if e.element_type == ConstructionPhasingElementType.TEMPORARY_FENCE]
    shoring = [e for e in analysis.construction_phasing_elements if e.element_type == ConstructionPhasingElementType.SHORING]
    scaffolding = [e for e in analysis.construction_phasing_elements if e.element_type == ConstructionPhasingElementType.SCAFFOLDING]

    assert len(temp_walls) >= 1
    assert len(temp_fences) >= 1
    assert len(shoring) >= 1
    assert len(scaffolding) >= 1


def test_construction_sequence(temp_dxf):
    """تست تشخیص توالی ساخت، بتن‌ریزی، نصب"""
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()

    doc.layers.add("SEQUENCE")
    doc.layers.add("CONSTRUCTION")

    # Pour sequence
    msp.add_text("POUR SEQUENCE 1", dxfattribs={"layer": "SEQUENCE"}).set_placement((1000, 1000))
    msp.add_text("POUR 2", dxfattribs={"layer": "SEQUENCE"}).set_placement((2000, 1000))

    # Erection sequence
    msp.add_text("ERECTION SEQUENCE 1", dxfattribs={"layer": "CONSTRUCTION"}).set_placement((3000, 1000))

    # General construction sequence
    msp.add_text("CONSTRUCTION SEQUENCE 3", dxfattribs={"layer": "CONSTRUCTION"}).set_placement((4000, 1000))

    doc.saveas(temp_dxf)

    analyzer = ArchitecturalAnalyzer(str(temp_dxf))
    analysis = analyzer.analyze()

    pour_seq = [e for e in analysis.construction_phasing_elements if e.element_type == ConstructionPhasingElementType.POUR_SEQUENCE]
    erection_seq = [e for e in analysis.construction_phasing_elements if e.element_type == ConstructionPhasingElementType.ERECTION_SEQUENCE]
    construction_seq = [e for e in analysis.construction_phasing_elements if e.element_type == ConstructionPhasingElementType.CONSTRUCTION_SEQUENCE]

    assert len(pour_seq) >= 2
    assert len(erection_seq) >= 1
    assert len(construction_seq) >= 1
    # Check sequence numbers
    assert pour_seq[0].sequence_order == 1
    assert pour_seq[1].sequence_order == 2


def test_staging_and_work_areas(temp_dxf):
    """تست تشخیص مناطق کاری: آماده‌سازی، ذخیره، موقعیت جرثقیل"""
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()

    doc.layers.add("STAGING")
    doc.layers.add("LAYDOWN")
    doc.layers.add("CRANE")

    # Staging area
    msp.add_lwpolyline([(0, 0), (5000, 0), (5000, 5000), (0, 5000), (0, 0)], dxfattribs={"layer": "STAGING"})

    # Laydown area
    msp.add_lwpolyline([(6000, 0), (10000, 0), (10000, 4000), (6000, 4000), (6000, 0)], dxfattribs={"layer": "LAYDOWN"})

    # Crane location as block
    msp.add_blockref("CRANE-1", insert=(12000, 2000), dxfattribs={"layer": "CRANE"})

    doc.saveas(temp_dxf)

    analyzer = ArchitecturalAnalyzer(str(temp_dxf))
    analysis = analyzer.analyze()

    staging = [e for e in analysis.construction_phasing_elements if e.element_type == ConstructionPhasingElementType.STAGING_AREA]
    laydown = [e for e in analysis.construction_phasing_elements if e.element_type == ConstructionPhasingElementType.LAYDOWN_AREA]
    cranes = [e for e in analysis.construction_phasing_elements if e.element_type == ConstructionPhasingElementType.CRANE_LOCATION]

    assert len(staging) >= 1
    assert len(laydown) >= 1
    assert len(cranes) >= 1


def test_phasing_metadata(temp_dxf):
    """تست متادیتا: num_construction_phasing_elements"""
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()

    doc.layers.add("PHASE")
    doc.layers.add("DEMO")
    doc.layers.add("TEMP")

    # Add various phasing elements
    msp.add_text("PHASE 1", dxfattribs={"layer": "PHASE"}).set_placement((1000, 1000))
    msp.add_lwpolyline([(0, 0), (2000, 0), (2000, 2000), (0, 2000), (0, 0)], dxfattribs={"layer": "DEMO"})
    msp.add_line((3000, 0), (5000, 0), dxfattribs={"layer": "TEMP"})

    doc.saveas(temp_dxf)

    analyzer = ArchitecturalAnalyzer(str(temp_dxf))
    analysis = analyzer.analyze()

    assert "num_construction_phasing_elements" in analysis.metadata
    assert analysis.metadata["num_construction_phasing_elements"] >= 3
