"""
Tests for Advanced Structural Analysis detection
تست‌های تشخیص عناصر تحلیل پیشرفته سازه‌ای
"""
import ezdxf
import pytest
from cad3d.architectural_analyzer import ArchitecturalAnalyzer, AdvancedStructuralElementType
from cad3d.dataset_builder import ArchitecturalDatasetBuilder


def _add_block(doc, name: str):
    if name in doc.blocks:
        return
    doc.blocks.new(name=name)


def test_seismic_elements_detection(tmp_path):
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()

    # Layer for seismic
    doc.layers.add("SEISMIC", color=1)

    # Base isolator block
    _add_block(doc, "BASE ISOLATOR")
    msp.add_blockref("BASE ISOLATOR", (1000, 1000), dxfattribs={"layer": "SEISMIC"})

    # Friction damper block
    _add_block(doc, "FRICTION DAMPER")
    msp.add_blockref("FRICTION DAMPER", (2000, 1000), dxfattribs={"layer": "SEISMIC"})

    # Shear wall polygon
    pts = [(0,0), (0,2000), (300,2000), (300,0), (0,0)]
    msp.add_lwpolyline(pts, dxfattribs={"layer": "SEISMIC"}).close()

    dxf_path = tmp_path / "seismic_plan.dxf"
    doc.saveas(dxf_path)

    analyzer = ArchitecturalAnalyzer(str(dxf_path))
    analysis = analyzer.analyze()

    types = [e.element_type for e in analysis.advanced_structural_elements]

    assert AdvancedStructuralElementType.BASE_ISOLATOR in types, "BASE_ISOLATOR not detected"
    assert AdvancedStructuralElementType.FRICTION_DAMPER in types, "FRICTION_DAMPER not detected"
    assert AdvancedStructuralElementType.SHEAR_WALL_REINFORCED in types, "SHEAR_WALL_REINFORCED not detected"


def test_specialized_foundations_detection(tmp_path):
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()

    # Foundation layers
    doc.layers.add("FOUNDATION", color=2)
    doc.layers.add("RAFT", color=3)

    # Pile as circle
    msp.add_circle((1000, 1000, 0), radius=300, dxfattribs={"layer": "FOUNDATION"})

    # Caisson block
    _add_block(doc, "CAISSON")
    msp.add_blockref("CAISSON", (2000, 1200), dxfattribs={"layer": "FOUNDATION"})

    # Raft foundation polygon
    pts = [(0,0), (5000,0), (5000,3000), (0,3000), (0,0)]
    msp.add_lwpolyline(pts, dxfattribs={"layer": "RAFT"}).close()

    dxf_path = tmp_path / "foundation_advanced.dxf"
    doc.saveas(dxf_path)

    analyzer = ArchitecturalAnalyzer(str(dxf_path))
    analysis = analyzer.analyze()

    types = [e.element_type for e in analysis.advanced_structural_elements]

    assert AdvancedStructuralElementType.PILE_FOUNDATION in types, "PILE_FOUNDATION not detected"
    assert AdvancedStructuralElementType.CAISSON in types, "CAISSON not detected"
    assert AdvancedStructuralElementType.RAFT_FOUNDATION in types, "RAFT_FOUNDATION not detected"


def test_advanced_connections_detection(tmp_path):
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()

    doc.layers.add("CONNECTION", color=4)

    # Bolted connection via circle
    msp.add_circle((1000, 1000, 0), radius=50, dxfattribs={"layer": "CONNECTION"})

    # Base plate block
    _add_block(doc, "BASE PLATE")
    msp.add_blockref("BASE PLATE", (1500, 1000), dxfattribs={"layer": "CONNECTION"})

    # Gusset plate block
    _add_block(doc, "GUSSET")
    msp.add_blockref("GUSSET", (2000, 1000), dxfattribs={"layer": "CONNECTION"})

    dxf_path = tmp_path / "connections.dxf"
    doc.saveas(dxf_path)

    analyzer = ArchitecturalAnalyzer(str(dxf_path))
    analysis = analyzer.analyze()

    types = [e.element_type for e in analysis.advanced_structural_elements]

    assert AdvancedStructuralElementType.BOLTED_CONNECTION in types, "BOLTED_CONNECTION not detected"
    assert AdvancedStructuralElementType.BASE_PLATE in types, "BASE_PLATE not detected"
    assert AdvancedStructuralElementType.GUSSET_PLATE in types, "GUSSET_PLATE not detected"


def test_retrofit_elements_detection(tmp_path):
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()

    doc.layers.add("RETROFIT", color=5)

    # Carbon fiber
    _add_block(doc, "CARBON FIBER")
    msp.add_blockref("CARBON FIBER", (1000, 1000), dxfattribs={"layer": "RETROFIT"})

    # Shotcrete
    _add_block(doc, "SHOTCRETE")
    msp.add_blockref("SHOTCRETE", (1500, 1000), dxfattribs={"layer": "RETROFIT"})

    # Post tension
    _add_block(doc, "POST TENSION")
    msp.add_blockref("POST TENSION", (2000, 1000), dxfattribs={"layer": "RETROFIT"})

    # Retrofit zone polygon
    pts = [(0,0), (0,2000), (2000,2000), (2000,0), (0,0)]
    msp.add_lwpolyline(pts, dxfattribs={"layer": "RETROFIT"}).close()

    dxf_path = tmp_path / "retrofit.dxf"
    doc.saveas(dxf_path)

    analyzer = ArchitecturalAnalyzer(str(dxf_path))
    analysis = analyzer.analyze()

    types = [e.element_type for e in analysis.advanced_structural_elements]

    assert AdvancedStructuralElementType.CARBON_FIBER in types, "CARBON_FIBER not detected"
    assert AdvancedStructuralElementType.SHOTCRETE in types, "SHOTCRETE not detected"
    assert AdvancedStructuralElementType.POST_TENSION in types, "POST_TENSION not detected"
    assert AdvancedStructuralElementType.FRP_REINFORCEMENT in types, "FRP_REINFORCEMENT polygon zone not detected"


def test_dataset_builder_counts_for_advanced(tmp_path):
    # Create a simple drawing with one advanced element
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    doc.layers.add("SEISMIC", color=1)
    _add_block(doc, "BASE ISOLATOR")

    msp.add_blockref("BASE ISOLATOR", (1000, 1000), dxfattribs={"layer": "SEISMIC"})

    dxf_path = tmp_path / "advanced_stats.dxf"
    doc.saveas(dxf_path)

    builder = ArchitecturalDatasetBuilder(output_dir=str(tmp_path))
    analysis = builder.process_drawing(str(dxf_path))

    assert len(analysis.advanced_structural_elements) == 1, "Expected one advanced element"
    assert builder.statistics["total_advanced_structural_elements"] == 1, "Statistics not updated"
    # Verify type count key exists
    et = analysis.advanced_structural_elements[0].element_type.value
    assert builder.statistics["advanced_structural_element_types_count"].get(et, 0) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-q"])