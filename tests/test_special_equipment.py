"""
Tests for Special Equipment detection
تست‌های تشخیص تجهیزات ویژه
"""
import ezdxf
import pytest
from cad3d.architectural_analyzer import ArchitecturalAnalyzer, SpecialEquipmentElementType


def _ensure_block(doc, name: str):
    if name in doc.blocks:
        return
    doc.blocks.new(name=name)


def run_analyze(doc, tmp_path):
    p = tmp_path / "special_equip_test.dxf"
    doc.saveas(p)
    analyzer = ArchitecturalAnalyzer(str(p))
    return analyzer.analyze()


def test_vertical_transport_detection(tmp_path):
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    doc.layers.add("ELEVATOR", color=3)
    _ensure_block(doc, "ELEVATOR-CAR")
    _ensure_block(doc, "ESCALATOR-UNIT")
    msp.add_blockref("ELEVATOR-CAR", (1000, 1000), dxfattribs={"layer": "ELEVATOR"})
    msp.add_blockref("ESCALATOR-UNIT", (2000, 1000), dxfattribs={"layer": "ESCALATOR"})
    result = run_analyze(doc, tmp_path)
    types = {e.element_type for e in result.special_equipment_elements}
    assert SpecialEquipmentElementType.ELEVATOR in types
    assert SpecialEquipmentElementType.ESCALATOR in types


def test_loading_handling_detection(tmp_path):
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    doc.layers.add("CRANE", color=1)
    _ensure_block(doc, "BRIDGE-CRANE")
    _ensure_block(doc, "DOCK-LEVELER-BLOCK")
    msp.add_blockref("BRIDGE-CRANE", (1000, 1000), dxfattribs={"layer": "CRANE"})
    msp.add_blockref("DOCK-LEVELER-BLOCK", (2000, 1000), dxfattribs={"layer": "DOCK"})
    result = run_analyze(doc, tmp_path)
    types = {e.element_type for e in result.special_equipment_elements}
    assert SpecialEquipmentElementType.CRANE in types
    assert SpecialEquipmentElementType.DOCK_LEVELER in types


def test_kitchen_and_medical_detection(tmp_path):
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    doc.layers.add("KITCHEN-EQUIP", color=2)
    doc.layers.add("MEDICAL-EQUIP", color=5)
    _ensure_block(doc, "KITCHEN-HOOD-01")
    _ensure_block(doc, "WALK-IN-FREEZER")
    _ensure_block(doc, "MRI-SIEMENS")
    _ensure_block(doc, "CT-64")
    msp.add_blockref("KITCHEN-HOOD-01", (1000, 1000), dxfattribs={"layer": "KITCHEN-EQUIP"})
    msp.add_blockref("WALK-IN-FREEZER", (2000, 1000), dxfattribs={"layer": "KITCHEN-EQUIP"})
    msp.add_blockref("MRI-SIEMENS", (3000, 1000), dxfattribs={"layer": "MEDICAL-EQUIP"})
    msp.add_blockref("CT-64", (4000, 1000), dxfattribs={"layer": "MEDICAL-EQUIP"})
    result = run_analyze(doc, tmp_path)
    types = {e.element_type for e in result.special_equipment_elements}
    assert SpecialEquipmentElementType.HOOD_EXHAUST in types
    assert SpecialEquipmentElementType.WALKIN_FREEZER in types
    assert SpecialEquipmentElementType.MRI in types
    assert SpecialEquipmentElementType.CT in types


def test_stage_and_pool_detection(tmp_path):
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    doc.layers.add("STAGE", color=6)
    doc.layers.add("POOL-EQUIP", color=4)
    _ensure_block(doc, "STAGE-LIFT")
    _ensure_block(doc, "POOL-FILTER")
    msp.add_blockref("STAGE-LIFT", (1000, 1000), dxfattribs={"layer": "STAGE"})
    msp.add_blockref("POOL-FILTER", (2000, 1000), dxfattribs={"layer": "POOL-EQUIP"})
    result = run_analyze(doc, tmp_path)
    types = {e.element_type for e in result.special_equipment_elements}
    assert SpecialEquipmentElementType.STAGE_LIFT in types
    assert SpecialEquipmentElementType.FILTRATION_UNIT in types


def test_special_equipment_metadata_and_counts(tmp_path):
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    doc.layers.add("ELEVATOR", color=3)
    doc.layers.add("KITCHEN-EQUIP", color=2)
    _ensure_block(doc, "ELEVATOR-CAR")
    _ensure_block(doc, "KITCHEN-HOOD-01")
    msp.add_blockref("ELEVATOR-CAR", (1000, 1000), dxfattribs={"layer": "ELEVATOR"})
    msp.add_blockref("KITCHEN-HOOD-01", (2000, 1000), dxfattribs={"layer": "KITCHEN-EQUIP"})
    result = run_analyze(doc, tmp_path)
    assert "num_special_equipment_elements" in result.metadata
    assert result.metadata["num_special_equipment_elements"] == 2
    assert len(result.special_equipment_elements) == 2
