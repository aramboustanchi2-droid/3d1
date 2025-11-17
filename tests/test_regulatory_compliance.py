"""
Regulatory & Compliance detection tests
تست‌های ضوابط و مقررات
"""
import ezdxf
from cad3d.architectural_analyzer import ArchitecturalAnalyzer, RegulatoryComplianceElementType


def _ensure_layer(doc, name: str):
    if name not in doc.layers:
        doc.layers.add(name)


def test_zoning_and_setbacks(tmp_path):
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    _ensure_layer(doc, "SETBACK")
    _ensure_layer(doc, "LEGAL")

    # Setback polyline
    msp.add_lwpolyline([(0,0), (1000,0), (1000,1000), (0,1000), (0,0)], dxfattribs={"layer": "SETBACK"})
    # Property line
    msp.add_line((2000,2000), (3000,2000), dxfattribs={"layer": "LEGAL"})

    p = tmp_path / "reg_zoning.dxf"
    doc.saveas(p)

    res = ArchitecturalAnalyzer(str(p)).analyze()
    assert any(e.element_type == RegulatoryComplianceElementType.SETBACK_LINE for e in res.regulatory_elements)
    assert any(e.element_type == RegulatoryComplianceElementType.PROPERTY_LINE for e in res.regulatory_elements)


def test_code_compliance_text_and_blocks(tmp_path):
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    _ensure_layer(doc, "EGRESS")
    _ensure_layer(doc, "FIRE-RATING")

    # Egress text
    t1 = msp.add_text("Egress Path to Exit", dxfattribs={"layer": "EGRESS"})
    t1.dxf.insert = (1000, 1000)
    # Fire rating text
    t2 = msp.add_text("2-hour fire wall", dxfattribs={"layer": "FIRE-RATING"})
    t2.dxf.insert = (2000, 2000)

    # Fire door block
    blk = doc.blocks.new(name="FIRE DOOR 1H")
    msp.add_blockref("FIRE DOOR 1H", (3000,3000), dxfattribs={"layer": "FIRE-RATING"})

    p = tmp_path / "reg_code.dxf"
    doc.saveas(p)

    res = ArchitecturalAnalyzer(str(p)).analyze()
    types = [e.element_type for e in res.regulatory_elements]
    assert RegulatoryComplianceElementType.EGRESS_PATH in types
    assert RegulatoryComplianceElementType.FIRE_RATING_WALL in types
    assert RegulatoryComplianceElementType.FIRE_RATING_DOOR in types


def test_accessibility_and_parking(tmp_path):
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    _ensure_layer(doc, "ADA")
    _ensure_layer(doc, "PARKING-REQ")

    # ADA restroom block
    blk = doc.blocks.new(name="ADA RESTROOM")
    msp.add_blockref("ADA RESTROOM", (1000,1000), dxfattribs={"layer": "ADA"})

    # EV Parking block
    blk2 = doc.blocks.new(name="EV PARKING STALL")
    msp.add_blockref("EV PARKING STALL", (2000,1000), dxfattribs={"layer": "PARKING-REQ"})

    p = tmp_path / "reg_ada_parking.dxf"
    doc.saveas(p)

    res = ArchitecturalAnalyzer(str(p)).analyze()
    types = [e.element_type for e in res.regulatory_elements]
    assert RegulatoryComplianceElementType.ADA_RESTROOM in types
    assert RegulatoryComplianceElementType.EV_PARKING in types


def test_environmental_constraints(tmp_path):
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    _ensure_layer(doc, "ENV")
    _ensure_layer(doc, "HEIGHT-LIMIT")

    # Flood zone polygon line
    msp.add_lwpolyline([(0,0), (0,500), (500,500), (500,0), (0,0)], dxfattribs={"layer": "ENV"})

    # Height limit text
    t = msp.add_text("Height limit 24m", dxfattribs={"layer": "HEIGHT-LIMIT"})
    t.dxf.insert = (1200, 1200)

    p = tmp_path / "reg_env.dxf"
    doc.saveas(p)

    res = ArchitecturalAnalyzer(str(p)).analyze()
    types = [e.element_type for e in res.regulatory_elements]
    assert RegulatoryComplianceElementType.FLOOD_ZONE in types or RegulatoryComplianceElementType.NO_BUILD_ZONE in types
    assert RegulatoryComplianceElementType.HEIGHT_LIMIT in types or any(e.geometry_type=="text" for e in res.regulatory_elements)


def test_permit_notes(tmp_path):
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    _ensure_layer(doc, "PERMIT")

    # Permit note text
    t = msp.add_text("Permit: BLD-2025-001 Inspection Req.", dxfattribs={"layer": "PERMIT"})
    t.dxf.insert = (100, 100)

    p = tmp_path / "reg_permit.dxf"
    doc.saveas(p)

    res = ArchitecturalAnalyzer(str(p)).analyze()
    assert any(e.element_type == RegulatoryComplianceElementType.PERMIT_NOTE for e in res.regulatory_elements)


def test_regulatory_metadata(tmp_path):
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    _ensure_layer(doc, "SETBACK")
    msp.add_lwpolyline([(0,0), (100,0), (100,100), (0,100), (0,0)], dxfattribs={"layer": "SETBACK"})

    p = tmp_path / "reg_meta.dxf"
    doc.saveas(p)

    res = ArchitecturalAnalyzer(str(p)).analyze()
    assert "num_regulatory_elements" in res.metadata
    assert res.metadata["num_regulatory_elements"] >= 1
