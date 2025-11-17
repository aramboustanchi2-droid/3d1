"""
Sustainability & Energy detection tests
تست‌های پایداری و انرژی
"""
import ezdxf
from cad3d.architectural_analyzer import ArchitecturalAnalyzer, SustainabilityElementType


def _ensure_layer(doc, name: str):
    if name not in doc.layers:
        doc.layers.add(name)


def test_renewables_detection(tmp_path):
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    _ensure_layer(doc, "SOLAR")

    # Solar PV, Inverter, Thermal collector, Battery
    doc.blocks.new(name="SOLAR PV PANEL")
    msp.add_blockref("SOLAR PV PANEL", (1000,1000), dxfattribs={"layer": "SOLAR"})
    doc.blocks.new(name="PV INVERTER")
    msp.add_blockref("PV INVERTER", (2000,1000), dxfattribs={"layer": "SOLAR"})
    doc.blocks.new(name="SOLAR THERMAL COLLECTOR")
    msp.add_blockref("SOLAR THERMAL COLLECTOR", (3000,1000), dxfattribs={"layer": "SOLAR"})
    doc.blocks.new(name="BATTERY STORAGE")
    msp.add_blockref("BATTERY STORAGE", (4000,1000), dxfattribs={"layer": "SOLAR"})

    p = tmp_path / "sustain_renew.dxf"
    doc.saveas(p)
    res = ArchitecturalAnalyzer(str(p)).analyze()

    types = [e.element_type for e in res.sustainability_elements]
    assert SustainabilityElementType.SOLAR_PV_PANEL in types
    assert SustainabilityElementType.SOLAR_INVERTER in types
    assert SustainabilityElementType.SOLAR_THERMAL_COLLECTOR in types
    assert SustainabilityElementType.BATTERY_STORAGE in types


def test_meters_and_bms(tmp_path):
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    _ensure_layer(doc, "METER")

    doc.blocks.new(name="ENERGY METER")
    msp.add_blockref("ENERGY METER", (1000,1000), dxfattribs={"layer": "METER"})
    doc.blocks.new(name="SUB METER")
    msp.add_blockref("SUB METER", (2000,1000), dxfattribs={"layer": "METER"})
    doc.blocks.new(name="BMS PANEL")
    msp.add_blockref("BMS PANEL", (3000,1000), dxfattribs={"layer": "METER"})

    p = tmp_path / "sustain_meter.dxf"
    doc.saveas(p)
    res = ArchitecturalAnalyzer(str(p)).analyze()

    types = [e.element_type for e in res.sustainability_elements]
    assert SustainabilityElementType.ENERGY_METER in types
    assert SustainabilityElementType.SUB_METER in types
    assert SustainabilityElementType.BMS_PANEL in types


def test_envelope_insulation_notes(tmp_path):
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    _ensure_layer(doc, "INSULATION")

    # Insulation zone polyline
    msp.add_lwpolyline([(0,0),(2000,0),(2000,1000),(0,1000),(0,0)], dxfattribs={"layer": "INSULATION"})
    # U-value and R-value texts
    t1 = msp.add_text("U-Value = 1.2", dxfattribs={"layer": "INSULATION"}); t1.dxf.insert=(2500,500)
    t2 = msp.add_text("R-Value = 3.5", dxfattribs={"layer": "INSULATION"}); t2.dxf.insert=(2500,800)

    p = tmp_path / "sustain_envelope.dxf"
    doc.saveas(p)
    res = ArchitecturalAnalyzer(str(p)).analyze()

    types = [e.element_type for e in res.sustainability_elements]
    assert SustainabilityElementType.INSULATION_ZONE in types
    assert SustainabilityElementType.U_VALUE_NOTE in types
    assert SustainabilityElementType.R_VALUE_NOTE in types


def test_daylight_and_shading(tmp_path):
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    _ensure_layer(doc, "SHADING")

    doc.blocks.new(name="SKYLIGHT 1")
    msp.add_blockref("SKYLIGHT 1", (1000,1000), dxfattribs={"layer": "SHADING"})
    doc.blocks.new(name="LIGHT SHELF 1")
    msp.add_blockref("LIGHT SHELF 1", (2000,1000), dxfattribs={"layer": "SHADING"})
    doc.blocks.new(name="LOUVER 1")
    msp.add_blockref("LOUVER 1", (3000,1000), dxfattribs={"layer": "SHADING"})

    p = tmp_path / "sustain_daylight.dxf"
    doc.saveas(p)
    res = ArchitecturalAnalyzer(str(p)).analyze()

    types = [e.element_type for e in res.sustainability_elements]
    assert SustainabilityElementType.SKYLIGHT in types
    assert SustainabilityElementType.LIGHT_SHELF in types
    assert SustainabilityElementType.LOUVER in types


def test_water_sustainability_and_green_features(tmp_path):
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    _ensure_layer(doc, "RAINWATER")
    _ensure_layer(doc, "GREEN")

    doc.blocks.new(name="RAINWATER FILTER")
    msp.add_blockref("RAINWATER FILTER", (1000,1000), dxfattribs={"layer": "RAINWATER"})
    doc.blocks.new(name="GREYWATER TANK")
    msp.add_blockref("GREYWATER TANK", (2000,1000), dxfattribs={"layer": "RAINWATER"})

    doc.blocks.new(name="GREEN ROOF")
    msp.add_blockref("GREEN ROOF", (3000,1000), dxfattribs={"layer": "GREEN"})
    doc.blocks.new(name="GREEN WALL")
    msp.add_blockref("GREEN WALL", (4000,1000), dxfattribs={"layer": "GREEN"})

    p = tmp_path / "sustain_water_green.dxf"
    doc.saveas(p)
    res = ArchitecturalAnalyzer(str(p)).analyze()

    types = [e.element_type for e in res.sustainability_elements]
    assert SustainabilityElementType.RAINWATER_FILTER in types
    assert SustainabilityElementType.GREYWATER_TANK in types
    assert SustainabilityElementType.GREEN_ROOF in types
    assert SustainabilityElementType.GREEN_WALL in types


def test_certifications_and_zones(tmp_path):
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    _ensure_layer(doc, "LEED")
    _ensure_layer(doc, "THERMAL-ZONE")

    t = msp.add_text("LEED Gold target", dxfattribs={"layer": "LEED"}); t.dxf.insert=(1000,1000)
    msp.add_lwpolyline([(0,0),(500,0),(500,500),(0,500),(0,0)], dxfattribs={"layer": "THERMAL-ZONE"})

    p = tmp_path / "sustain_cert_zone.dxf"
    doc.saveas(p)
    res = ArchitecturalAnalyzer(str(p)).analyze()

    types = [e.element_type for e in res.sustainability_elements]
    assert SustainabilityElementType.LEED_NOTE in types
    assert SustainabilityElementType.ENERGY_ZONE in types or SustainabilityElementType.THERMAL_ZONE in types


def test_sustainability_metadata(tmp_path):
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    _ensure_layer(doc, "SOLAR")
    doc.blocks.new(name="SOLAR PV PANEL")
    msp.add_blockref("SOLAR PV PANEL", (100,100), dxfattribs={"layer": "SOLAR"})

    p = tmp_path / "sustain_meta.dxf"
    doc.saveas(p)
    res = ArchitecturalAnalyzer(str(p)).analyze()

    assert "num_sustainability_elements" in res.metadata
    assert res.metadata["num_sustainability_elements"] >= 1
