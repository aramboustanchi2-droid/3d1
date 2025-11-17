"""
Tests for IT & Network Infrastructure detection
فاز ۹: زیرساخت IT و شبکه
"""
import pytest
import ezdxf
import tempfile
from pathlib import Path

from cad3d.architectural_analyzer import (
    ArchitecturalAnalyzer,
    ITNetworkElementType,
)


@pytest.fixture
def temp_dxf():
    """Create a temporary DXF file"""
    with tempfile.TemporaryDirectory() as tmpdir:
        dxf_path = Path(tmpdir) / "test_it.dxf"
        yield dxf_path


def test_rack_and_patch_panel_blocks(temp_dxf):
    """تست تشخیص رک‌ها و پچ پنل‌ها از بلوک‌ها"""
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()

    # Add IT layer
    doc.layers.add("IT-RACK")

    # Rack block
    msp.add_blockref("RACK-42U", insert=(1000, 1000), dxfattribs={"layer": "IT-RACK"})
    # Patch Panel block
    msp.add_blockref("PATCH-PANEL-24", insert=(2000, 1000), dxfattribs={"layer": "IT-RACK"})

    doc.saveas(temp_dxf)

    analyzer = ArchitecturalAnalyzer(str(temp_dxf))
    analysis = analyzer.analyze()

    assert len(analysis.it_network_elements) >= 2
    racks = [e for e in analysis.it_network_elements if e.element_type == ITNetworkElementType.RACK]
    panels = [e for e in analysis.it_network_elements if e.element_type == ITNetworkElementType.PATCH_PANEL]
    assert len(racks) == 1
    assert len(panels) == 1


def test_wifi_ap_and_server_equipment(temp_dxf):
    """تست تشخیص اکسس پوینت و تجهیزات سرور"""
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()

    doc.layers.add("NETWORK")

    # WiFi AP block
    msp.add_blockref("WIFI-AP-01", insert=(3000, 2000), dxfattribs={"layer": "NETWORK"})
    # Server equipment block
    msp.add_blockref("SERVER-BLADE", insert=(4000, 2000), dxfattribs={"layer": "NETWORK"})

    doc.saveas(temp_dxf)

    analyzer = ArchitecturalAnalyzer(str(temp_dxf))
    analysis = analyzer.analyze()

    aps = [e for e in analysis.it_network_elements if e.element_type == ITNetworkElementType.WIFI_AP]
    servers = [e for e in analysis.it_network_elements if e.element_type == ITNetworkElementType.SERVER_EQUIPMENT]
    assert len(aps) == 1
    assert len(servers) == 1


def test_data_ports_via_text(temp_dxf):
    """تست تشخیص پریزهای شبکه از طریق متن (DATA، RJ45، CAT6)"""
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()

    doc.layers.add("DATA-PORTS")

    # Data port labels
    msp.add_text("DATA-01", dxfattribs={"layer": "DATA-PORTS"}).set_placement((500, 500))
    msp.add_text("RJ45 Port", dxfattribs={"layer": "DATA-PORTS"}).set_placement((1000, 500))
    msp.add_text("CAT6", dxfattribs={"layer": "DATA-PORTS"}).set_placement((1500, 500))
    msp.add_text("LAN Socket", dxfattribs={"layer": "DATA-PORTS"}).set_placement((2000, 500))

    doc.saveas(temp_dxf)

    analyzer = ArchitecturalAnalyzer(str(temp_dxf))
    analysis = analyzer.analyze()

    data_ports = [e for e in analysis.it_network_elements if e.element_type == ITNetworkElementType.DATA_PORT]
    assert len(data_ports) >= 4


def test_cable_tray_vs_fiber_cable(temp_dxf):
    """تست تشخیص سینی کابل در مقابل فیبر نوری (از لایه و نوع خط)"""
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()

    doc.layers.add("CABLE-TRAY")
    doc.layers.add("FIBER-BACKBONE")

    # Cable tray as polyline
    msp.add_lwpolyline([(0, 0), (1000, 0), (1000, 1000)], dxfattribs={"layer": "CABLE-TRAY"})

    # Fiber cable as line
    msp.add_line((2000, 0), (2000, 2000), dxfattribs={"layer": "FIBER-BACKBONE"})

    doc.saveas(temp_dxf)

    analyzer = ArchitecturalAnalyzer(str(temp_dxf))
    analysis = analyzer.analyze()

    trays = [e for e in analysis.it_network_elements if e.element_type == ITNetworkElementType.CABLE_TRAY]
    fibers = [e for e in analysis.it_network_elements if e.element_type == ITNetworkElementType.FIBER_CABLE]

    assert len(trays) >= 1
    assert len(fibers) >= 1


def test_server_room_polygon_and_text(temp_dxf):
    """تست تشخیص اتاق سرور از پلیگون لایه و لیبل متنی"""
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()

    doc.layers.add("SERVER-ROOM")

    # Server room as polygon
    msp.add_lwpolyline([(0, 0), (5000, 0), (5000, 5000), (0, 5000), (0, 0)], dxfattribs={"layer": "SERVER-ROOM"})

    # Server room text label
    msp.add_text("SERVER ROOM", dxfattribs={"layer": "SERVER-ROOM"}).set_placement((2500, 2500))

    doc.saveas(temp_dxf)

    analyzer = ArchitecturalAnalyzer(str(temp_dxf))
    analysis = analyzer.analyze()

    server_rooms = [e for e in analysis.it_network_elements if e.element_type == ITNetworkElementType.SERVER_ROOM]
    # Should detect at least polygon and text label
    assert len(server_rooms) >= 2


def test_it_metadata(temp_dxf):
    """تست متادیتا: num_it_network_elements"""
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()

    doc.layers.add("IT")

    # Add a few IT elements
    msp.add_blockref("RACK-1", insert=(1000, 1000), dxfattribs={"layer": "IT"})
    msp.add_text("DATA Port", dxfattribs={"layer": "IT"}).set_placement((2000, 1000))
    msp.add_line((0, 0), (1000, 0), dxfattribs={"layer": "IT"})  # will not match IT keywords, so should not count

    doc.saveas(temp_dxf)

    analyzer = ArchitecturalAnalyzer(str(temp_dxf))
    analysis = analyzer.analyze()

    assert "num_it_network_elements" in analysis.metadata
    # Should have at least rack and data port
    assert analysis.metadata["num_it_network_elements"] >= 2
