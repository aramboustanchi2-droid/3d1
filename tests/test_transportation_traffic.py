"""
Transportation & Traffic detection tests
تست‌های حمل‌ونقل و ترافیک
"""
import ezdxf
from cad3d.architectural_analyzer import ArchitecturalAnalyzer, TransportationTrafficElementType


def _ensure_layer(doc, name: str):
    if name not in doc.layers:
        doc.layers.add(name)


def test_parking_stalls_and_aisles(tmp_path):
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    _ensure_layer(doc, "PARKING")

    # Parking stall ~ 2.5m x 5.0m
    msp.add_lwpolyline([(0,0),(2500,0),(2500,5000),(0,5000),(0,0)], dxfattribs={"layer": "PARKING"})
    # Aisle area
    msp.add_lwpolyline([(3000,0),(8000,0),(8000,6000),(3000,6000),(3000,0)], dxfattribs={"layer": "PARKING"})

    p = tmp_path / "traffic_parking.dxf"
    doc.saveas(p)
    res = ArchitecturalAnalyzer(str(p)).analyze()

    types = [e.element_type for e in res.transportation_elements]
    assert TransportationTrafficElementType.PARKING_STALL in types
    assert TransportationTrafficElementType.PARKING_AISLE in types


def test_vehicle_routes_arrows_and_turning(tmp_path):
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    _ensure_layer(doc, "LANE")
    _ensure_layer(doc, "ARROW")
    _ensure_layer(doc, "TURN")

    # Lane polyline
    msp.add_lwpolyline([(0,0),(10000,0)], dxfattribs={"layer": "LANE"})
    # Traffic arrow block
    doc.blocks.new(name="TRAFFIC ARROW")
    msp.add_blockref("TRAFFIC ARROW", (2000,1000), dxfattribs={"layer": "ARROW"})
    # Turning radius circle
    msp.add_circle((5000,5000), radius=6000, dxfattribs={"layer": "TURN"})

    p = tmp_path / "traffic_routes.dxf"
    doc.saveas(p)
    res = ArchitecturalAnalyzer(str(p)).analyze()

    types = [e.element_type for e in res.transportation_elements]
    assert TransportationTrafficElementType.TRAFFIC_LANE in types
    assert TransportationTrafficElementType.TRAFFIC_FLOW_ARROW in types
    assert TransportationTrafficElementType.TURNING_RADIUS in types


def test_pedestrian_paths(tmp_path):
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    _ensure_layer(doc, "WALKWAY")
    _ensure_layer(doc, "CROSSWALK")

    msp.add_lwpolyline([(0,0),(0,5000)], dxfattribs={"layer": "WALKWAY"})
    msp.add_lwpolyline([(1000,0),(6000,0)], dxfattribs={"layer": "CROSSWALK"})

    p = tmp_path / "traffic_ped.dxf"
    doc.saveas(p)
    res = ArchitecturalAnalyzer(str(p)).analyze()

    types = [e.element_type for e in res.transportation_elements]
    assert TransportationTrafficElementType.WALKWAY in types
    assert TransportationTrafficElementType.CROSSWALK in types


def test_signage_speed_limit_and_stop(tmp_path):
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    _ensure_layer(doc, "SIGN")

    t1 = msp.add_text("SPEED LIMIT 25", dxfattribs={"layer": "SIGN"}); t1.dxf.insert=(1000,1000)
    t2 = msp.add_text("STOP", dxfattribs={"layer": "SIGN"}); t2.dxf.insert=(1500,1200)

    p = tmp_path / "traffic_signs.dxf"
    doc.saveas(p)
    res = ArchitecturalAnalyzer(str(p)).analyze()

    types = [e.element_type for e in res.transportation_elements]
    assert TransportationTrafficElementType.SPEED_LIMIT in types
    assert TransportationTrafficElementType.STOP_SIGN in types


def test_loading_bike_bus(tmp_path):
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    _ensure_layer(doc, "LOADING")
    _ensure_layer(doc, "BIKE")
    _ensure_layer(doc, "BUS")

    # Loading zone polygon
    msp.add_lwpolyline([(0,0),(4000,0),(4000,3000),(0,3000),(0,0)], dxfattribs={"layer": "LOADING"})
    # Bike lane line
    msp.add_lwpolyline([(0,0),(0,8000)], dxfattribs={"layer": "BIKE"})
    # Bus stop block
    doc.blocks.new(name="BUS STOP")
    msp.add_blockref("BUS STOP", (500,500), dxfattribs={"layer": "BUS"})

    p = tmp_path / "traffic_misc.dxf"
    doc.saveas(p)
    res = ArchitecturalAnalyzer(str(p)).analyze()

    types = [e.element_type for e in res.transportation_elements]
    assert TransportationTrafficElementType.LOADING_ZONE in types
    assert TransportationTrafficElementType.BIKE_LANE in types
    assert TransportationTrafficElementType.BUS_STOP in types


def test_transportation_metadata(tmp_path):
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    _ensure_layer(doc, "ARROW")
    doc.blocks.new(name="ARROW")
    msp.add_blockref("ARROW", (100,100), dxfattribs={"layer": "ARROW"})

    p = tmp_path / "traffic_meta.dxf"
    doc.saveas(p)
    res = ArchitecturalAnalyzer(str(p)).analyze()

    assert "num_transportation_elements" in res.metadata
    assert res.metadata["num_transportation_elements"] >= 1
