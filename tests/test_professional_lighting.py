"""
Tests for Professional Lighting Detection
تست تشخیص نقشه‌های روشنایی حرفه‌ای
"""

import pytest
import ezdxf
from pathlib import Path


def test_lighting_detector_import():
    """تست: import موفق"""
    from cad3d.professional_lighting_detector import (
        ProfessionalLightingDetector,
        LightingType,
        LightingZone,
        LightingFixtureInfo
    )
    assert ProfessionalLightingDetector is not None


def test_lighting_fixture_detection(tmp_path):
    """تست: تشخیص چراغ‌های روشنایی"""
    from cad3d.professional_lighting_detector import ProfessionalLightingDetector
    
    # ساخت DXF تست
    test_dxf = tmp_path / "lighting_plan.dxf"
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    
    # اضافه کردن لایه روشنایی
    doc.layers.add("LIGHTING")
    
    # اضافه کردن چراغ‌های توکار (دایره)
    msp.add_circle((1000, 1000), 150, dxfattribs={"layer": "LIGHTING"})
    msp.add_circle((3000, 1000), 150, dxfattribs={"layer": "LIGHTING"})
    msp.add_circle((5000, 1000), 150, dxfattribs={"layer": "LIGHTING"})
    
    # اضافه کردن چراغ آویز (بلاک)
    if "PENDANT" not in doc.blocks:
        block = doc.blocks.new(name="PENDANT")
        block.add_circle((0, 0), 200)
    msp.add_blockref("PENDANT", (2000, 3000), dxfattribs={"layer": "LIGHTING"})
    
    doc.saveas(test_dxf)
    
    # تحلیل
    detector = ProfessionalLightingDetector(str(test_dxf))
    fixtures = detector.detect_lighting_fixtures()
    
    assert len(fixtures) >= 3, f"Expected at least 3 fixtures, got {len(fixtures)}"


def test_lighting_circuits_detection(tmp_path):
    """تست: تشخیص مدارهای روشنایی"""
    from cad3d.professional_lighting_detector import (
        ProfessionalLightingDetector,
        LightingFixtureInfo,
        LightingType
    )
    
    # ساخت DXF
    test_dxf = tmp_path / "circuits.dxf"
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    doc.layers.add("LIGHTING-1")
    
    # چند چراغ نزدیک هم
    for i in range(5):
        msp.add_circle((i * 1000, 1000), 150, dxfattribs={"layer": "LIGHTING-1"})
    
    doc.saveas(test_dxf)
    
    # تحلیل
    detector = ProfessionalLightingDetector(str(test_dxf))
    fixtures = detector.detect_lighting_fixtures()
    circuits = detector.detect_lighting_circuits(fixtures)
    
    assert len(circuits) >= 1
    assert circuits[0].total_power > 0


def test_lighting_zones_detection(tmp_path):
    """تست: تشخیص نواحی نورپردازی"""
    from cad3d.professional_lighting_detector import ProfessionalLightingDetector
    
    # ساخت DXF
    test_dxf = tmp_path / "zones.dxf"
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    
    # ناحیه داخلی
    doc.layers.add("LIGHTING-INTERIOR")
    for i in range(3):
        msp.add_circle((i * 2000, 1000), 150, dxfattribs={"layer": "LIGHTING-INTERIOR"})
    
    # ناحیه نما
    doc.layers.add("LIGHTING-FACADE")
    for i in range(2):
        msp.add_circle((i * 2000, 5000), 150, dxfattribs={"layer": "LIGHTING-FACADE"})
    
    doc.saveas(test_dxf)
    
    # تحلیل
    detector = ProfessionalLightingDetector(str(test_dxf))
    fixtures = detector.detect_lighting_fixtures()
    zones = detector.detect_lighting_zones(fixtures)
    
    assert len(zones) >= 1


def test_complete_lighting_analysis(tmp_path):
    """تست: تحلیل کامل نقشه روشنایی"""
    from cad3d.professional_lighting_detector import ProfessionalLightingDetector
    
    # ساخت DXF جامع
    test_dxf = tmp_path / "complete_lighting.dxf"
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    
    # چند لایه روشنایی
    doc.layers.add("LIGHTING-GENERAL")
    doc.layers.add("LIGHTING-FACADE")
    doc.layers.add("LIGHTING-EMERGENCY")
    
    # چراغ‌های مختلف
    for i in range(10):
        layer = "LIGHTING-GENERAL" if i < 7 else "LIGHTING-EMERGENCY"
        msp.add_circle((i * 1000, 1000), 150, dxfattribs={"layer": layer})
    
    # چراغ‌های نما
    for i in range(5):
        msp.add_circle((i * 1500, 6000), 200, dxfattribs={"layer": "LIGHTING-FACADE"})
    
    doc.saveas(test_dxf)
    
    # تحلیل کامل
    detector = ProfessionalLightingDetector(str(test_dxf))
    result = detector.analyze()
    
    # بررسی نتایج
    assert result.total_fixtures > 0
    assert result.total_power_kw > 0
    assert len(result.fixture_type_counts) > 0
    assert len(result.circuits) > 0
    assert result.illuminated_area_sqm > 0
    assert result.metadata is not None
    
    print(f"\n✅ Lighting Analysis Results:")
    print(f"   Total Fixtures: {result.total_fixtures}")
    print(f"   Total Power: {result.total_power_kw:.2f} kW")
    print(f"   Circuits: {len(result.circuits)}")
    print(f"   Zones: {len(result.zones)}")
    print(f"   Illuminated Area: {result.illuminated_area_sqm:.2f} m²")
    print(f"   Power Density: {result.average_power_density:.2f} W/m²")


def test_lighting_system_summary():
    """تست: خلاصه سیستم روشنایی"""
    from cad3d.professional_lighting_detector import (
        LightingType,
        LightingZone,
        ProfessionalLightingDetector
    )
    
    print("\n" + "="*60)
    print("Professional Lighting Detection System")
    print("="*60)
    print(f"✅ Lighting Types: {len(LightingType)}")
    print(f"✅ Lighting Zones: {len(LightingZone)}")
    print(f"✅ Detection Features:")
    print(f"   - Interior Lighting (Recessed, Pendant, Track, ...)")
    print(f"   - Facade Lighting (Uplighting, Downlighting, ...)")
    print(f"   - Landscape Lighting (Trees, Gardens, Pools, ...)")
    print(f"   - Emergency & Exit Lighting")
    print(f"   - Smart Controls & Sensors")
    print(f"✅ Analysis:")
    print(f"   - Fixture Detection")
    print(f"   - Circuit Grouping")
    print(f"   - Zone Definition")
    print(f"   - Power Calculation")
    print(f"   - Lux Level Standards")
    print("="*60)


if __name__ == "__main__":
    # Run summary
    test_lighting_system_summary()
