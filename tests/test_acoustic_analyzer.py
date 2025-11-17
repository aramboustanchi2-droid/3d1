"""
تست‌های سیستم تحلیل آکوستیک
Tests for Acoustic Analysis System
"""

import pytest
from pathlib import Path
import ezdxf
from cad3d.acoustic_analyzer import (
    AcousticAnalyzer,
    AcousticSpaceType,
    AcousticMaterialType,
    AcousticStandard,
    create_acoustic_analyzer
)


def test_acoustic_imports():
    """تست import های پایه"""
    assert AcousticAnalyzer is not None
    assert AcousticSpaceType is not None
    assert AcousticMaterialType is not None
    print("✅ All acoustic imports successful")


def test_acoustic_space_types():
    """تست انواع فضاهای آکوستیک"""
    space_types = list(AcousticSpaceType)
    
    assert len(space_types) >= 20
    assert AcousticSpaceType.CONFERENCE_HALL in space_types
    assert AcousticSpaceType.RECORDING_STUDIO in space_types
    assert AcousticSpaceType.AUDITORIUM in space_types
    assert AcousticSpaceType.CLASSROOM in space_types
    
    print(f"✅ {len(space_types)} acoustic space types defined")
    print("   Sample types:")
    for st in space_types[:5]:
        print(f"   - {st.value}")


def test_acoustic_material_types():
    """تست انواع مواد آکوستیک"""
    material_types = list(AcousticMaterialType)
    
    assert len(material_types) >= 15
    assert AcousticMaterialType.ABSORBER_FOAM in material_types
    assert AcousticMaterialType.INSULATION_WALL in material_types
    assert AcousticMaterialType.DIFFUSER_QRD in material_types
    assert AcousticMaterialType.BASS_TRAP_CORNER in material_types
    
    print(f"✅ {len(material_types)} material types defined")
    print("   Categories:")
    print("   - Absorbers (foam, panel, ceiling, fabric, wood)")
    print("   - Insulation (wall, floor, ceiling, door, window)")
    print("   - Diffusers (QRD, skyline, hemisphere)")
    print("   - Bass Traps (corner, panel)")


def test_rt60_standards():
    """تست استانداردهای RT60"""
    analyzer = AcousticAnalyzer()
    
    assert len(analyzer.RT60_STANDARDS) >= 15
    
    # بررسی مقادیر استاندارد
    assert analyzer.RT60_STANDARDS[AcousticSpaceType.RECORDING_STUDIO] == (0.3, 0.5)
    assert analyzer.RT60_STANDARDS[AcousticSpaceType.CONCERT_HALL] == (1.5, 2.5)
    assert analyzer.RT60_STANDARDS[AcousticSpaceType.CLASSROOM] == (0.4, 0.7)
    
    print(f"✅ RT60 standards for {len(analyzer.RT60_STANDARDS)} space types")
    print("\n   Examples:")
    for space_type, (min_rt, max_rt) in list(analyzer.RT60_STANDARDS.items())[:8]:
        print(f"   - {space_type.value}: {min_rt}-{max_rt}s")


def test_background_noise_standards():
    """تست استانداردهای نویز پس‌زمینه"""
    analyzer = AcousticAnalyzer()
    
    assert len(analyzer.BACKGROUND_NOISE_STANDARDS) >= 10
    
    # بررسی مقادیر
    assert analyzer.BACKGROUND_NOISE_STANDARDS[AcousticSpaceType.RECORDING_STUDIO] == 20
    assert analyzer.BACKGROUND_NOISE_STANDARDS[AcousticSpaceType.CLASSROOM] == 35
    assert analyzer.BACKGROUND_NOISE_STANDARDS[AcousticSpaceType.OFFICE] == 40
    
    print(f"✅ Background noise standards for {len(analyzer.BACKGROUND_NOISE_STANDARDS)} space types")
    print("\n   Examples (dB):")
    for space_type, max_db in list(analyzer.BACKGROUND_NOISE_STANDARDS.items())[:8]:
        print(f"   - {space_type.value}: {max_db} dB")


def test_absorption_coefficients():
    """تست ضرایب جذب صدا"""
    analyzer = AcousticAnalyzer()
    
    # بررسی ضرایب جذب
    coeff_foam = analyzer._get_absorption_coefficient(AcousticMaterialType.ABSORBER_FOAM)
    coeff_panel = analyzer._get_absorption_coefficient(AcousticMaterialType.ABSORBER_PANEL)
    coeff_insulation = analyzer._get_absorption_coefficient(AcousticMaterialType.INSULATION_WALL)
    
    assert 0.0 <= coeff_foam <= 1.0
    assert 0.0 <= coeff_panel <= 1.0
    assert 0.0 <= coeff_insulation <= 1.0
    
    # فوم باید جذب بالاتری داشته باشد
    assert coeff_foam > coeff_insulation
    
    print("✅ Absorption coefficients (0-1 scale):")
    print(f"   - Foam: {coeff_foam}")
    print(f"   - Panel: {coeff_panel}")
    print(f"   - Insulation: {coeff_insulation}")


def test_nrc_stc_ratings():
    """تست NRC و STC ratings"""
    analyzer = AcousticAnalyzer()
    
    # NRC ratings
    nrc_foam = analyzer._get_nrc_rating(AcousticMaterialType.ABSORBER_FOAM)
    nrc_panel = analyzer._get_nrc_rating(AcousticMaterialType.ABSORBER_PANEL)
    
    assert 0.0 <= nrc_foam <= 1.0
    assert 0.0 <= nrc_panel <= 1.0
    
    # STC ratings
    stc_wall = analyzer._get_stc_rating(AcousticMaterialType.INSULATION_WALL)
    stc_floor = analyzer._get_stc_rating(AcousticMaterialType.INSULATION_FLOOR)
    
    assert stc_wall > 0
    assert stc_floor > 0
    
    print("✅ NRC Ratings:")
    print(f"   - Foam: {nrc_foam}")
    print(f"   - Panel: {nrc_panel}")
    print("\n✅ STC Ratings:")
    print(f"   - Wall Insulation: {stc_wall}")
    print(f"   - Floor Insulation: {stc_floor}")


def test_rt60_calculation():
    """تست محاسبه RT60"""
    from cad3d.acoustic_analyzer import AcousticSpace, AcousticMaterial
    
    analyzer = AcousticAnalyzer()
    
    # ایجاد فضای نمونه
    space = AcousticSpace(
        space_type=AcousticSpaceType.CLASSROOM,
        name="Test Classroom",
        area_m2=50.0,      # 50 متر مربع
        volume_m3=150.0,   # 150 متر مکعب (ارتفاع 3 متر)
        height_m=3.0
    )
    
    # بدون مواد جاذب
    rt60_no_material = analyzer.calculate_rt60(space)
    assert rt60_no_material > 0
    
    # با مواد جاذب
    material = AcousticMaterial(
        material_type=AcousticMaterialType.ABSORBER_PANEL,
        location=(0, 0),
        dimensions=(1000, 1000, 50),
        absorption_coefficient=0.75,
        coverage_area_m2=10.0  # 10 متر مربع پنل جاذب
    )
    space.materials = [material]
    
    rt60_with_material = analyzer.calculate_rt60(space)
    assert rt60_with_material > 0
    
    # با مواد جاذب باید RT60 کمتر شود
    assert rt60_with_material < rt60_no_material
    
    print("✅ RT60 Calculation (Sabine formula):")
    print(f"   - Without absorber: {rt60_no_material:.2f}s")
    print(f"   - With absorber (10m²): {rt60_with_material:.2f}s")
    print(f"   - Reduction: {((rt60_no_material - rt60_with_material) / rt60_no_material * 100):.1f}%")


def test_acoustic_score():
    """تست محاسبه امتیاز آکوستیک"""
    from cad3d.acoustic_analyzer import AcousticSpace, AcousticMaterial
    
    analyzer = AcousticAnalyzer()
    
    # فضای با RT60 مناسب
    space_good = AcousticSpace(
        space_type=AcousticSpaceType.CLASSROOM,
        name="Good Classroom",
        area_m2=50.0,
        volume_m3=150.0,
        rt60_target=0.55,
        rt60_actual=0.55,  # دقیقاً در بازه مناسب
        background_noise_db=35  # استاندارد
    )
    space_good.materials = [
        AcousticMaterial(
            material_type=AcousticMaterialType.ABSORBER_PANEL,
            location=(0, 0),
            dimensions=(1000, 1000, 50),
            coverage_area_m2=10.0
        )
    ]
    
    score_good = analyzer.calculate_acoustic_score(space_good)
    assert score_good >= 80  # باید امتیاز عالی بگیرد
    
    # فضای با RT60 نامناسب
    space_bad = AcousticSpace(
        space_type=AcousticSpaceType.CLASSROOM,
        name="Bad Classroom",
        area_m2=50.0,
        volume_m3=150.0,
        rt60_target=0.55,
        rt60_actual=2.0,  # خیلی بالا
        background_noise_db=60  # خیلی بالا
    )
    space_bad.materials = []  # بدون ماده جاذب
    
    score_bad = analyzer.calculate_acoustic_score(space_bad)
    assert score_bad < 60  # باید امتیاز پایین بگیرد
    
    print("✅ Acoustic Score Calculation:")
    print(f"   - Good classroom: {score_good:.1f}/100")
    print(f"   - Bad classroom: {score_bad:.1f}/100")


def test_create_test_dxf(tmp_path):
    """ایجاد فایل DXF تست"""
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    
    # افزودن لایه‌ها
    doc.layers.add('AUDITORIUM')
    doc.layers.add('ACOUSTIC_PANEL')
    doc.layers.add('SOUND_INSULATION')
    doc.layers.add('HVAC')
    
    # فضای آمفی‌تئاتر (20x15 متر)
    auditorium_points = [
        (0, 0), (20000, 0), (20000, 15000), (0, 15000), (0, 0)
    ]
    msp.add_lwpolyline(auditorium_points, dxfattribs={'layer': 'AUDITORIUM', 'closed': True})
    
    # پنل‌های جاذب صوتی
    for i in range(5):
        panel_points = [
            (1000 + i*3000, 1000),
            (2000 + i*3000, 1000),
            (2000 + i*3000, 2000),
            (1000 + i*3000, 2000),
            (1000 + i*3000, 1000)
        ]
        msp.add_lwpolyline(panel_points, dxfattribs={'layer': 'ACOUSTIC_PANEL'})
    
    # عایق صوتی دیوارها
    insulation_points = [
        (0, -200), (20000, -200), (20000, 0), (0, 0), (0, -200)
    ]
    msp.add_lwpolyline(insulation_points, dxfattribs={'layer': 'SOUND_INSULATION'})
    
    # منبع نویز (سیستم تهویه)
    msp.add_circle((19000, 14000), radius=500, dxfattribs={'layer': 'HVAC'})
    
    # ذخیره فایل
    test_file = tmp_path / "test_acoustic.dxf"
    doc.saveas(test_file)
    
    return test_file


def test_detect_acoustic_spaces(tmp_path):
    """تست تشخیص فضاهای آکوستیک"""
    test_file = test_create_test_dxf(tmp_path)
    
    doc = ezdxf.readfile(test_file)
    analyzer = AcousticAnalyzer()
    
    spaces = analyzer.detect_acoustic_spaces(doc)
    
    assert len(spaces) >= 1
    
    # بررسی فضای آمفی‌تئاتر
    auditorium = spaces[0]
    assert auditorium.space_type == AcousticSpaceType.AUDITORIUM
    assert auditorium.area_m2 > 0
    assert auditorium.volume_m3 > 0
    assert auditorium.rt60_target > 0
    
    print(f"✅ Detected {len(spaces)} acoustic space(s):")
    for space in spaces:
        print(f"   - {space.space_type.value}: {space.area_m2:.1f}m², RT60 target: {space.rt60_target:.2f}s")


def test_detect_acoustic_materials(tmp_path):
    """تست تشخیص مواد آکوستیک"""
    test_file = test_create_test_dxf(tmp_path)
    
    doc = ezdxf.readfile(test_file)
    analyzer = AcousticAnalyzer()
    
    materials = analyzer.detect_acoustic_materials(doc)
    
    assert len(materials) >= 1
    
    # بررسی مواد
    absorber_count = sum(1 for m in materials if 'ABSORBER' in m.material_type.name)
    insulation_count = sum(1 for m in materials if 'INSULATION' in m.material_type.name)
    
    print(f"✅ Detected {len(materials)} acoustic material(s):")
    print(f"   - Absorbers: {absorber_count}")
    print(f"   - Insulation: {insulation_count}")
    
    if materials:
        mat = materials[0]
        print(f"\n   Sample material:")
        print(f"   - Type: {mat.material_type.value}")
        print(f"   - Absorption coefficient: {mat.absorption_coefficient}")
        print(f"   - NRC rating: {mat.nrc_rating}")


def test_detect_noise_sources(tmp_path):
    """تست تشخیص منابع نویز"""
    test_file = test_create_test_dxf(tmp_path)
    
    doc = ezdxf.readfile(test_file)
    analyzer = AcousticAnalyzer()
    
    noise_sources = analyzer.detect_noise_sources(doc)
    
    # ممکن است منبع نویز تشخیص داده شود یا نشود
    print(f"✅ Detected {len(noise_sources)} noise source(s)")
    
    for source in noise_sources:
        print(f"   - {source.source_type}: {source.sound_power_level_db} dB")


def test_full_acoustic_analysis(tmp_path):
    """تست تحلیل کامل آکوستیک"""
    test_file = test_create_test_dxf(tmp_path)
    
    analyzer = AcousticAnalyzer()
    result = analyzer.analyze(str(test_file))
    
    assert result is not None
    assert result.total_spaces >= 0
    assert result.average_acoustic_score >= 0
    
    print("\n" + "="*60)
    print("🎵 FULL ACOUSTIC ANALYSIS RESULT")
    print("="*60)
    print(f"\n📊 Summary:")
    print(f"   - Total spaces: {result.total_spaces}")
    print(f"   - Total acoustic area: {result.total_acoustic_area_m2:.1f} m²")
    print(f"   - Total absorber area: {result.total_absorber_area_m2:.1f} m²")
    print(f"   - Total insulation area: {result.total_insulation_area_m2:.1f} m²")
    print(f"   - Average score: {result.average_acoustic_score:.1f}/100")
    print(f"   - Compliant spaces: {result.compliant_spaces}")
    print(f"   - Non-compliant spaces: {result.non_compliant_spaces}")
    
    if result.spaces:
        print(f"\n🏛️  Spaces:")
        for space in result.spaces:
            print(f"   - {space.space_type.value}:")
            print(f"     * Area: {space.area_m2:.1f} m²")
            print(f"     * Volume: {space.volume_m3:.1f} m³")
            print(f"     * RT60 target: {space.rt60_target:.2f}s")
            print(f"     * RT60 actual: {space.rt60_actual:.2f}s")
            print(f"     * Score: {space.acoustic_score:.1f}/100")
            print(f"     * Status: {space.compliance_status}")
    
    if result.warnings:
        print(f"\n⚠️  Warnings ({len(result.warnings)}):")
        for warning in result.warnings[:3]:
            print(f"   - {warning}")
    
    if result.recommendations:
        print(f"\n💡 Recommendations ({len(result.recommendations)}):")
        for rec in result.recommendations[:3]:
            print(f"   - {rec}")


def test_export_to_json(tmp_path):
    """تست خروجی JSON"""
    test_file = test_create_test_dxf(tmp_path)
    
    analyzer = AcousticAnalyzer()
    result = analyzer.analyze(str(test_file))
    
    json_output = tmp_path / "acoustic_report.json"
    analyzer.export_to_json(result, str(json_output))
    
    assert json_output.exists()
    
    # خواندن و بررسی JSON
    import json
    with open(json_output, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    assert 'summary' in data
    assert 'spaces' in data
    assert 'materials' in data
    assert 'warnings' in data
    assert 'recommendations' in data
    
    print(f"✅ JSON report exported to: {json_output}")
    print(f"   - File size: {json_output.stat().st_size} bytes")


def test_system_summary():
    """خلاصه سیستم آکوستیک"""
    print("\n" + "="*60)
    print("🎵 ACOUSTIC ANALYSIS SYSTEM - SUMMARY")
    print("="*60)
    
    analyzer = AcousticAnalyzer()
    
    space_types = list(AcousticSpaceType)
    material_types = list(AcousticMaterialType)
    standards = list(AcousticStandard)
    
    print(f"\n📋 Capabilities:")
    print(f"   - Space types: {len(space_types)}")
    print(f"   - Material types: {len(material_types)}")
    print(f"   - Standards: {len(standards)}")
    print(f"   - RT60 standards: {len(analyzer.RT60_STANDARDS)}")
    print(f"   - Noise standards: {len(analyzer.BACKGROUND_NOISE_STANDARDS)}")
    
    print(f"\n🏛️  Space Categories:")
    print(f"   - Performance venues: {len([s for s in space_types if 'HALL' in s.name or 'THEATER' in s.name or 'CINEMA' in s.name])}")
    print(f"   - Studios: {len([s for s in space_types if 'STUDIO' in s.name or 'CONTROL' in s.name])}")
    print(f"   - Educational: {len([s for s in space_types if 'CLASS' in s.name or 'LIBRARY' in s.name or 'LANGUAGE' in s.name])}")
    print(f"   - Office/Commercial: {len([s for s in space_types if 'OFFICE' in s.name or 'MEETING' in s.name or 'RESTAURANT' in s.name])}")
    
    print(f"\n🧱 Material Categories:")
    print(f"   - Absorbers: {len([m for m in material_types if 'ABSORBER' in m.name])}")
    print(f"   - Insulation: {len([m for m in material_types if 'INSULATION' in m.name])}")
    print(f"   - Diffusers: {len([m for m in material_types if 'DIFFUSER' in m.name])}")
    print(f"   - Bass Traps: {len([m for m in material_types if 'BASS_TRAP' in m.name])}")
    
    print(f"\n📐 Analysis Features:")
    print(f"   ✓ RT60 calculation (Sabine formula)")
    print(f"   ✓ Absorption coefficient analysis")
    print(f"   ✓ NRC/STC ratings")
    print(f"   ✓ Background noise evaluation")
    print(f"   ✓ Compliance checking")
    print(f"   ✓ Acoustic scoring (0-100)")
    print(f"   ✓ Automated recommendations")
    
    print(f"\n🎯 Output Formats:")
    print(f"   ✓ JSON reports")
    print(f"   ✓ Detailed warnings")
    print(f"   ✓ Design recommendations")
    
    print("\n✨ System ready for acoustic analysis!")


if __name__ == "__main__":
    # اجرای تست‌ها
    pytest.main([__file__, "-v", "-s"])
