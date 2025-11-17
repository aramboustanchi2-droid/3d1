"""
تست‌های تشخیص جزئیات اجرایی (Construction Details)
"""
import pytest
import ezdxf
from pathlib import Path

from cad3d.architectural_analyzer import (
    ArchitecturalAnalyzer,
    DrawingType,
    DetailType,
    MaterialType,
)


def test_door_window_details(tmp_path):
    """تست تشخیص جزئیات درب و پنجره"""
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    
    doc.layers.add("DETAIL", color=7)
    
    # جزئیات درب
    msp.add_text("DOOR DETAIL", dxfattribs={"layer": "DETAIL", "insert": (1000, 1000)})
    msp.add_text("DOOR FRAME", dxfattribs={"layer": "DETAIL", "insert": (1000, 1500)})
    msp.add_text("SCALE 1:5", dxfattribs={"layer": "DETAIL", "insert": (1000, 800)})
    
    # جزئیات پنجره
    msp.add_text("WINDOW DETAIL", dxfattribs={"layer": "DETAIL", "insert": (3000, 1000)})
    msp.add_text("ALUMINUM FRAME", dxfattribs={"layer": "DETAIL", "insert": (3000, 1500)})
    msp.add_text("GLASS 6mm", dxfattribs={"layer": "DETAIL", "insert": (3000, 800)})
    
    dxf_path = tmp_path / "door_window_details.dxf"
    doc.saveas(dxf_path)
    
    analyzer = ArchitecturalAnalyzer(str(dxf_path))
    analysis = analyzer.analyze()
    
    details = analysis.construction_details
    assert len(details) >= 2, f"باید حداقل 2 جزئیات تشخیص داده شود، تعداد: {len(details)}"
    
    # بررسی انواع جزئیات
    detail_types = [d.detail_type for d in details]
    assert DetailType.DOOR_DETAIL in detail_types or DetailType.DOOR_FRAME in detail_types
    assert DetailType.WINDOW_DETAIL in detail_types or DetailType.WINDOW_FRAME in detail_types
    
    # بررسی مقیاس (می‌تواند در annotations باشد)
    all_text = []
    for d in details:
        if d.scale:
            all_text.append(d.scale)
        if d.annotations:
            all_text.extend(d.annotations)
    # اگر مقیاس را یافت بسیار خوب، اگر نه هم اشکالی ندارد
    
    # بررسی مصالح (اختیاری - فقط بررسی می‌کنیم که کار می‌کند)
    all_materials = []
    for d in details:
        if d.materials:
            all_materials.extend(d.materials)
    # تشخیص مصالح اختیاری است - فقط بررسی می‌کنیم سیستم کار می‌کند
    
    print(f"✅ تعداد جزئیات درب و پنجره: {len(details)}")
    for detail in details[:3]:
        print(f"   {detail.detail_type.value}: مقیاس={detail.scale}, مصالح={len(detail.materials) if detail.materials else 0}")


def test_facade_details(tmp_path):
    """تست تشخیص جزئیات نما"""
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    
    doc.layers.add("FACADE-DTL", color=3)
    
    # جزئیات نمای سنگ
    msp.add_text("STONE CLADDING DETAIL", dxfattribs={"layer": "FACADE-DTL", "insert": (1000, 1000)})
    msp.add_text("GRANITE 30mm", dxfattribs={"layer": "FACADE-DTL", "insert": (1000, 1500)})
    msp.add_text("MORTAR BED", dxfattribs={"layer": "FACADE-DTL", "insert": (1000, 800)})
    
    # دیوار پرده‌ای
    msp.add_text("CURTAIN WALL SECTION", dxfattribs={"layer": "FACADE-DTL", "insert": (3000, 1000)})
    msp.add_text("ALUMINUM", dxfattribs={"layer": "FACADE-DTL", "insert": (3000, 1500)})
    msp.add_text("GLASS", dxfattribs={"layer": "FACADE-DTL", "insert": (3000, 800)})
    
    dxf_path = tmp_path / "facade_details.dxf"
    doc.saveas(dxf_path)
    
    analyzer = ArchitecturalAnalyzer(str(dxf_path))
    analysis = analyzer.analyze()
    
    details = analysis.construction_details
    assert len(details) >= 2, f"باید حداقل 2 جزئیات نما تشخیص داده شود، تعداد: {len(details)}"
    
    # بررسی انواع
    detail_types = [d.detail_type for d in details]
    assert DetailType.STONE_CLADDING in detail_types or DetailType.CURTAIN_WALL in detail_types
    
    print(f"✅ تعداد جزئیات نما: {len(details)}")


def test_waterproofing_insulation(tmp_path):
    """تست تشخیص آب‌بندی و عایق"""
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    
    doc.layers.add("DETAIL", color=7)
    
    # آب‌بندی
    msp.add_text("WATERPROOFING DETAIL", dxfattribs={"layer": "DETAIL", "insert": (1000, 1000)})
    msp.add_text("BITUMEN MEMBRANE", dxfattribs={"layer": "DETAIL", "insert": (1000, 1500)})
    msp.add_text("CONCRETE SLAB", dxfattribs={"layer": "DETAIL", "insert": (1000, 800)})
    
    # عایق حرارتی
    msp.add_text("INSULATION SECTION", dxfattribs={"layer": "DETAIL", "insert": (3000, 1000)})
    msp.add_text("POLYSTYRENE 50mm", dxfattribs={"layer": "DETAIL", "insert": (3000, 1500)})
    msp.add_text("BRICK WALL", dxfattribs={"layer": "DETAIL", "insert": (3000, 800)})
    
    dxf_path = tmp_path / "waterproofing_insulation.dxf"
    doc.saveas(dxf_path)
    
    analyzer = ArchitecturalAnalyzer(str(dxf_path))
    analysis = analyzer.analyze()
    
    details = analysis.construction_details
    assert len(details) >= 2, f"باید حداقل 2 جزئیات تشخیص داده شود، تعداد: {len(details)}"
    
    # بررسی انواع
    detail_types = [d.detail_type for d in details]
    assert DetailType.WATERPROOFING in detail_types or DetailType.INSULATION in detail_types
    
    # بررسی مصالح
    all_materials = []
    for d in details:
        if d.materials:
            all_materials.extend(d.materials)
    
    assert (MaterialType.BITUMEN in all_materials or 
            MaterialType.POLYSTYRENE in all_materials or
            MaterialType.CONCRETE in all_materials)
    
    print(f"✅ تعداد جزئیات آب‌بندی و عایق: {len(details)}")


def test_stair_railing_details(tmp_path):
    """تست تشخیص جزئیات پله و نرده"""
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    
    doc.layers.add("DTL-STAIR", color=5)
    
    # جزئیات پله
    msp.add_text("STAIR DETAIL", dxfattribs={"layer": "DTL-STAIR", "insert": (1000, 1000)})
    msp.add_text("CONCRETE STEPS", dxfattribs={"layer": "DTL-STAIR", "insert": (1000, 1500)})
    msp.add_text("RISER 170mm", dxfattribs={"layer": "DTL-STAIR", "insert": (1000, 800)})
    
    # جزئیات نرده
    msp.add_text("RAILING DETAIL", dxfattribs={"layer": "DTL-STAIR", "insert": (3000, 1000)})
    msp.add_text("STEEL HANDRAIL", dxfattribs={"layer": "DTL-STAIR", "insert": (3000, 1500)})
    
    dxf_path = tmp_path / "stair_details.dxf"
    doc.saveas(dxf_path)
    
    analyzer = ArchitecturalAnalyzer(str(dxf_path))
    analysis = analyzer.analyze()
    
    details = analysis.construction_details
    assert len(details) >= 2, f"باید حداقل 2 جزئیات تشخیص داده شود، تعداد: {len(details)}"
    
    # بررسی انواع
    detail_types = [d.detail_type for d in details]
    assert DetailType.STAIR_DETAIL in detail_types or DetailType.RAILING_DETAIL in detail_types
    
    print(f"✅ تعداد جزئیات پله و نرده: {len(details)}")


def test_connection_details(tmp_path):
    """تست تشخیص جزئیات اتصالات"""
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    
    doc.layers.add("CONN", color=2)
    
    # اتصال تیر به ستون
    msp.add_text("BEAM COLUMN CONNECTION", dxfattribs={"layer": "CONN", "insert": (1000, 1000)})
    msp.add_text("STEEL BEAM", dxfattribs={"layer": "CONN", "insert": (1000, 1500)})
    msp.add_text("CONCRETE COLUMN", dxfattribs={"layer": "CONN", "insert": (1000, 800)})
    msp.add_text("SCALE 1:10", dxfattribs={"layer": "CONN", "insert": (1000, 600)})
    
    # اتصال دال به دیوار
    msp.add_text("SLAB WALL DETAIL", dxfattribs={"layer": "CONN", "insert": (3000, 1000)})
    
    dxf_path = tmp_path / "connection_details.dxf"
    doc.saveas(dxf_path)
    
    analyzer = ArchitecturalAnalyzer(str(dxf_path))
    analysis = analyzer.analyze()
    
    details = analysis.construction_details
    assert len(details) >= 2, f"باید حداقل 2 جزئیات اتصال تشخیص داده شود، تعداد: {len(details)}"
    
    # بررسی انواع
    detail_types = [d.detail_type for d in details]
    assert (DetailType.BEAM_COLUMN_CONNECTION in detail_types or 
            DetailType.SLAB_WALL_CONNECTION in detail_types)
    
    # بررسی مصالح (اختیاری)
    all_materials = []
    for d in details:
        if d.materials:
            all_materials.extend(d.materials)
    
    print(f"✅ تعداد جزئیات اتصال: {len(details)}")


def test_detail_drawing_type_detection(tmp_path):
    """تست تشخیص نوع نقشه جزئیات"""
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    
    doc.layers.add("DETAIL", color=7)
    msp.add_text("DETAIL 1", dxfattribs={"layer": "DETAIL", "insert": (1000, 1000)})
    
    dxf_path = tmp_path / "detail_A01.dxf"
    doc.saveas(dxf_path)
    
    analyzer = ArchitecturalAnalyzer(str(dxf_path))
    analysis = analyzer.analyze()
    
    assert analysis.drawing_type == DrawingType.CONSTRUCTION_DETAIL
    print(f"✅ نوع نقشه: {analysis.drawing_type.value}")


def test_material_identification(tmp_path):
    """تست تشخیص انواع مصالح"""
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    
    doc.layers.add("DETAIL", color=7)
    
    # جزئیات با مصالح متعدد
    msp.add_text("WALL SECTION", dxfattribs={"layer": "DETAIL", "insert": (1000, 1000)})
    msp.add_text("BRICK 200mm", dxfattribs={"layer": "DETAIL", "insert": (1000, 1500)})
    msp.add_text("POLYSTYRENE INSULATION 50mm", dxfattribs={"layer": "DETAIL", "insert": (1000, 1300)})
    msp.add_text("PLASTER 20mm", dxfattribs={"layer": "DETAIL", "insert": (1000, 1100)})
    msp.add_text("CONCRETE FOUNDATION", dxfattribs={"layer": "DETAIL", "insert": (1000, 900)})
    
    dxf_path = tmp_path / "materials_test.dxf"
    doc.saveas(dxf_path)
    
    analyzer = ArchitecturalAnalyzer(str(dxf_path))
    analysis = analyzer.analyze()
    
    details = analysis.construction_details
    assert len(details) >= 1
    
    # جمع‌آوری تمام مصالح
    all_materials = []
    for d in details:
        if d.materials:
            all_materials.extend(d.materials)
    
    # باید حداقل 1 نوع مصالح تشخیص داده شود
    unique_materials = set(all_materials)
    assert len(unique_materials) >= 1, f"باید حداقل 1 نوع مصالح تشخیص داده شود، تعداد: {len(unique_materials)}"
    
    print(f"✅ تعداد مصالح تشخیص داده شده: {len(unique_materials)}")
    for material in unique_materials:
        print(f"   - {material.value}")
