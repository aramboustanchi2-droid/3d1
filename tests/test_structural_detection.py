"""
Tests for Structural Element Detection
تست‌های تشخیص عناصر سازه‌ای
"""
import pytest
import ezdxf
from pathlib import Path
from cad3d.architectural_analyzer import (
    ArchitecturalAnalyzer,
    StructuralElementType,
    DrawingType,
)


def test_column_detection(tmp_path):
    """تست تشخیص ستون‌ها"""
    # ایجاد نقشه با ستون‌ها
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    
    # ایجاد لایه ستون
    doc.layers.add("S-COLS", color=1)
    
    # ستون دایره‌ای: C400 (قطر 400 میلی‌متر)
    circle = msp.add_circle((1000, 1000, 0), radius=200, dxfattribs={"layer": "S-COLS"})
    msp.add_text("C400", dxfattribs={"layer": "S-COLS", "insert": (1000, 1300)})
    
    # ستون مستطیلی: C30x40 (300x400 میلی‌متر)
    col_points = [(2000, 1000), (2300, 1000), (2300, 1400), (2000, 1400), (2000, 1000)]
    polyline = msp.add_lwpolyline(col_points, dxfattribs={"layer": "S-COLS"})
    polyline.close()
    msp.add_text("C30x40", dxfattribs={"layer": "S-COLS", "insert": (2150, 1500)})
    
    # ذخیره و تحلیل
    dxf_path = tmp_path / "structural_plan.dxf"
    doc.saveas(dxf_path)
    
    analyzer = ArchitecturalAnalyzer(str(dxf_path))
    analysis = analyzer.analyze()
    
    # بررسی نتایج
    columns = [e for e in analysis.structural_elements if e.element_type == StructuralElementType.COLUMN]
    assert len(columns) >= 2, f"باید حداقل 2 ستون تشخیص داده شود، تعداد: {len(columns)}"
    
    # بررسی ستون دایره‌ای
    circular_cols = [c for c in columns if abs(c.dimensions[0] - 400) < 50]  # قطر ~400
    assert len(circular_cols) >= 1, "ستون دایره‌ای تشخیص داده نشد"
    
    # بررسی ستون مستطیلی
    rect_cols = [c for c in columns if abs(c.dimensions[0] - 300) < 50 and abs(c.dimensions[1] - 400) < 50]
    assert len(rect_cols) >= 1, "ستون مستطیلی تشخیص داده نشد"
    
    print(f"✅ تعداد ستون‌های تشخیص داده شده: {len(columns)}")
    for col in columns:
        print(f"   {col.size_designation}: {col.dimensions[0]:.0f} × {col.dimensions[1]:.0f} mm")


def test_beam_detection(tmp_path):
    """تست تشخیص تیرها"""
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    
    doc.layers.add("S-BEAM", color=2)
    
    # تیر B25x50 (250x500 میلی‌متر)
    beam_line = msp.add_line((1000, 2000), (5000, 2000), dxfattribs={"layer": "S-BEAM"})
    msp.add_text("B25x50", dxfattribs={"layer": "S-BEAM", "insert": (3000, 2200)})
    
    # تیر دیگر
    beam_line2 = msp.add_line((1000, 3000), (4000, 3000), dxfattribs={"layer": "S-BEAM"})
    msp.add_text("BEAM 30x60", dxfattribs={"layer": "S-BEAM", "insert": (2500, 3200)})
    
    dxf_path = tmp_path / "beam_layout.dxf"
    doc.saveas(dxf_path)
    
    analyzer = ArchitecturalAnalyzer(str(dxf_path))
    analysis = analyzer.analyze()
    
    beams = [e for e in analysis.structural_elements if e.element_type == StructuralElementType.BEAM]
    assert len(beams) >= 2, f"باید حداقل 2 تیر تشخیص داده شود، تعداد: {len(beams)}"
    
    print(f"✅ تعداد تیرهای تشخیص داده شده: {len(beams)}")
    for beam in beams:
        print(f"   {beam.size_designation}")


def test_slab_detection(tmp_path):
    """تست تشخیص دال‌ها"""
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    
    doc.layers.add("S-SLAB", color=3)
    
    # دال با مساحت بزرگ (10m × 8m = 80 متر مربع)
    slab_points = [
        (0, 0), (10000, 0), (10000, 8000), (0, 8000), (0, 0)
    ]
    polyline = msp.add_lwpolyline(slab_points, dxfattribs={"layer": "S-SLAB"})
    polyline.close()
    msp.add_text("SLAB t=200mm", dxfattribs={"layer": "S-SLAB", "insert": (5000, 4000)})
    
    dxf_path = tmp_path / "slab_plan.dxf"
    doc.saveas(dxf_path)
    
    analyzer = ArchitecturalAnalyzer(str(dxf_path))
    analysis = analyzer.analyze()
    
    slabs = [e for e in analysis.structural_elements if e.element_type == StructuralElementType.SLAB]
    assert len(slabs) >= 1, f"باید حداقل 1 دال تشخیص داده شود، تعداد: {len(slabs)}"
    
    # بررسی ضخامت
    slab_with_thickness = [s for s in slabs if s.dimensions[2] == 200]
    assert len(slab_with_thickness) >= 1, "ضخامت دال (200mm) تشخیص داده نشد"
    
    print(f"✅ تعداد دال‌های تشخیص داده شده: {len(slabs)}")
    for slab in slabs:
        print(f"   {slab.size_designation}, ضخامت: {slab.dimensions[2]:.0f}mm")


def test_foundation_detection(tmp_path):
    """تست تشخیص فونداسیون"""
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    
    doc.layers.add("S-FOUND", color=4)
    
    # فونداسیون دایره‌ای (پی)
    foundation = msp.add_circle((3000, 3000, 0), radius=800, dxfattribs={"layer": "S-FOUND"})
    msp.add_text("F1", dxfattribs={"layer": "S-FOUND", "insert": (3000, 3900)})
    
    # فونداسیون مستطیلی
    fdn_points = [(5000, 2500), (7000, 2500), (7000, 3500), (5000, 3500), (5000, 2500)]
    polyline = msp.add_lwpolyline(fdn_points, dxfattribs={"layer": "S-FOUND"})
    polyline.close()
    msp.add_text("FOOTING 2x1m", dxfattribs={"layer": "S-FOUND", "insert": (6000, 3600)})
    
    dxf_path = tmp_path / "foundation_plan.dxf"
    doc.saveas(dxf_path)
    
    analyzer = ArchitecturalAnalyzer(str(dxf_path))
    analysis = analyzer.analyze()
    
    foundations = [e for e in analysis.structural_elements if e.element_type == StructuralElementType.FOUNDATION]
    assert len(foundations) >= 2, f"باید حداقل 2 فونداسیون تشخیص داده شود، تعداد: {len(foundations)}"
    
    print(f"✅ تعداد فونداسیون‌های تشخیص داده شده: {len(foundations)}")
    for fdn in foundations:
        print(f"   {fdn.size_designation}: {fdn.dimensions[0]:.0f} × {fdn.dimensions[1]:.0f} mm")


def test_drawing_type_detection(tmp_path):
    """تست تشخیص نوع نقشه سازه‌ای"""
    doc = ezdxf.new(setup=True)
    doc.modelspace()
    
    # نقشه فونداسیون
    dxf_path = tmp_path / "foundation_plan.dxf"
    doc.saveas(dxf_path)
    
    analyzer = ArchitecturalAnalyzer(str(dxf_path))
    analysis = analyzer.analyze()
    
    assert analysis.drawing_type == DrawingType.FOUNDATION, \
        f"نوع نقشه باید FOUNDATION باشد، نه {analysis.drawing_type.value}"
    
    # نقشه ستون
    dxf_path2 = tmp_path / "column_layout.dxf"
    doc.saveas(dxf_path2)
    
    analyzer2 = ArchitecturalAnalyzer(str(dxf_path2))
    analysis2 = analyzer2.analyze()
    
    assert analysis2.drawing_type == DrawingType.COLUMN_LAYOUT, \
        f"نوع نقشه باید COLUMN_LAYOUT باشد، نه {analysis2.drawing_type.value}"
    
    print("✅ تشخیص نوع نقشه سازه‌ای موفقیت‌آمیز بود")


def test_mixed_structural_architectural(tmp_path):
    """تست نقشه ترکیبی (معماری + سازه‌ای)"""
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    
    # لایه‌های معماری
    doc.layers.add("WALLS", color=7)
    doc.layers.add("ROOMS", color=8)
    
    # لایه‌های سازه‌ای
    doc.layers.add("S-COLS", color=1)
    doc.layers.add("S-BEAM", color=2)
    
    # دیوارهای معماری
    wall_points = [(0, 0), (10000, 0), (10000, 8000), (0, 8000), (0, 0)]
    msp.add_lwpolyline(wall_points, dxfattribs={"layer": "WALLS"}).close()
    
    # اتاق
    room_points = [(1000, 1000), (5000, 1000), (5000, 4000), (1000, 4000), (1000, 1000)]
    msp.add_lwpolyline(room_points, dxfattribs={"layer": "ROOMS"}).close()
    msp.add_text("اتاق خواب", dxfattribs={"layer": "ROOMS", "insert": (3000, 2500)})
    
    # ستون‌های سازه‌ای
    msp.add_circle((2000, 2000, 0), radius=200, dxfattribs={"layer": "S-COLS"})
    msp.add_text("C40", dxfattribs={"layer": "S-COLS", "insert": (2000, 2300)})
    
    msp.add_circle((4000, 2000, 0), radius=200, dxfattribs={"layer": "S-COLS"})
    msp.add_text("C40", dxfattribs={"layer": "S-COLS", "insert": (4000, 2300)})
    
    # تیر
    msp.add_line((2000, 2000), (4000, 2000), dxfattribs={"layer": "S-BEAM"})
    msp.add_text("B30x50", dxfattribs={"layer": "S-BEAM", "insert": (3000, 2100)})
    
    dxf_path = tmp_path / "mixed_plan.dxf"
    doc.saveas(dxf_path)
    
    analyzer = ArchitecturalAnalyzer(str(dxf_path))
    analysis = analyzer.analyze()
    
    # باید هم فضاها و هم عناصر سازه‌ای را تشخیص دهد
    assert len(analysis.rooms) >= 1, "باید حداقل 1 اتاق تشخیص داده شود"
    assert len(analysis.structural_elements) >= 3, "باید حداقل 3 عنصر سازه‌ای (2 ستون + 1 تیر) تشخیص داده شود"
    
    columns = [e for e in analysis.structural_elements if e.element_type == StructuralElementType.COLUMN]
    beams = [e for e in analysis.structural_elements if e.element_type == StructuralElementType.BEAM]
    
    assert len(columns) >= 2, f"باید 2 ستون تشخیص داده شود، تعداد: {len(columns)}"
    assert len(beams) >= 1, f"باید 1 تیر تشخیص داده شود، تعداد: {len(beams)}"
    
    print(f"✅ نقشه ترکیبی:")
    print(f"   اتاق‌ها: {len(analysis.rooms)}")
    print(f"   دیوارها: {len(analysis.walls)}")
    print(f"   ستون‌ها: {len(columns)}")
    print(f"   تیرها: {len(beams)}")
    print(f"   کل عناصر سازه‌ای: {len(analysis.structural_elements)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
