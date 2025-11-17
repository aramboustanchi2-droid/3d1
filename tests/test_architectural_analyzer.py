"""
Test architectural analyzer
"""
import ezdxf
from cad3d.architectural_analyzer import ArchitecturalAnalyzer, generate_analysis_report


def create_sample_floor_plan(path: str):
    """ساخت یک نقشه نمونه برای تست"""
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    
    # ایجاد لایه‌ها
    doc.layers.add("WALLS", color=7)
    doc.layers.add("ROOMS", color=3)
    doc.layers.add("DIMENSIONS", color=1)
    
    # دیوارهای خارجی (10m × 8m)
    msp.add_lwpolyline([
        (0, 0), (10000, 0), (10000, 8000), (0, 8000)
    ], close=True, dxfattribs={"layer": "WALLS"})
    
    # دیوار داخلی عمودی
    msp.add_line((4000, 0), (4000, 8000), dxfattribs={"layer": "WALLS"})
    
    # دیوار داخلی افقی
    msp.add_line((4000, 4000), (10000, 4000), dxfattribs={"layer": "WALLS"})
    
    # اتاق 1 - پذیرایی (سمت چپ)
    msp.add_lwpolyline([
        (100, 100), (3900, 100), (3900, 7900), (100, 7900)
    ], close=True, dxfattribs={"layer": "ROOMS"})
    msp.add_text("Living Room", dxfattribs={
        "layer": "ROOMS",
        "height": 200,
    }).set_placement((2000, 4000))
    msp.add_text("پذیرایی", dxfattribs={
        "layer": "ROOMS",
        "height": 200,
    }).set_placement((2000, 3600))
    
    # اتاق 2 - اتاق خواب (بالا راست)
    msp.add_lwpolyline([
        (4100, 4100), (9900, 4100), (9900, 7900), (4100, 7900)
    ], close=True, dxfattribs={"layer": "ROOMS"})
    msp.add_text("Bedroom", dxfattribs={
        "layer": "ROOMS",
        "height": 200,
    }).set_placement((7000, 6000))
    
    # اتاق 3 - آشپزخانه (پایین راست)
    msp.add_lwpolyline([
        (4100, 100), (9900, 100), (9900, 3900), (4100, 3900)
    ], close=True, dxfattribs={"layer": "ROOMS"})
    msp.add_text("Kitchen", dxfattribs={
        "layer": "ROOMS",
        "height": 200,
    }).set_placement((7000, 2000))
    msp.add_text("آشپزخانه", dxfattribs={
        "layer": "ROOMS",
        "height": 200,
    }).set_placement((7000, 1600))
    
    doc.saveas(path)
    print(f"✅ نقشه نمونه ایجاد شد: {path}")


def test_analyzer(tmp_path):
    """تست تحلیلگر معماری"""
    # ساخت نقشه نمونه
    sample_file = tmp_path / "sample_floor_plan.dxf"
    create_sample_floor_plan(str(sample_file))
    
    # تحلیل
    analyzer = ArchitecturalAnalyzer(str(sample_file))
    analysis = analyzer.analyze()
    
    # بررسی نتایج
    print("\n" + "="*80)
    print("نتایج تحلیل:")
    print("="*80)
    
    print(f"نوع نقشه: {analysis.drawing_type.value}")
    print(f"تعداد فضاها: {len(analysis.rooms)}")
    print(f"تعداد دیوارها: {len(analysis.walls)}")
    print(f"مساحت کل: {analysis.total_area:.2f} m²")
    
    print("\nفضاهای شناسایی شده:")
    for room in analysis.rooms:
        print(f"  - {room.name} ({room.space_type.value}): {room.area:.2f} m²")
        print(f"    ابعاد: {room.width:.2f}m × {room.length:.2f}m")
    
    # گزارش کامل
    report = generate_analysis_report(analysis)
    print("\n" + report)
    
    # ذخیره گزارش
    report_file = tmp_path / "analysis_report.txt"
    report_file.write_text(report, encoding="utf-8")
    print(f"\n✅ گزارش ذخیره شد: {report_file}")
    
    # بررسی‌های assertion
    assert len(analysis.rooms) >= 3, "باید حداقل 3 اتاق شناسایی شود"
    assert analysis.total_area > 60, "مساحت کل باید بیش از 60 متر مربع باشد"
    assert len(analysis.walls) > 0, "باید دیوارها شناسایی شوند"
    
    print("\n✅ همه تست‌ها موفق!")
