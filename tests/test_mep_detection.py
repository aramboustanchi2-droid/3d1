"""
Tests for MEP (Mechanical, Electrical, Plumbing) Element Detection
تست‌های تشخیص عناصر تأسیساتی
"""
import pytest
import ezdxf
from pathlib import Path
from cad3d.architectural_analyzer import (
    ArchitecturalAnalyzer,
    MEPElementType,
    DrawingType,
)


def test_plumbing_detection(tmp_path):
    """تست تشخیص لوله‌های آب و فاضلاب"""
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    
    doc.layers.add("P-WATER", color=1)
    doc.layers.add("P-DRAIN", color=2)
    
    # لوله آب سرد CW
    line1 = msp.add_line((1000, 1000), (5000, 1000), dxfattribs={"layer": "P-WATER"})
    msp.add_text("CW Ø100mm", dxfattribs={"layer": "P-WATER", "insert": (3000, 1200)})
    
    # لوله آب گرم HW
    line2 = msp.add_line((1000, 2000), (5000, 2000), dxfattribs={"layer": "P-WATER"})
    msp.add_text("HW Ø50mm", dxfattribs={"layer": "P-WATER", "insert": (3000, 2200)})
    
    # لوله فاضلاب
    line3 = msp.add_line((1000, 3000), (5000, 3000), dxfattribs={"layer": "P-DRAIN"})
    msp.add_text("DRAIN Ø100mm", dxfattribs={"layer": "P-DRAIN", "insert": (3000, 3200)})
    
    dxf_path = tmp_path / "plumbing_plan.dxf"
    doc.saveas(dxf_path)
    
    analyzer = ArchitecturalAnalyzer(str(dxf_path))
    analysis = analyzer.analyze()
    
    # بررسی نتایج
    pipes = [e for e in analysis.mep_elements if e.element_type in [
        MEPElementType.WATER_PIPE, MEPElementType.DRAIN_PIPE
    ]]
    assert len(pipes) >= 3, f"باید حداقل 3 لوله تشخیص داده شود، تعداد: {len(pipes)}"
    
    # بررسی لوله آب
    water_pipes = [p for p in pipes if p.element_type == MEPElementType.WATER_PIPE]
    assert len(water_pipes) >= 2, f"باید حداقل 2 لوله آب تشخیص داده شود"
    
    # بررسی لوله فاضلاب
    drain_pipes = [p for p in pipes if p.element_type == MEPElementType.DRAIN_PIPE]
    assert len(drain_pipes) >= 1, f"باید حداقل 1 لوله فاضلاب تشخیص داده شود"
    
    print(f"✅ تعداد لوله‌های تشخیص داده شده: {len(pipes)}")
    for pipe in pipes:
        print(f"   {pipe.element_type.value}: {pipe.size_designation}, System: {pipe.system}")


def test_hvac_detection(tmp_path):
    """تست تشخیص سیستم HVAC"""
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    
    doc.layers.add("H-DUCT", color=3)
    doc.layers.add("H-SUPPLY", color=4)
    
    # کانال هوا
    duct_points = [(1000, 1000), (5000, 1000), (5000, 1500), (1000, 1500)]
    msp.add_lwpolyline(duct_points, dxfattribs={"layer": "H-DUCT"})
    msp.add_text("600x400", dxfattribs={"layer": "H-DUCT", "insert": (3000, 1250)})
    
    # دریچه هوا
    diffuser1 = msp.add_circle((2000, 3000, 0), radius=150, dxfattribs={"layer": "H-SUPPLY"})
    msp.add_text("DIFFUSER 300x300", dxfattribs={"layer": "H-SUPPLY", "insert": (2000, 3300)})
    
    # FCU
    fcu_points = [(3000, 4000), (3500, 4000), (3500, 4800), (3000, 4800), (3000, 4000)]
    fcu = msp.add_lwpolyline(fcu_points, dxfattribs={"layer": "H-SUPPLY"})
    fcu.close()
    msp.add_text("FCU-1", dxfattribs={"layer": "H-SUPPLY", "insert": (3250, 4400)})
    
    dxf_path = tmp_path / "hvac_plan.dxf"
    doc.saveas(dxf_path)
    
    analyzer = ArchitecturalAnalyzer(str(dxf_path))
    analysis = analyzer.analyze()
    
    hvac_elements = [e for e in analysis.mep_elements if e.element_type in [
        MEPElementType.DUCT, MEPElementType.DIFFUSER, MEPElementType.FCU
    ]]
    assert len(hvac_elements) >= 2, f"باید حداقل 2 عنصر HVAC تشخیص داده شود، تعداد: {len(hvac_elements)}"
    
    print(f"✅ تعداد عناصر HVAC تشخیص داده شده: {len(hvac_elements)}")
    for elem in hvac_elements:
        print(f"   {elem.element_type.value}: {elem.size_designation}")


def test_electrical_detection(tmp_path):
    """تست تشخیص سیستم برق"""
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    
    doc.layers.add("E-POWER", color=5)
    
    # تابلو برق
    panel = msp.add_circle((2000, 2000, 0), radius=300, dxfattribs={"layer": "E-POWER"})
    msp.add_text("PANEL DB-1 380V", dxfattribs={"layer": "E-POWER", "insert": (2000, 2400)})
    
    # پریز
    outlet1 = msp.add_circle((3000, 1000, 0), radius=50, dxfattribs={"layer": "E-POWER"})
    msp.add_text("OUTLET 220V", dxfattribs={"layer": "E-POWER", "insert": (3000, 1150)})
    
    # کلید
    switch1 = msp.add_circle((4000, 1000, 0), radius=50, dxfattribs={"layer": "E-POWER"})
    msp.add_text("SWITCH", dxfattribs={"layer": "E-POWER", "insert": (4000, 1150)})
    
    dxf_path = tmp_path / "electrical_plan.dxf"
    doc.saveas(dxf_path)
    
    analyzer = ArchitecturalAnalyzer(str(dxf_path))
    analysis = analyzer.analyze()
    
    electrical_elements = [e for e in analysis.mep_elements if e.element_type in [
        MEPElementType.PANEL, MEPElementType.OUTLET, MEPElementType.SWITCH
    ]]
    assert len(electrical_elements) >= 3, f"باید حداقل 3 عنصر برقی تشخیص داده شود، تعداد: {len(electrical_elements)}"
    
    # بررسی تابلو
    panels = [e for e in electrical_elements if e.element_type == MEPElementType.PANEL]
    assert len(panels) >= 1, "باید حداقل 1 تابلو تشخیص داده شود"
    assert panels[0].voltage is not None, "ولتاژ تابلو باید تشخیص داده شود"
    
    print(f"✅ تعداد عناصر برقی تشخیص داده شده: {len(electrical_elements)}")
    for elem in electrical_elements:
        info = f"   {elem.element_type.value}: {elem.size_designation}"
        if elem.voltage:
            info += f", Voltage: {elem.voltage}"
        print(info)


def test_lighting_detection(tmp_path):
    """تست تشخیص سیستم روشنایی"""
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    
    doc.layers.add("E-LITE", color=6)
    
    # چراغ‌های سقفی
    for i in range(4):
        x = 2000 + i * 1000
        light = msp.add_circle((x, 3000, 0), radius=100, dxfattribs={"layer": "E-LITE"})
        msp.add_text(f"LED 18W", dxfattribs={"layer": "E-LITE", "insert": (x, 3200)})
    
    dxf_path = tmp_path / "lighting_plan.dxf"
    doc.saveas(dxf_path)
    
    analyzer = ArchitecturalAnalyzer(str(dxf_path))
    analysis = analyzer.analyze()
    
    lights = [e for e in analysis.mep_elements if e.element_type == MEPElementType.LIGHT_FIXTURE]
    assert len(lights) >= 4, f"باید حداقل 4 چراغ تشخیص داده شود، تعداد: {len(lights)}"
    
    # بررسی توان
    lights_with_power = [l for l in lights if l.power is not None and l.power > 0]
    assert len(lights_with_power) >= 1, "حداقل یک چراغ باید دارای مشخصات توان باشد"
    
    print(f"✅ تعداد چراغ‌های تشخیص داده شده: {len(lights)}")
    for light in lights[:3]:
        info = f"   {light.size_designation}"
        if light.power:
            info += f", Power: {light.power}W"
        print(info)


def test_fire_protection_detection(tmp_path):
    """تست تشخیص سیستم اطفاء حریق"""
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    
    doc.layers.add("FP-SPRNK", color=7)
    
    # اسپرینکلرها با برچسب
    sprinkler1 = msp.add_circle((2000, 2000, 0), radius=80, dxfattribs={"layer": "FP-SPRNK"})
    msp.add_text("SPRINKLER", dxfattribs={"layer": "FP-SPRNK", "insert": (2000, 2200)})
    
    sprinkler2 = msp.add_circle((4000, 2000, 0), radius=80, dxfattribs={"layer": "FP-SPRNK"})
    msp.add_text("SPK", dxfattribs={"layer": "FP-SPRNK", "insert": (4000, 2200)})
    
    # دتکتور دود
    smoke = msp.add_circle((3000, 4000, 0), radius=100, dxfattribs={"layer": "FP-SPRNK"})
    msp.add_text("SMOKE DETECTOR", dxfattribs={"layer": "FP-SPRNK", "insert": (3000, 4200)})
    
    # آلارم حریق
    alarm = msp.add_circle((5000, 4000, 0), radius=100, dxfattribs={"layer": "FP-SPRNK"})
    msp.add_text("FIRE ALARM", dxfattribs={"layer": "FP-SPRNK", "insert": (5000, 4200)})
    
    dxf_path = tmp_path / "fire_protection_plan.dxf"
    doc.saveas(dxf_path)
    
    analyzer = ArchitecturalAnalyzer(str(dxf_path))
    analysis = analyzer.analyze()
    
    fire_elements = [e for e in analysis.mep_elements if e.element_type in [
        MEPElementType.SPRINKLER, MEPElementType.SMOKE_DETECTOR, MEPElementType.FIRE_ALARM
    ]]
    assert len(fire_elements) >= 4, f"باید حداقل 4 عنصر اطفاء حریق تشخیص داده شود، تعداد: {len(fire_elements)}"
    
    sprinklers = [e for e in fire_elements if e.element_type == MEPElementType.SPRINKLER]
    assert len(sprinklers) >= 2, "باید حداقل 2 اسپرینکلر تشخیص داده شود"
    
    print(f"✅ تعداد عناصر اطفاء حریق تشخیص داده شده: {len(fire_elements)}")
    for elem in fire_elements[:5]:
        print(f"   {elem.element_type.value}: {elem.size_designation}")


def test_mep_drawing_type_detection(tmp_path):
    """تست تشخیص نوع نقشه تأسیساتی"""
    # نقشه لوله‌کشی
    doc1 = ezdxf.new(setup=True)
    doc1.modelspace()
    dxf_path1 = tmp_path / "plumbing_plan.dxf"
    doc1.saveas(dxf_path1)
    
    analyzer1 = ArchitecturalAnalyzer(str(dxf_path1))
    analysis1 = analyzer1.analyze()
    assert analysis1.drawing_type == DrawingType.PLUMBING, \
        f"نوع نقشه باید PLUMBING باشد، نه {analysis1.drawing_type.value}"
    
    # نقشه HVAC
    doc2 = ezdxf.new(setup=True)
    doc2.modelspace()
    dxf_path2 = tmp_path / "hvac_layout.dxf"
    doc2.saveas(dxf_path2)
    
    analyzer2 = ArchitecturalAnalyzer(str(dxf_path2))
    analysis2 = analyzer2.analyze()
    assert analysis2.drawing_type == DrawingType.HVAC, \
        f"نوع نقشه باید HVAC باشد، نه {analysis2.drawing_type.value}"
    
    # نقشه برق
    doc3 = ezdxf.new(setup=True)
    doc3.modelspace()
    dxf_path3 = tmp_path / "electrical_plan.dxf"
    doc3.saveas(dxf_path3)
    
    analyzer3 = ArchitecturalAnalyzer(str(dxf_path3))
    analysis3 = analyzer3.analyze()
    assert analysis3.drawing_type == DrawingType.ELECTRICAL, \
        f"نوع نقشه باید ELECTRICAL باشد، نه {analysis3.drawing_type.value}"
    
    print("✅ تشخیص نوع نقشه تأسیساتی موفقیت‌آمیز بود")


def test_mixed_mep_architectural(tmp_path):
    """تست نقشه ترکیبی (معماری + تأسیسات)"""
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    
    # لایه‌های معماری
    doc.layers.add("WALLS", color=7)
    doc.layers.add("ROOMS", color=8)
    
    # لایه‌های تأسیسات
    doc.layers.add("P-WATER", color=1)
    doc.layers.add("E-LITE", color=6)
    
    # دیوارها
    wall_points = [(0, 0), (10000, 0), (10000, 8000), (0, 8000), (0, 0)]
    msp.add_lwpolyline(wall_points, dxfattribs={"layer": "WALLS"}).close()
    
    # اتاق
    room_points = [(1000, 1000), (5000, 1000), (5000, 4000), (1000, 4000), (1000, 1000)]
    msp.add_lwpolyline(room_points, dxfattribs={"layer": "ROOMS"}).close()
    msp.add_text("اتاق خواب", dxfattribs={"layer": "ROOMS", "insert": (3000, 2500)})
    
    # لوله آب
    msp.add_line((2000, 2000), (4000, 2000), dxfattribs={"layer": "P-WATER"})
    msp.add_text("CW Ø50", dxfattribs={"layer": "P-WATER", "insert": (3000, 2200)})
    
    # چراغ‌ها
    for i in range(3):
        x = 2000 + i * 1000
        msp.add_circle((x, 3000, 0), radius=100, dxfattribs={"layer": "E-LITE"})
        if i == 0:
            msp.add_text("LED 18W", dxfattribs={"layer": "E-LITE", "insert": (x, 3200)})
    
    dxf_path = tmp_path / "mixed_mep_plan.dxf"
    doc.saveas(dxf_path)
    
    analyzer = ArchitecturalAnalyzer(str(dxf_path))
    analysis = analyzer.analyze()
    
    # باید هم فضاها و هم عناصر تأسیساتی را تشخیص دهد
    assert len(analysis.rooms) >= 1, "باید حداقل 1 اتاق تشخیص داده شود"
    assert len(analysis.mep_elements) >= 3, "باید حداقل 3 عنصر تأسیساتی تشخیص داده شود"
    
    pipes = [e for e in analysis.mep_elements if e.element_type == MEPElementType.WATER_PIPE]
    lights = [e for e in analysis.mep_elements if e.element_type == MEPElementType.LIGHT_FIXTURE]
    
    assert len(pipes) >= 1, f"باید 1 لوله تشخیص داده شود، تعداد: {len(pipes)}"
    assert len(lights) >= 3, f"باید 3 چراغ تشخیص داده شود، تعداد: {len(lights)}"
    
    print(f"✅ نقشه ترکیبی:")
    print(f"   اتاق‌ها: {len(analysis.rooms)}")
    print(f"   دیوارها: {len(analysis.walls)}")
    print(f"   لوله‌ها: {len(pipes)}")
    print(f"   چراغ‌ها: {len(lights)}")
    print(f"   کل عناصر تأسیساتی: {len(analysis.mep_elements)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
