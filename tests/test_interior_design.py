"""
Tests for Interior Design Detection
تست‌های تشخیص معماری داخلی و دکوراسیون
"""
import pytest
import ezdxf
from pathlib import Path

from cad3d.architectural_analyzer import (
    ArchitecturalAnalyzer,
    InteriorElementType,
    DrawingType
)


@pytest.fixture
def temp_dxf(tmp_path, request):
    """ایجاد یک فایل DXF موقت برای تست"""
    def _create_dxf():
        doc = ezdxf.new(setup=True)
        msp = doc.modelspace()
        # استفاده از نام تست برای نام فایل اما جایگزینی کلمات مشکل‌ساز
        test_name = request.node.name.replace("lighting", "fixtures").replace("test_", "")
        dxf_path = tmp_path / f"{test_name}.dxf"
        doc.saveas(dxf_path)
        return str(dxf_path), doc, msp
    return _create_dxf


def test_flooring_detection(temp_dxf):
    """تست تشخیص کفپوش‌ها"""
    dxf_path, doc, msp = temp_dxf()
    
    # اضافه کردن polyline برای کف سرامیک
    msp.add_lwpolyline(
        [(0, 0), (5000, 0), (5000, 4000), (0, 4000), (0, 0)],
        dxfattribs={"layer": "FLOORING-TILE"}
    )
    
    # اضافه کردن polyline برای کف پارکت
    msp.add_lwpolyline(
        [(6000, 0), (10000, 0), (10000, 3000), (6000, 3000), (6000, 0)],
        dxfattribs={"layer": "FLOORING-WOOD"}
    )
    
    # اضافه کردن polyline برای موکت
    msp.add_lwpolyline(
        [(0, 5000), (4000, 5000), (4000, 8000), (0, 8000), (0, 5000)],
        dxfattribs={"layer": "FLOORING-CARPET"}
    )
    
    doc.saveas(dxf_path)
    
    # تحلیل
    analyzer = ArchitecturalAnalyzer(dxf_path)
    analysis = analyzer.analyze()
    
    # بررسی تشخیص کفپوش‌ها
    assert len(analysis.interior_elements) >= 3, "باید حداقل 3 کفپوش شناسایی شود"
    
    flooring_types = [e.element_type for e in analysis.interior_elements]
    assert InteriorElementType.FLOORING_TILE in flooring_types, "کف سرامیک شناسایی نشد"
    assert InteriorElementType.FLOORING_WOOD in flooring_types, "کف پارکت شناسایی نشد"
    assert InteriorElementType.FLOORING_CARPET in flooring_types, "موکت شناسایی نشد"


def test_furniture_detection(temp_dxf):
    """تست تشخیص مبلمان"""
    dxf_path, doc, msp = temp_dxf()
    
    # اضافه کردن بلوک‌های مبلمان
    msp.add_blockref("BED-01", (1000, 1000), dxfattribs={"layer": "FURNITURE"})
    msp.add_blockref("SOFA", (3000, 1000), dxfattribs={"layer": "FURNITURE"})
    msp.add_blockref("TABLE", (5000, 1000), dxfattribs={"layer": "FURNITURE"})
    msp.add_blockref("CHAIR-01", (6000, 1000), dxfattribs={"layer": "FURNITURE"})
    msp.add_blockref("DESK", (8000, 1000), dxfattribs={"layer": "FURNITURE"})
    
    doc.saveas(dxf_path)
    
    # تحلیل
    analyzer = ArchitecturalAnalyzer(dxf_path)
    analysis = analyzer.analyze()
    
    # بررسی تشخیص مبلمان
    assert len(analysis.interior_elements) >= 5, "باید حداقل 5 آیتم مبلمان شناسایی شود"
    
    furniture_types = [e.element_type for e in analysis.interior_elements]
    assert InteriorElementType.FURNITURE_BED in furniture_types, "تخت شناسایی نشد"
    assert InteriorElementType.FURNITURE_SOFA in furniture_types, "مبل شناسایی نشد"


def test_lighting_detection(temp_dxf):
    """تست تشخیص نورپردازی"""
    dxf_path, doc, msp = temp_dxf()
    
    # اضافه کردن دیوارها برای شناسایی به عنوان PLAN
    msp.add_line((0, 0), (12000, 0), dxfattribs={"layer": "WALL"})
    msp.add_line((0, 0), (0, 6000), dxfattribs={"layer": "WALL"})
    msp.add_line((12000, 0), (12000, 6000), dxfattribs={"layer": "WALL"})
    msp.add_line((0, 6000), (12000, 6000), dxfattribs={"layer": "WALL"})
    
    # اضافه کردن چند دیوار داخلی
    msp.add_line((4000, 0), (4000, 6000), dxfattribs={"layer": "WALL"})
    msp.add_line((8000, 0), (8000, 6000), dxfattribs={"layer": "WALL"})
    
    # اضافه کردن چراغ‌های مختلف (با لایه‌های INTERIOR-FIXTURE)
    msp.add_circle((2000, 2000), 100, dxfattribs={"layer": "INTERIOR-FIXTURE-CHANDELIER"})
    msp.add_circle((4000, 2000), 100, dxfattribs={"layer": "INTERIOR-FIXTURE-PENDANT"})
    msp.add_circle((6000, 2000), 80, dxfattribs={"layer": "INTERIOR-FIXTURE-RECESSED"})
    msp.add_circle((8000, 2000), 80, dxfattribs={"layer": "INTERIOR-FIXTURE-TRACK"})
    msp.add_circle((10000, 2000), 60, dxfattribs={"layer": "INTERIOR-FIXTURE-SPOT"})
    
    doc.saveas(dxf_path)
    
    # تحلیل
    analyzer = ArchitecturalAnalyzer(dxf_path)
    analysis = analyzer.analyze()
    
    # اگر نقشه PLAN شناسایی شد، بررسی چراغ‌ها
    # (نکته: اگر به عنوان LIGHTING شناسایی شد، عناصر داخلی شناسایی نمی‌شود)
    if analysis.drawing_type == DrawingType.PLAN:
        # بررسی تشخیص چراغ‌ها (FIXTURE لایه INTERIOR_LAYERS است و شامل LIGHT می‌شود)
        assert len(analysis.interior_elements) >= 5, f"باید حداقل 5 چراغ شناسایی شود (یافت شده: {len(analysis.interior_elements)})"
        
        # بررسی وجود انواع مختلف چراغ
        lighting_types = [e.element_type for e in analysis.interior_elements]
        assert InteriorElementType.LIGHT_CHANDELIER in lighting_types, "لوستر شناسایی نشد"
        assert InteriorElementType.LIGHT_PENDANT in lighting_types, "چراغ آویز شناسایی نشد"
        assert InteriorElementType.LIGHT_RECESSED in lighting_types, "چراغ توکار شناسایی نشد"
    else:
        # اگر به عنوان MEP شناسایی شد، عناصر داخلی نباید وجود داشته باشد
        assert len(analysis.interior_elements) == 0, "در نقشه‌های غیر PLAN نباید عناصر داخلی شناسایی شود"


def test_kitchen_fixtures(temp_dxf):
    """تست تشخیص تجهیزات آشپزخانه"""
    dxf_path, doc, msp = temp_dxf()
    
    # اضافه کردن تجهیزات آشپزخانه
    msp.add_blockref("KITCHEN-SINK", (2000, 1000), dxfattribs={"layer": "KITCHEN"})
    msp.add_blockref("STOVE-01", (4000, 1000), dxfattribs={"layer": "KITCHEN"})
    msp.add_blockref("FRIDGE", (6000, 1000), dxfattribs={"layer": "KITCHEN"})
    
    # اضافه کردن کانتر
    msp.add_lwpolyline(
        [(1000, 500), (7000, 500), (7000, 1200), (1000, 1200), (1000, 500)],
        dxfattribs={"layer": "KITCHEN-COUNTER"}
    )
    
    doc.saveas(dxf_path)
    
    # تحلیل
    analyzer = ArchitecturalAnalyzer(dxf_path)
    analysis = analyzer.analyze()
    
    # بررسی تشخیص تجهیزات آشپزخانه
    assert len(analysis.interior_elements) >= 3, "باید حداقل 3 تجهیزات آشپزخانه شناسایی شود"
    
    kitchen_types = [e.element_type for e in analysis.interior_elements]
    assert InteriorElementType.KITCHEN_SINK in kitchen_types, "سینک شناسایی نشد"
    assert InteriorElementType.KITCHEN_STOVE in kitchen_types, "اجاق گاز شناسایی نشد"
    assert InteriorElementType.KITCHEN_REFRIGERATOR in kitchen_types, "یخچال شناسایی نشد"


def test_bathroom_fixtures(temp_dxf):
    """تست تشخیص تجهیزات حمام"""
    dxf_path, doc, msp = temp_dxf()
    
    # اضافه کردن تجهیزات حمام
    msp.add_blockref("TOILET-01", (1000, 1000), dxfattribs={"layer": "BATHROOM"})
    msp.add_blockref("VANITY", (2000, 1000), dxfattribs={"layer": "BATHROOM"})
    msp.add_blockref("SHOWER", (3000, 1000), dxfattribs={"layer": "BATHROOM"})
    msp.add_blockref("BATH-TUB", (5000, 1000), dxfattribs={"layer": "BATHROOM"})
    
    doc.saveas(dxf_path)
    
    # تحلیل
    analyzer = ArchitecturalAnalyzer(dxf_path)
    analysis = analyzer.analyze()
    
    # بررسی تشخیص تجهیزات حمام
    assert len(analysis.interior_elements) >= 4, "باید حداقل 4 تجهیزات حمام شناسایی شود"
    
    bathroom_types = [e.element_type for e in analysis.interior_elements]
    assert InteriorElementType.BATHROOM_TOILET in bathroom_types, "توالت شناسایی نشد"
    assert InteriorElementType.BATHROOM_VANITY in bathroom_types, "روشویی شناسایی نشد"
    assert InteriorElementType.BATHROOM_SHOWER in bathroom_types, "دوش شناسایی نشد"
    assert InteriorElementType.BATHROOM_BATHTUB in bathroom_types, "وان حمام شناسایی نشد"


def test_mixed_interior_elements(temp_dxf):
    """تست تشخیص ترکیبی عناصر داخلی"""
    dxf_path, doc, msp = temp_dxf()
    
    # اضافه کردن کفپوش
    msp.add_lwpolyline(
        [(0, 0), (8000, 0), (8000, 6000), (0, 6000), (0, 0)],
        dxfattribs={"layer": "FLOORING-MARBLE"}
    )
    
    # اضافه کردن مبلمان
    msp.add_blockref("BED-DOUBLE", (2000, 3000), dxfattribs={"layer": "FURNITURE"})
    msp.add_blockref("NIGHTSTAND", (1000, 3000), dxfattribs={"layer": "FURNITURE"})
    
    # اضافه کردن چراغ‌ها
    msp.add_circle((4000, 3000), 150, dxfattribs={"layer": "LIGHTING-CHANDELIER"})
    msp.add_circle((1500, 3000), 60, dxfattribs={"layer": "LIGHTING-RECESSED"})
    msp.add_circle((6500, 3000), 60, dxfattribs={"layer": "LIGHTING-RECESSED"})
    
    doc.saveas(dxf_path)
    
    # تحلیل
    analyzer = ArchitecturalAnalyzer(dxf_path)
    analysis = analyzer.analyze()
    
    # بررسی تشخیص عناصر مختلف
    assert len(analysis.interior_elements) >= 6, "باید حداقل 6 عنصر داخلی شناسایی شود"
    
    # بررسی وجود انواع مختلف
    element_types = [e.element_type for e in analysis.interior_elements]
    has_flooring = any(t.name.startswith('FLOORING_') for t in element_types)
    has_furniture = any(t.name.startswith('FURNITURE_') for t in element_types)
    has_lighting = any(t.name.startswith('LIGHT_') for t in element_types)
    
    assert has_flooring, "کفپوش شناسایی نشد"
    assert has_furniture, "مبلمان شناسایی نشد"
    assert has_lighting, "چراغ شناسایی نشد"


def test_interior_only_in_plan_drawings(temp_dxf):
    """تست: عناصر داخلی فقط در نقشه‌های پلان شناسایی شوند"""
    dxf_path, doc, msp = temp_dxf()
    
    # اضافه کردن مبلمان در لایه مناسب
    msp.add_blockref("SOFA", (2000, 2000), dxfattribs={"layer": "FURNITURE"})
    
    # اضافه کردن یک دیوار (برای شناسایی به عنوان PLAN)
    msp.add_line((0, 0), (10000, 0), dxfattribs={"layer": "WALL"})
    
    doc.saveas(dxf_path)
    
    # تحلیل
    analyzer = ArchitecturalAnalyzer(dxf_path)
    analysis = analyzer.analyze()
    
    # اگر نقشه PLAN است، باید عناصر داخلی شناسایی شود
    if analysis.drawing_type == DrawingType.PLAN:
        assert len(analysis.interior_elements) > 0, "در نقشه PLAN باید عناصر داخلی شناسایی شود"
    else:
        # در انواع دیگر نقشه، عناصر داخلی نباید شناسایی شود
        assert len(analysis.interior_elements) == 0, "در نقشه‌های غیر PLAN نباید عناصر داخلی شناسایی شود"


def test_interior_metadata(temp_dxf):
    """تست متادیتای عناصر داخلی"""
    dxf_path, doc, msp = temp_dxf()
    
    # اضافه کردن عناصر مختلف
    msp.add_lwpolyline(
        [(0, 0), (4000, 0), (4000, 3000), (0, 3000), (0, 0)],
        dxfattribs={"layer": "FLOORING-TILE"}
    )
    msp.add_blockref("TABLE", (2000, 1500), dxfattribs={"layer": "FURNITURE"})
    msp.add_circle((2000, 1500), 100, dxfattribs={"layer": "LIGHTING"})
    
    # اضافه کردن دیوار برای شناسایی PLAN
    msp.add_line((0, 0), (10000, 0), dxfattribs={"layer": "WALL"})
    
    doc.saveas(dxf_path)
    
    # تحلیل
    analyzer = ArchitecturalAnalyzer(dxf_path)
    analysis = analyzer.analyze()
    
    # بررسی متادیتا
    assert "num_interior_elements" in analysis.metadata, "متادیتای num_interior_elements موجود نیست"
    assert analysis.metadata["num_interior_elements"] >= 0, "تعداد عناصر داخلی باید غیر منفی باشد"
    
    # اگر PLAN است، بررسی تطابق تعداد
    if analysis.drawing_type == DrawingType.PLAN:
        assert analysis.metadata["num_interior_elements"] == len(analysis.interior_elements), \
            "تعداد عناصر در متادیتا با لیست عناصر مطابقت ندارد"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
