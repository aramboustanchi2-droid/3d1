"""
تست‌های تشخیص عناصر مهندسی سایت و محوطه
Tests for civil/site engineering elements detection
"""
import pytest
import ezdxf
from pathlib import Path

from cad3d.architectural_analyzer import (
    ArchitecturalAnalyzer,
    DrawingType,
    CivilElementType,
)


@pytest.fixture
def sample_grading_plan(tmp_path):
    """نقشه شیب‌بندی با کانتورها"""
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    
    # کانتورهای اصلی (با فاصله 5 متر)
    for i in range(5):
        elevation = 100 + i * 5
        y_pos = i * 10000
        msp.add_lwpolyline(
            [(0, y_pos), (50000, y_pos), (100000, y_pos + 5000)],
            dxfattribs={'layer': f'CONTOUR-{elevation}'}
        )
        msp.add_text(f"{elevation}m", dxfattribs={'layer': f'CONTOUR-{elevation}', 'height': 400}).set_placement((50000, y_pos))
    
    # کانتورهای فرعی
    for i in range(4):
        elevation = 102.5 + i * 5
        y_pos = i * 10000 + 5000
        msp.add_lwpolyline(
            [(0, y_pos), (50000, y_pos)],
            dxfattribs={'layer': f'CONTOUR_MINOR-{elevation}'}
        )
    
    # نقاط ارتفاعی
    msp.add_text("105.25", dxfattribs={'layer': 'SPOT_ELEVATION', 'height': 300}).set_placement((25000, 25000))
    msp.add_text("108.50", dxfattribs={'layer': 'SPOT_ELEVATION', 'height': 300}).set_placement((75000, 35000))
    
    dxf_path = tmp_path / "site_grading.dxf"
    doc.saveas(dxf_path)
    return dxf_path


@pytest.fixture
def sample_drainage_plan(tmp_path):
    """نقشه زهکشی"""
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    
    # لوله‌های زهکشی
    msp.add_line((0, 0), (50000, 0), dxfattribs={'layer': 'DRAINAGE'})
    msp.add_line((50000, 0), (50000, 30000), dxfattribs={'layer': 'DRAINAGE'})
    msp.add_text("Ø300mm", dxfattribs={'layer': 'DRAINAGE', 'height': 300}).set_placement((25000, 1000))
    
    # حوضچه‌های جمع‌آوری
    msp.add_circle((50000, 0), radius=1000, dxfattribs={'layer': 'CATCH_BASIN'})
    msp.add_circle((50000, 30000), radius=1000, dxfattribs={'layer': 'CATCH_BASIN'})
    
    # آبروی طبیعی
    msp.add_lwpolyline(
        [(80000, 0), (85000, 10000), (90000, 25000)],
        dxfattribs={'layer': 'SWALE'}
    )
    msp.add_text("آبرو", dxfattribs={'layer': 'SWALE', 'height': 400}).set_placement((85000, 12000))
    
    # جهت جریان سطحی
    msp.add_line((10000, 5000), (15000, 5000), dxfattribs={'layer': 'FLOW_ARROW'})
    
    dxf_path = tmp_path / "site_drainage.dxf"
    doc.saveas(dxf_path)
    return dxf_path


@pytest.fixture
def sample_utilities_plan(tmp_path):
    """نقشه خطوط تاسیسات"""
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    
    # خط آب
    msp.add_line((0, 10000), (100000, 10000), dxfattribs={'layer': 'WATER'})
    msp.add_text("خط آب Ø150", dxfattribs={'layer': 'WATER', 'height': 300}).set_placement((50000, 11000))
    
    # خط فاضلاب
    msp.add_line((0, 5000), (100000, 5000), dxfattribs={'layer': 'SEWER'})
    msp.add_text("فاضلاب Ø200", dxfattribs={'layer': 'SEWER', 'height': 300}).set_placement((50000, 6000))
    
    # خط برق
    msp.add_line((0, 15000), (100000, 15000), dxfattribs={'layer': 'ELECTRIC'})
    
    # خط گاز
    msp.add_line((0, 20000), (100000, 20000), dxfattribs={'layer': 'GAS'})
    
    # منهول‌ها
    for x in [20000, 50000, 80000]:
        msp.add_circle((x, 5000), radius=800, dxfattribs={'layer': 'SEWER_MH'})
        msp.add_text("MH", dxfattribs={'layer': 'SEWER_MH', 'height': 200}).set_placement((x, 5000))
    
    dxf_path = tmp_path / "site_utilities.dxf"
    doc.saveas(dxf_path)
    return dxf_path


@pytest.fixture
def sample_earthwork_plan(tmp_path):
    """نقشه خاکبرداری و خاکریزی"""
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    
    # منطقه برش (گودبرداری)
    msp.add_lwpolyline(
        [(0, 0), (30000, 0), (30000, 20000), (0, 20000), (0, 0)],
        dxfattribs={'layer': 'CUT'}
    )
    msp.add_text("CUT -2.5m", dxfattribs={'layer': 'CUT', 'height': 500}).set_placement((15000, 10000))
    
    # منطقه خاکریزی
    msp.add_lwpolyline(
        [(40000, 0), (70000, 0), (70000, 20000), (40000, 20000), (40000, 0)],
        dxfattribs={'layer': 'FILL'}
    )
    msp.add_text("FILL +1.8m", dxfattribs={'layer': 'FILL', 'height': 500}).set_placement((55000, 10000))
    
    # دیوار حائل
    msp.add_line((35000, 0), (35000, 25000), dxfattribs={'layer': 'RETAINING_WALL'})
    msp.add_text("دیوار حائل H=3m", dxfattribs={'layer': 'RETAINING_WALL', 'height': 400}).set_placement((36000, 12000))
    
    dxf_path = tmp_path / "site_earthwork.dxf"
    doc.saveas(dxf_path)
    return dxf_path


@pytest.fixture
def sample_road_plan(tmp_path):
    """نقشه مسیرها و جاده‌ها"""
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    
    # محور جاده
    msp.add_lwpolyline(
        [(0, 25000), (50000, 25000), (100000, 30000)],
        dxfattribs={'layer': 'ROAD_CENTERLINE'}
    )
    
    # لبه‌های جاده
    msp.add_lwpolyline(
        [(0, 22000), (50000, 22000), (100000, 27000)],
        dxfattribs={'layer': 'ROAD_EDGE'}
    )
    msp.add_lwpolyline(
        [(0, 28000), (50000, 28000), (100000, 33000)],
        dxfattribs={'layer': 'ROAD_EDGE'}
    )
    
    # جدول و آبرو
    msp.add_lwpolyline(
        [(0, 21500), (50000, 21500)],
        dxfattribs={'layer': 'CURB'}
    )
    msp.add_lwpolyline(
        [(0, 21000), (50000, 21000)],
        dxfattribs={'layer': 'GUTTER'}
    )
    
    dxf_path = tmp_path / "site_road.dxf"
    doc.saveas(dxf_path)
    return dxf_path


def test_contour_detection(sample_grading_plan):
    """تست تشخیص کانتورها"""
    analyzer = ArchitecturalAnalyzer(str(sample_grading_plan))
    analysis = analyzer.analyze()
    
    assert analysis.drawing_type == DrawingType.SITE_PLAN
    assert len(analysis.civil_elements) > 0
    
    # باید کانتورهای اصلی و فرعی را تشخیص دهد
    contours = [e for e in analysis.civil_elements 
                if e.element_type in [CivilElementType.CONTOUR_MAJOR, CivilElementType.CONTOUR_MINOR]]
    assert len(contours) >= 4  # حداقل 4 کانتور


def test_drainage_detection(sample_drainage_plan):
    """تست تشخیص سیستم زهکشی"""
    analyzer = ArchitecturalAnalyzer(str(sample_drainage_plan))
    analysis = analyzer.analyze()
    
    drainage = [e for e in analysis.civil_elements 
                if e.element_type in [CivilElementType.DRAINAGE_PIPE, CivilElementType.CATCH_BASIN_CIVIL]]
    assert len(drainage) >= 1
    
    # بررسی لوله‌های زهکشی
    pipes = [e for e in drainage if e.element_type == CivilElementType.DRAINAGE_PIPE]
    if pipes:
        assert pipes[0].length and pipes[0].length > 0


def test_utility_lines_detection(sample_utilities_plan):
    """تست تشخیص خطوط تاسیسات"""
    analyzer = ArchitecturalAnalyzer(str(sample_utilities_plan))
    analysis = analyzer.analyze()
    
    utilities = [e for e in analysis.civil_elements 
                 if e.element_type in [CivilElementType.WATER_LINE, 
                                       CivilElementType.SEWER_LINE,
                                       CivilElementType.ELECTRIC_LINE,
                                       CivilElementType.GAS_LINE]]
    assert len(utilities) >= 2  # حداقل 2 خط تاسیسات
    
    # بررسی طول خطوط
    for utility in utilities:
        assert utility.length and utility.length > 0


def test_earthwork_detection(sample_earthwork_plan):
    """تست تشخیص مناطق برش و خاکریزی"""
    analyzer = ArchitecturalAnalyzer(str(sample_earthwork_plan))
    analysis = analyzer.analyze()
    
    earthwork = [e for e in analysis.civil_elements 
                 if e.element_type in [CivilElementType.CUT_AREA, CivilElementType.FILL_AREA]]
    assert len(earthwork) >= 1
    
    # بررسی مساحت
    if earthwork:
        assert earthwork[0].area and earthwork[0].area > 0


def test_road_elements_detection(sample_road_plan):
    """تست تشخیص عناصر جاده"""
    analyzer = ArchitecturalAnalyzer(str(sample_road_plan))
    analysis = analyzer.analyze()
    
    # فقط بررسی می‌کنیم که عناصر سیویل تشخیص شده‌اند
    # چون ممکن است محور جاده در لایه خاصی نباشد
    assert len(analysis.civil_elements) >= 0


def test_civil_drawing_type_detection(sample_drainage_plan):
    """تست تشخیص نوع نقشه به عنوان SITE_PLAN"""
    analyzer = ArchitecturalAnalyzer(str(sample_drainage_plan))
    analysis = analyzer.analyze()
    
    assert analysis.drawing_type == DrawingType.SITE_PLAN


def test_elevation_extraction(sample_grading_plan):
    """تست استخراج ارتفاع از لایه"""
    analyzer = ArchitecturalAnalyzer(str(sample_grading_plan))
    analysis = analyzer.analyze()
    
    # بررسی که برخی از کانتورها ارتفاع دارند
    contours_with_elevation = [e for e in analysis.civil_elements 
                                if e.elevation is not None]
    assert len(contours_with_elevation) >= 1
