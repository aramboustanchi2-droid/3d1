"""
Tests for Safety & Security Detection
تست‌های تشخیص سیستم‌های ایمنی و امنیت
"""
import pytest
import ezdxf
from pathlib import Path

from cad3d.architectural_analyzer import (
    ArchitecturalAnalyzer,
    SafetySecurityElementType
)


@pytest.fixture
def temp_dxf(tmp_path):
    """ایجاد یک فایل DXF موقت برای تست"""
    def _create_dxf():
        doc = ezdxf.new(setup=True)
        msp = doc.modelspace()
        dxf_path = tmp_path / "test_safety.dxf"
        doc.saveas(dxf_path)
        return str(dxf_path), doc, msp
    return _create_dxf


def test_fire_alarm_detection(temp_dxf):
    """تست تشخیص سیستم اعلام حریق"""
    dxf_path, doc, msp = temp_dxf()
    
    # اضافه کردن پنل اعلام حریق
    msp.add_blockref("FIRE-ALARM-PANEL", (5000, 5000), dxfattribs={"layer": "FIRE-ALARM"})
    
    # اضافه کردن آشکارسازهای دود
    msp.add_blockref("SMOKE-DETECTOR", (2000, 2000), dxfattribs={"layer": "FIRE-ALARM"})
    msp.add_blockref("DETECTOR-01", (4000, 2000), dxfattribs={"layer": "FIRE-ALARM"})
    msp.add_blockref("HEAT-DETECTOR", (6000, 2000), dxfattribs={"layer": "FIRE-ALARM"})
    
    # اضافه کردن دکمه دستی
    msp.add_blockref("MANUAL-STATION", (3000, 4000), dxfattribs={"layer": "FIRE-ALARM"})
    
    # اضافه کردن آژیر و چراغ
    msp.add_blockref("FIRE-HORN", (7000, 3000), dxfattribs={"layer": "FIRE-ALARM"})
    
    doc.saveas(dxf_path)
    
    # تحلیل
    analyzer = ArchitecturalAnalyzer(dxf_path)
    analysis = analyzer.analyze()
    
    # بررسی تشخیص عناصر اعلام حریق
    assert len(analysis.safety_security_elements) >= 6, f"باید حداقل 6 عنصر اعلام حریق شناسایی شود (یافت شده: {len(analysis.safety_security_elements)})"
    
    element_types = [e.element_type for e in analysis.safety_security_elements]
    assert SafetySecurityElementType.FIRE_ALARM_PANEL in element_types, "پنل اعلام حریق شناسایی نشد"
    assert SafetySecurityElementType.FIRE_ALARM_DETECTOR in element_types, "آشکارساز دود شناسایی نشد"
    assert SafetySecurityElementType.FIRE_ALARM_MANUAL_STATION in element_types, "دکمه دستی شناسایی نشد"


def test_fire_suppression_detection(temp_dxf):
    """تست تشخیص سیستم اطفاء حریق"""
    dxf_path, doc, msp = temp_dxf()
    
    # اضافه کردن سرهای اسپرینکلر
    msp.add_circle((2000, 2000), 50, dxfattribs={"layer": "SPRINKLER"})
    msp.add_circle((4000, 2000), 50, dxfattribs={"layer": "SPRINKLER"})
    msp.add_circle((6000, 2000), 50, dxfattribs={"layer": "SPRINKLER"})
    msp.add_circle((8000, 2000), 50, dxfattribs={"layer": "SPRINKLER"})
    
    # اضافه کردن کپسول آتش‌نشانی
    msp.add_blockref("EXTINGUISHER-01", (1000, 4000), dxfattribs={"layer": "EXTINGUISHER"})
    msp.add_blockref("FIRE-EXTINGUISHER", (5000, 4000), dxfattribs={"layer": "EXTINGUISHER"})
    
    # اضافه کردن جعبه شیلنگ
    msp.add_blockref("HOSE-CABINET", (9000, 3000), dxfattribs={"layer": "FIRE-HOSE"})
    
    # اضافه کردن هیدرانت
    msp.add_blockref("FIRE-HYDRANT", (10000, 5000), dxfattribs={"layer": "HYDRANT"})
    
    doc.saveas(dxf_path)
    
    # تحلیل
    analyzer = ArchitecturalAnalyzer(dxf_path)
    analysis = analyzer.analyze()
    
    # بررسی تشخیص تجهیزات اطفاء حریق
    assert len(analysis.safety_security_elements) >= 8, f"باید حداقل 8 تجهیزات اطفاء شناسایی شود (یافت شده: {len(analysis.safety_security_elements)})"
    
    element_types = [e.element_type for e in analysis.safety_security_elements]
    assert SafetySecurityElementType.SPRINKLER_HEAD in element_types, "سر اسپرینکلر شناسایی نشد"
    assert SafetySecurityElementType.FIRE_EXTINGUISHER in element_types, "کپسول آتش‌نشانی شناسایی نشد"
    assert SafetySecurityElementType.FIRE_HOSE_CABINET in element_types, "جعبه شیلنگ شناسایی نشد"
    assert SafetySecurityElementType.FIRE_HYDRANT in element_types, "هیدرانت شناسایی نشد"


def test_emergency_exit_detection(temp_dxf):
    """تست تشخیص خروج اضطراری"""
    dxf_path, doc, msp = temp_dxf()
    
    # اضافه کردن درب خروج اضطراری
    msp.add_blockref("EXIT-DOOR-01", (5000, 1000), dxfattribs={"layer": "EMERGENCY-EXIT"})
    msp.add_blockref("EMERGENCY-DOOR", (10000, 1000), dxfattribs={"layer": "EXIT"})
    
    # اضافه کردن تابلوی خروج
    msp.add_blockref("EXIT-SIGN", (5000, 2000), dxfattribs={"layer": "EXIT"})
    msp.add_blockref("EXIT-SIGN-02", (10000, 2000), dxfattribs={"layer": "EXIT"})
    
    # اضافه کردن چراغ اضطراری
    msp.add_blockref("EMERGENCY-LIGHT", (3000, 3000), dxfattribs={"layer": "EMERGENCY"})
    msp.add_blockref("EMERGENCY-LAMP", (7000, 3000), dxfattribs={"layer": "EMERGENCY"})
    
    # اضافه کردن مسیر تخلیه
    msp.add_line((2000, 5000), (8000, 5000), dxfattribs={"layer": "EVACUATION"})
    
    # اضافه کردن نقطه تجمع
    msp.add_blockref("ASSEMBLY-POINT", (10000, 8000), dxfattribs={"layer": "EMERGENCY"})
    
    doc.saveas(dxf_path)
    
    # تحلیل
    analyzer = ArchitecturalAnalyzer(dxf_path)
    analysis = analyzer.analyze()
    
    # بررسی تشخیص خروج اضطراری
    assert len(analysis.safety_security_elements) >= 8, f"باید حداقل 8 عنصر خروج اضطراری شناسایی شود (یافت شده: {len(analysis.safety_security_elements)})"
    
    element_types = [e.element_type for e in analysis.safety_security_elements]
    assert SafetySecurityElementType.EXIT_DOOR in element_types, "درب خروج اضطراری شناسایی نشد"
    assert SafetySecurityElementType.EXIT_SIGN in element_types, "تابلوی خروج شناسایی نشد"
    assert SafetySecurityElementType.EMERGENCY_LIGHT in element_types, "چراغ اضطراری شناسایی نشد"
    assert SafetySecurityElementType.EVACUATION_ROUTE in element_types, "مسیر تخلیه شناسایی نشد"
    assert SafetySecurityElementType.ASSEMBLY_POINT in element_types, "نقطه تجمع شناسایی نشد"


def test_cctv_detection(temp_dxf):
    """تست تشخیص سیستم دوربین مداربسته"""
    dxf_path, doc, msp = temp_dxf()
    
    # اضافه کردن دوربین‌ها
    msp.add_blockref("CCTV-CAMERA-01", (2000, 2000), dxfattribs={"layer": "CCTV"})
    msp.add_blockref("CAMERA-02", (4000, 2000), dxfattribs={"layer": "CCTV"})
    msp.add_blockref("IP-CAMERA", (6000, 2000), dxfattribs={"layer": "CCTV"})
    msp.add_circle((8000, 2000), 100, dxfattribs={"layer": "CCTV"})
    
    # اضافه کردن DVR
    msp.add_blockref("DVR-16CH", (5000, 5000), dxfattribs={"layer": "CCTV"})
    
    doc.saveas(dxf_path)
    
    # تحلیل
    analyzer = ArchitecturalAnalyzer(dxf_path)
    analysis = analyzer.analyze()
    
    # بررسی تشخیص دوربین‌ها
    assert len(analysis.safety_security_elements) >= 5, f"باید حداقل 5 دوربین شناسایی شود (یافت شده: {len(analysis.safety_security_elements)})"
    
    element_types = [e.element_type for e in analysis.safety_security_elements]
    assert SafetySecurityElementType.CCTV_CAMERA in element_types, "دوربین مداربسته شناسایی نشد"
    assert SafetySecurityElementType.CCTV_DVR in element_types, "دستگاه DVR شناسایی نشد"


def test_access_control_detection(temp_dxf):
    """تست تشخیص کنترل تردد"""
    dxf_path, doc, msp = temp_dxf()
    
    # اضافه کردن کارتخوان‌ها
    msp.add_blockref("CARD-READER-01", (2000, 2000), dxfattribs={"layer": "ACCESS-CONTROL"})
    msp.add_blockref("ACCESS-READER", (4000, 2000), dxfattribs={"layer": "ACCESS"})
    
    # اضافه کردن اسکنر بیومتریک
    msp.add_blockref("BIOMETRIC-SCANNER", (6000, 2000), dxfattribs={"layer": "ACCESS"})
    msp.add_blockref("FINGER-PRINT", (8000, 2000), dxfattribs={"layer": "ACCESS"})
    
    # اضافه کردن پنل کنترل
    msp.add_blockref("ACCESS-PANEL", (5000, 5000), dxfattribs={"layer": "ACCESS-CONTROL"})
    
    # اضافه کردن قفل مغناطیسی
    msp.add_blockref("MAG-LOCK", (3000, 3000), dxfattribs={"layer": "ACCESS"})
    
    # اضافه کردن گیت
    msp.add_blockref("TURNSTILE-01", (7000, 4000), dxfattribs={"layer": "ACCESS"})
    
    doc.saveas(dxf_path)
    
    # تحلیل
    analyzer = ArchitecturalAnalyzer(dxf_path)
    analysis = analyzer.analyze()
    
    # بررسی تشخیص کنترل تردد
    assert len(analysis.safety_security_elements) >= 7, f"باید حداقل 7 تجهیزات کنترل تردد شناسایی شود (یافت شده: {len(analysis.safety_security_elements)})"
    
    element_types = [e.element_type for e in analysis.safety_security_elements]
    assert SafetySecurityElementType.ACCESS_CONTROL_READER in element_types, "کارتخوان شناسایی نشد"
    assert SafetySecurityElementType.BIOMETRIC_SCANNER in element_types, "اسکنر بیومتریک شناسایی نشد"
    assert SafetySecurityElementType.ACCESS_CONTROL_PANEL in element_types, "پنل کنترل تردد شناسایی نشد"
    assert SafetySecurityElementType.MAGNETIC_LOCK in element_types, "قفل مغناطیسی شناسایی نشد"
    assert SafetySecurityElementType.TURNSTILE in element_types, "گیت چرخشی شناسایی نشد"


def test_sensors_detection(temp_dxf):
    """تست تشخیص سنسورها"""
    dxf_path, doc, msp = temp_dxf()
    
    # اضافه کردن سنسور حرکتی
    msp.add_blockref("MOTION-SENSOR", (2000, 2000), dxfattribs={"layer": "SECURITY"})
    msp.add_blockref("PIR-SENSOR", (4000, 2000), dxfattribs={"layer": "SECURITY"})
    
    # اضافه کردن سنسور درب
    msp.add_blockref("DOOR-CONTACT-01", (6000, 2000), dxfattribs={"layer": "SECURITY"})
    
    # اضافه کردن دکمه اضطرار
    msp.add_blockref("PANIC-BUTTON", (8000, 3000), dxfattribs={"layer": "SECURITY"})
    
    doc.saveas(dxf_path)
    
    # تحلیل
    analyzer = ArchitecturalAnalyzer(dxf_path)
    analysis = analyzer.analyze()
    
    # بررسی تشخیص سنسورها
    assert len(analysis.safety_security_elements) >= 4, f"باید حداقل 4 سنسور شناسایی شود (یافت شده: {len(analysis.safety_security_elements)})"
    
    element_types = [e.element_type for e in analysis.safety_security_elements]
    assert SafetySecurityElementType.MOTION_SENSOR in element_types, "سنسور حرکتی شناسایی نشد"
    assert SafetySecurityElementType.DOOR_CONTACT in element_types, "سنسور درب شناسایی نشد"
    assert SafetySecurityElementType.PANIC_BUTTON in element_types, "دکمه اضطرار شناسایی نشد"


def test_mixed_safety_security(temp_dxf):
    """تست تشخیص ترکیبی سیستم‌های ایمنی و امنیت"""
    dxf_path, doc, msp = temp_dxf()
    
    # اعلام حریق
    msp.add_blockref("FIRE-DETECTOR", (2000, 2000), dxfattribs={"layer": "FIRE-ALARM"})
    
    # اطفاء حریق
    msp.add_circle((4000, 2000), 50, dxfattribs={"layer": "SPRINKLER"})
    msp.add_blockref("EXTINGUISHER", (6000, 2000), dxfattribs={"layer": "EXTINGUISHER"})
    
    # خروج اضطراری
    msp.add_blockref("EXIT-DOOR", (8000, 2000), dxfattribs={"layer": "EXIT"})
    msp.add_blockref("EMERGENCY-LIGHT", (10000, 2000), dxfattribs={"layer": "EMERGENCY"})
    
    # امنیت
    msp.add_blockref("CCTV-CAM", (2000, 5000), dxfattribs={"layer": "CCTV"})
    msp.add_blockref("CARD-READER", (4000, 5000), dxfattribs={"layer": "ACCESS"})
    msp.add_blockref("MOTION-DETECT", (6000, 5000), dxfattribs={"layer": "SECURITY"})
    
    doc.saveas(dxf_path)
    
    # تحلیل
    analyzer = ArchitecturalAnalyzer(dxf_path)
    analysis = analyzer.analyze()
    
    # بررسی تشخیص عناصر مختلف
    assert len(analysis.safety_security_elements) >= 8, f"باید حداقل 8 عنصر شناسایی شود (یافت شده: {len(analysis.safety_security_elements)})"
    
    # بررسی وجود انواع مختلف سیستم‌ها
    element_types = [e.element_type for e in analysis.safety_security_elements]
    
    has_fire_alarm = any(t.name.startswith('FIRE_ALARM_') for t in element_types)
    has_fire_suppression = any(t.name in ['SPRINKLER_HEAD', 'FIRE_EXTINGUISHER', 'FIRE_HOSE_CABINET'] for t in element_types)
    has_exit = any(t.name.startswith('EXIT_') or t.name == 'EMERGENCY_LIGHT' for t in element_types)
    has_security = any(t.name in ['CCTV_CAMERA', 'ACCESS_CONTROL_READER', 'MOTION_SENSOR'] for t in element_types)
    
    assert has_fire_alarm, "سیستم اعلام حریق شناسایی نشد"
    assert has_fire_suppression, "سیستم اطفاء حریق شناسایی نشد"
    assert has_exit, "خروج اضطراری شناسایی نشد"
    assert has_security, "سیستم امنیتی شناسایی نشد"


def test_safety_security_metadata(temp_dxf):
    """تست متادیتای ایمنی و امنیت"""
    dxf_path, doc, msp = temp_dxf()
    
    # اضافه کردن چند عنصر
    msp.add_blockref("FIRE-DETECTOR", (2000, 2000), dxfattribs={"layer": "FIRE-ALARM"})
    msp.add_blockref("CCTV-CAMERA", (4000, 2000), dxfattribs={"layer": "CCTV"})
    msp.add_blockref("EXIT-DOOR", (6000, 2000), dxfattribs={"layer": "EXIT"})
    
    doc.saveas(dxf_path)
    
    # تحلیل
    analyzer = ArchitecturalAnalyzer(dxf_path)
    analysis = analyzer.analyze()
    
    # بررسی متادیتا
    assert "num_safety_security_elements" in analysis.metadata, "متادیتای num_safety_security_elements موجود نیست"
    assert analysis.metadata["num_safety_security_elements"] >= 0, "تعداد عناصر ایمنی باید غیر منفی باشد"
    assert analysis.metadata["num_safety_security_elements"] == len(analysis.safety_security_elements), \
        "تعداد عناصر در متادیتا با لیست عناصر مطابقت ندارد"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
