"""
Professional Lighting & Illumination Detection
تشخیص نقشه‌های روشنایی و نورپردازی حرفه‌ای

شامل:
- نورپردازی داخلی (Interior Lighting)
- نورپردازی خارجی و نما (Facade Lighting)
- نورپردازی فضای سبز (Landscape Lighting)
- نورپردازی سالن‌ها و فضاهای عمومی
- سیستم‌های کنترل روشنایی هوشمند
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from enum import Enum


class LightingType(Enum):
    """انواع روشنایی"""
    # روشنایی داخلی
    RECESSED = "recessed"  # توکار
    PENDANT = "pendant"  # آویز
    TRACK = "track"  # ریلی
    CHANDELIER = "chandelier"  # لوستر
    WALL_SCONCE = "wall_sconce"  # دیواری
    CEILING_MOUNTED = "ceiling_mounted"  # سقفی
    UNDER_CABINET = "under_cabinet"  # زیر کابینت
    COVE = "cove"  # نور مخفی
    
    # روشنایی خارجی
    FACADE_UPLIGHTING = "facade_uplighting"  # نورافکن نما (از پایین)
    FACADE_DOWNLIGHTING = "facade_downlighting"  # نور نما (از بالا)
    WALL_WASHER = "wall_washer"  # شستشوی نوری دیوار
    BOLLARD = "bollard"  # پایه‌ای
    PATHWAY = "pathway"  # مسیر
    FLOOD = "flood"  # سیل‌نور
    SPOT = "spot"  # نورافکن
    
    # روشنایی فضای سبز
    TREE_UPLIGHTING = "tree_uplighting"  # نور درخت
    GARDEN_SPOT = "garden_spot"  # نورافکن باغ
    POOL_LIGHTING = "pool_lighting"  # نور استخر
    FOUNTAIN_LIGHTING = "fountain_lighting"  # نور آبنما
    
    # روشنایی تخصصی
    EMERGENCY = "emergency"  # اضطراری
    EXIT_SIGN = "exit_sign"  # تابلو خروج
    ACCENT = "accent"  # تاکیدی
    TASK = "task"  # کاری
    AMBIENT = "ambient"  # محیطی
    DECORATIVE = "decorative"  # تزئینی
    
    # سیستم‌های کنترل
    DIMMER = "dimmer"  # دیمر
    SENSOR = "sensor"  # سنسور
    SMART_CONTROL = "smart_control"  # کنترل هوشمند
    DMX_CONTROL = "dmx_control"  # کنترل DMX


class LightingZone(Enum):
    """مناطق نورپردازی"""
    INTERIOR_GENERAL = "interior_general"
    INTERIOR_TASK = "interior_task"
    INTERIOR_ACCENT = "interior_accent"
    FACADE = "facade"
    LANDSCAPE = "landscape"
    PARKING = "parking"
    PATHWAY = "pathway"
    SECURITY = "security"
    DECORATIVE = "decorative"
    EMERGENCY = "emergency"


@dataclass
class LightingFixtureInfo:
    """اطلاعات چراغ روشنایی"""
    fixture_type: LightingType
    position: Tuple[float, float, float]  # x, y, z
    layer: str
    
    # مشخصات فنی
    power_watts: Optional[float] = None  # قدرت (وات)
    lumens: Optional[float] = None  # شار نوری (لومن)
    color_temp: Optional[int] = None  # دمای رنگ (کلوین)
    beam_angle: Optional[float] = None  # زاویه تابش
    
    # ابعاد
    dimensions: Optional[Tuple[float, float, float]] = None
    mounting_height: Optional[float] = None  # ارتفاع نصب
    
    # کنترل
    is_dimmable: bool = False
    has_sensor: bool = False
    control_zone: Optional[str] = None
    
    # اطلاعات اضافی
    manufacturer: Optional[str] = None
    model: Optional[str] = None
    ip_rating: Optional[str] = None  # IP65, IP67, ...
    is_emergency: bool = False


@dataclass
class LightingCircuitInfo:
    """اطلاعات مدار روشنایی"""
    circuit_id: str
    fixtures: List[LightingFixtureInfo]
    layer: str
    
    # مشخصات برق
    voltage: int = 220  # ولتاژ
    total_power: float = 0  # جمع قدرت (وات)
    breaker_size: Optional[int] = None  # سایز فیوز (آمپر)
    
    # مسیر کابل
    cable_path: Optional[List[Tuple[float, float]]] = None
    cable_type: Optional[str] = None
    
    # کنترل
    switch_position: Optional[Tuple[float, float]] = None
    control_type: Optional[str] = None  # manual, sensor, smart, DMX


@dataclass
class LightingZoneInfo:
    """اطلاعات ناحیه نورپردازی"""
    zone_name: str
    zone_type: LightingZone
    boundary: List[Tuple[float, float]]  # مرز ناحیه
    layer: str
    
    # چراغ‌ها
    fixtures: List[LightingFixtureInfo]
    circuits: List[LightingCircuitInfo]
    
    # سطح روشنایی
    target_lux: Optional[float] = None  # سطح روشنایی هدف (لوکس)
    measured_lux: Optional[float] = None  # سطح اندازه‌گیری شده
    
    # کنترل
    control_panel: Optional[str] = None
    automation: bool = False
    daylight_harvesting: bool = False  # استفاده از نور روز


@dataclass
class LightingAnalysisResult:
    """نتیجه تحلیل نورپردازی"""
    fixtures: List[LightingFixtureInfo]
    circuits: List[LightingCircuitInfo]
    zones: List[LightingZoneInfo]
    
    # آمار کلی
    total_fixtures: int
    total_power_kw: float
    fixture_type_counts: Dict[str, int]
    zone_type_counts: Dict[str, int]
    
    # پوشش
    illuminated_area_sqm: float
    average_power_density: float  # وات بر متر مربع
    
    # انرژی
    estimated_annual_kwh: Optional[float] = None
    energy_efficiency_score: Optional[float] = None
    
    # کنترل
    has_automation: bool = False
    has_daylight_sensors: bool = False
    has_occupancy_sensors: bool = False
    
    metadata: Dict[str, any] = None


class ProfessionalLightingDetector:
    """تشخیص‌دهنده نقشه‌های روشنایی حرفه‌ای"""
    
    # کلمات کلیدی لایه‌های روشنایی
    LIGHTING_LAYER_KEYWORDS = {
        'interior': ['light', 'lighting', 'lamp', 'fixture', 'luminaire', 'روشنایی', 'چراغ', 'لامپ'],
        'facade': ['facade', 'نما', 'uplighting', 'downlighting', 'wall-wash'],
        'landscape': ['landscape', 'garden', 'فضای-سبز', 'باغ', 'محوطه'],
        'emergency': ['emergency', 'exit', 'اضطراری', 'خروج'],
        'control': ['switch', 'dimmer', 'sensor', 'control', 'کلید', 'دیمر', 'سنسور'],
    }
    
    # سمبل‌های چراغ (بلاک‌ها)
    FIXTURE_SYMBOLS = {
        'recessed': ['RECESSED', 'DOWN', 'DL', 'توکار'],
        'pendant': ['PENDANT', 'HANGING', 'PL', 'آویز'],
        'track': ['TRACK', 'RAIL', 'ریلی'],
        'chandelier': ['CHANDELIER', 'CRYSTAL', 'لوستر'],
        'wall_sconce': ['SCONCE', 'WALL-LIGHT', 'دیواری'],
        'bollard': ['BOLLARD', 'POST', 'پایه'],
        'flood': ['FLOOD', 'FLOODLIGHT', 'سیل-نور'],
        'spot': ['SPOT', 'SPOTLIGHT', 'نورافکن'],
        'exit': ['EXIT', 'EMERGENCY', 'خروج'],
    }
    
    # الگوهای توان
    POWER_PATTERNS = [
        r'(\d+)\s*W',  # 50W
        r'(\d+)\s*وات',  # 50 وات
        r'(\d+)\s*WATT',  # 50WATT
    ]
    
    def __init__(self, dxf_path: str):
        """
        Args:
            dxf_path: مسیر فایل DXF
        """
        import ezdxf
        self.doc = ezdxf.readfile(dxf_path)
        self.msp = self.doc.modelspace()
        self.dxf_path = dxf_path
    
    def detect_lighting_fixtures(self) -> List[LightingFixtureInfo]:
        """تشخیص چراغ‌های روشنایی"""
        fixtures = []
        
        # جستجوی بلاک‌ها (سمبل‌های چراغ)
        for insert in self.msp.query('INSERT'):
            block_name = insert.dxf.name.upper()
            layer = insert.dxf.layer
            position = (insert.dxf.insert.x, insert.dxf.insert.y, insert.dxf.insert.z)
            
            # شناسایی نوع چراغ
            fixture_type = self._identify_fixture_type(block_name)
            if fixture_type:
                fixture = LightingFixtureInfo(
                    fixture_type=fixture_type,
                    position=position,
                    layer=layer
                )
                
                # استخراج اطلاعات از Attributes
                if insert.has_attrib:
                    self._extract_fixture_attributes(insert, fixture)
                
                fixtures.append(fixture)
        
        # جستجوی دایره‌ها (چراغ‌های توکار)
        for circle in self.msp.query('CIRCLE'):
            layer = circle.dxf.layer
            if self._is_lighting_layer(layer):
                position = (circle.dxf.center.x, circle.dxf.center.y, 0)
                radius = circle.dxf.radius
                
                fixture = LightingFixtureInfo(
                    fixture_type=LightingType.RECESSED,
                    position=position,
                    layer=layer,
                    dimensions=(radius * 2, radius * 2, 0)
                )
                fixtures.append(fixture)
        
        return fixtures
    
    def detect_lighting_circuits(self, fixtures: List[LightingFixtureInfo]) -> List[LightingCircuitInfo]:
        """تشخیص مدارهای روشنایی"""
        circuits = []
        
        # گروه‌بندی چراغ‌ها بر اساس لایه و نزدیکی
        fixture_groups = self._group_fixtures_by_proximity(fixtures, max_distance=5000)
        
        for i, group in enumerate(fixture_groups):
            circuit_id = f"LC-{i+1:03d}"
            total_power = sum(f.power_watts or 50 for f in group)  # فرض 50W برای نامشخص
            
            circuit = LightingCircuitInfo(
                circuit_id=circuit_id,
                fixtures=group,
                layer=group[0].layer if group else "LIGHTING",
                total_power=total_power,
                breaker_size=self._calculate_breaker_size(total_power)
            )
            
            # شناسایی مسیر کابل (خطوط بین چراغ‌ها)
            circuit.cable_path = self._detect_cable_path(group)
            
            circuits.append(circuit)
        
        return circuits
    
    def detect_lighting_zones(self, fixtures: List[LightingFixtureInfo]) -> List[LightingZoneInfo]:
        """تشخیص نواحی نورپردازی"""
        zones = []
        
        # گروه‌بندی بر اساس لایه و نوع
        zone_groups = {}
        for fixture in fixtures:
            zone_key = (fixture.layer, self._determine_zone_type(fixture))
            if zone_key not in zone_groups:
                zone_groups[zone_key] = []
            zone_groups[zone_key].append(fixture)
        
        for (layer, zone_type), fixtures_in_zone in zone_groups.items():
            # محاسبه مرز ناحیه
            boundary = self._calculate_zone_boundary(fixtures_in_zone)
            
            zone = LightingZoneInfo(
                zone_name=f"{zone_type.value}_{layer}",
                zone_type=zone_type,
                boundary=boundary,
                layer=layer,
                fixtures=fixtures_in_zone,
                circuits=[]
            )
            
            # محاسبه سطح روشنایی مورد نیاز
            zone.target_lux = self._get_target_lux(zone_type)
            
            zones.append(zone)
        
        return zones
    
    def analyze(self) -> LightingAnalysisResult:
        """تحلیل کامل نقشه روشنایی"""
        # تشخیص المان‌ها
        fixtures = self.detect_lighting_fixtures()
        circuits = self.detect_lighting_circuits(fixtures)
        zones = self.detect_lighting_zones(fixtures)
        
        # محاسبه آمار
        total_power = sum(c.total_power for c in circuits)
        fixture_type_counts = {}
        for f in fixtures:
            key = f.fixture_type.value
            fixture_type_counts[key] = fixture_type_counts.get(key, 0) + 1
        
        zone_type_counts = {}
        for z in zones:
            key = z.zone_type.value
            zone_type_counts[key] = zone_type_counts.get(key, 0) + 1
        
        # محاسبه مساحت روشنایی
        illuminated_area = sum(self._calculate_polygon_area(z.boundary) for z in zones)
        
        # چک کردن اتوماسیون
        has_automation = any(f.has_sensor for f in fixtures)
        has_daylight = any(z.daylight_harvesting for z in zones)
        has_occupancy = any(f.control_zone and 'SENSOR' in f.control_zone.upper() for f in fixtures)
        
        return LightingAnalysisResult(
            fixtures=fixtures,
            circuits=circuits,
            zones=zones,
            total_fixtures=len(fixtures),
            total_power_kw=total_power / 1000,
            fixture_type_counts=fixture_type_counts,
            zone_type_counts=zone_type_counts,
            illuminated_area_sqm=illuminated_area,
            average_power_density=total_power / illuminated_area if illuminated_area > 0 else 0,
            has_automation=has_automation,
            has_daylight_sensors=has_daylight,
            has_occupancy_sensors=has_occupancy,
            metadata={
                'file_path': self.dxf_path,
                'num_circuits': len(circuits),
                'num_zones': len(zones),
                'total_power_watts': total_power,
            }
        )
    
    # Helper methods
    def _identify_fixture_type(self, block_name: str) -> Optional[LightingType]:
        """شناسایی نوع چراغ از نام بلاک"""
        block_upper = block_name.upper()
        for fixture_type, keywords in self.FIXTURE_SYMBOLS.items():
            if any(kw in block_upper for kw in keywords):
                return LightingType[fixture_type.upper()]
        return None
    
    def _is_lighting_layer(self, layer: str) -> bool:
        """بررسی لایه روشنایی"""
        layer_upper = layer.upper()
        for keywords in self.LIGHTING_LAYER_KEYWORDS.values():
            if any(kw.upper() in layer_upper for kw in keywords):
                return True
        return False
    
    def _extract_fixture_attributes(self, insert, fixture: LightingFixtureInfo):
        """استخراج اطلاعات از Attributes بلاک"""
        for attrib in insert.attribs:
            tag = attrib.dxf.tag.upper()
            value = attrib.dxf.text
            
            if 'POWER' in tag or 'WATT' in tag or 'قدرت' in tag:
                import re
                match = re.search(r'(\d+)', value)
                if match:
                    fixture.power_watts = float(match.group(1))
            
            elif 'LUMEN' in tag or 'لومن' in tag:
                import re
                match = re.search(r'(\d+)', value)
                if match:
                    fixture.lumens = float(match.group(1))
            
            elif 'TEMP' in tag or 'KELVIN' in tag or 'دما' in tag:
                import re
                match = re.search(r'(\d+)', value)
                if match:
                    fixture.color_temp = int(match.group(1))
    
    def _group_fixtures_by_proximity(self, fixtures: List[LightingFixtureInfo], max_distance: float) -> List[List[LightingFixtureInfo]]:
        """گروه‌بندی چراغ‌ها بر اساس فاصله"""
        if not fixtures:
            return []
        
        groups = []
        remaining = fixtures.copy()
        
        while remaining:
            current_group = [remaining.pop(0)]
            i = 0
            while i < len(remaining):
                fixture = remaining[i]
                # بررسی فاصله با اعضای گروه فعلی
                min_dist = min(
                    self._distance_2d(fixture.position[:2], g.position[:2])
                    for g in current_group
                )
                if min_dist <= max_distance:
                    current_group.append(remaining.pop(i))
                else:
                    i += 1
            
            groups.append(current_group)
        
        return groups
    
    def _distance_2d(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """محاسبه فاصله دوبعدی"""
        import math
        return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    
    def _calculate_breaker_size(self, total_power_watts: float) -> int:
        """محاسبه سایز فیوز"""
        current_amps = total_power_watts / 220
        # فیوز 25% بیشتر از جریان
        breaker = int(current_amps * 1.25)
        # رند کردن به سایزهای استاندارد
        standard_sizes = [6, 10, 16, 20, 25, 32, 40, 50, 63]
        for size in standard_sizes:
            if breaker <= size:
                return size
        return 63
    
    def _detect_cable_path(self, fixtures: List[LightingFixtureInfo]) -> List[Tuple[float, float]]:
        """تشخیص مسیر کابل بین چراغ‌ها"""
        if not fixtures:
            return []
        path = [f.position[:2] for f in fixtures]
        return path
    
    def _determine_zone_type(self, fixture: LightingFixtureInfo) -> LightingZone:
        """تعیین نوع ناحیه بر اساس چراغ"""
        layer_upper = fixture.layer.upper()
        
        if 'FACADE' in layer_upper or 'نما' in layer_upper:
            return LightingZone.FACADE
        elif 'LANDSCAPE' in layer_upper or 'GARDEN' in layer_upper or 'فضای-سبز' in layer_upper:
            return LightingZone.LANDSCAPE
        elif 'EMERGENCY' in layer_upper or 'EXIT' in layer_upper:
            return LightingZone.EMERGENCY
        elif 'PARKING' in layer_upper:
            return LightingZone.PARKING
        else:
            return LightingZone.INTERIOR_GENERAL
    
    def _calculate_zone_boundary(self, fixtures: List[LightingFixtureInfo]) -> List[Tuple[float, float]]:
        """محاسبه مرز ناحیه (Convex Hull ساده)"""
        if not fixtures:
            return []
        
        points = [f.position[:2] for f in fixtures]
        
        # محاسبه Bounding Box ساده
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        
        margin = 1000  # حاشیه 1 متر
        return [
            (min(xs) - margin, min(ys) - margin),
            (max(xs) + margin, min(ys) - margin),
            (max(xs) + margin, max(ys) + margin),
            (min(xs) - margin, max(ys) + margin),
        ]
    
    def _calculate_polygon_area(self, boundary: List[Tuple[float, float]]) -> float:
        """محاسبه مساحت چندضلعی (Shoelace formula)"""
        if len(boundary) < 3:
            return 0
        
        area = 0
        for i in range(len(boundary)):
            j = (i + 1) % len(boundary)
            area += boundary[i][0] * boundary[j][1]
            area -= boundary[j][0] * boundary[i][1]
        
        return abs(area) / 2
    
    def _get_target_lux(self, zone_type: LightingZone) -> float:
        """سطح روشنایی استاندارد برای هر ناحیه (لوکس)"""
        lux_standards = {
            LightingZone.INTERIOR_GENERAL: 300,
            LightingZone.INTERIOR_TASK: 500,
            LightingZone.INTERIOR_ACCENT: 150,
            LightingZone.FACADE: 100,
            LightingZone.LANDSCAPE: 50,
            LightingZone.PARKING: 75,
            LightingZone.PATHWAY: 50,
            LightingZone.SECURITY: 100,
            LightingZone.EMERGENCY: 10,
        }
        return lux_standards.get(zone_type, 200)
