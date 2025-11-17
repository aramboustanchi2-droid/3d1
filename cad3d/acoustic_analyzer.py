"""
سیستم تحلیل آکوستیک و صوتی حرفه‌ای
Professional Acoustic Analysis System

این ماژول برای تحلیل و تشخیص عناصر آکوستیک در نقشه‌های معماری طراحی شده است:
- تشخیص فضاهای آکوستیک (سالن همایش، استودیو، کلاس)
- تحلیل عایق صوتی و جاذب صدا
- محاسبه زمان پسماند (RT60)
- بررسی استانداردهای آکوستیک
- تحلیل سطوح صوتی و نویز محیط

Author: CAD 3D Converter Team
Date: 2025-11-15
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple
import ezdxf
from pathlib import Path
import json


class AcousticSpaceType(Enum):
    """انواع فضاهای آکوستیک"""
    # سالن‌ها و تالارها
    CONFERENCE_HALL = "conference_hall"           # سالن کنفرانس
    AUDITORIUM = "auditorium"                     # آمفی‌تئاتر
    LECTURE_HALL = "lecture_hall"                 # سالن سخنرانی
    CONCERT_HALL = "concert_hall"                 # سالن کنسرت
    THEATER = "theater"                           # تئاتر
    CINEMA = "cinema"                             # سینما
    
    # استودیوها
    RECORDING_STUDIO = "recording_studio"         # استودیو ضبط صدا
    BROADCAST_STUDIO = "broadcast_studio"         # استودیو پخش
    MUSIC_STUDIO = "music_studio"                 # استودیو موسیقی
    CONTROL_ROOM = "control_room"                 # اتاق کنترل
    VOCAL_BOOTH = "vocal_booth"                   # بوت آوازخوانی
    
    # آموزشی
    CLASSROOM = "classroom"                       # کلاس درس
    LANGUAGE_LAB = "language_lab"                 # آزمایشگاه زبان
    MUSIC_ROOM = "music_room"                     # اتاق موسیقی
    LIBRARY = "library"                           # کتابخانه
    
    # اداری و عمومی
    OFFICE = "office"                             # دفتر کار
    MEETING_ROOM = "meeting_room"                 # اتاق جلسه
    CALL_CENTER = "call_center"                   # مرکز تماس
    RESTAURANT = "restaurant"                     # رستوران
    
    # صنعتی
    INDUSTRIAL_SPACE = "industrial_space"         # فضای صنعتی
    MACHINE_ROOM = "machine_room"                 # موتورخانه
    
    # سلامت
    HOSPITAL_ROOM = "hospital_room"               # اتاق بیمارستان
    SURGERY_ROOM = "surgery_room"                 # اتاق عمل
    
    UNKNOWN = "unknown"                           # نامشخص


class AcousticMaterialType(Enum):
    """انواع مواد آکوستیک"""
    # جاذب‌های صوتی
    ABSORBER_FOAM = "absorber_foam"               # فوم جاذب
    ABSORBER_PANEL = "absorber_panel"             # پنل جاذب
    ABSORBER_CEILING = "absorber_ceiling"         # سقف کاذب جاذب
    ABSORBER_FABRIC = "absorber_fabric"           # پارچه جاذب
    ABSORBER_WOOD = "absorber_wood"               # چوب جاذب
    
    # عایق‌های صوتی
    INSULATION_WALL = "insulation_wall"           # عایق دیوار
    INSULATION_FLOOR = "insulation_floor"         # عایق کف
    INSULATION_CEILING = "insulation_ceiling"     # عایق سقف
    INSULATION_DOOR = "insulation_door"           # در عایق
    INSULATION_WINDOW = "insulation_window"       # پنجره عایق
    
    # پراکننده‌های صوتی
    DIFFUSER_QRD = "diffuser_qrd"                 # پراکننده QRD
    DIFFUSER_SKYLINE = "diffuser_skyline"         # پراکننده Skyline
    DIFFUSER_HEMISPHERE = "diffuser_hemisphere"   # پراکننده نیم‌کره
    
    # تله‌های باس
    BASS_TRAP_CORNER = "bass_trap_corner"         # تله باس گوشه
    BASS_TRAP_PANEL = "bass_trap_panel"           # پنل تله باس


class AcousticStandard(Enum):
    """استانداردهای آکوستیک"""
    ISO_3382 = "ISO 3382"                         # استاندارد اندازه‌گیری آکوستیک اتاق
    ANSI_S12 = "ANSI S12"                         # استاندارد آمریکایی صدا
    DIN_18041 = "DIN 18041"                       # استاندارد آلمانی آکوستیک اتاق
    WHO_GUIDELINES = "WHO Guidelines"              # راهنمای سازمان بهداشت جهانی
    BUILDING_CODE = "Building Code"                # ضوابط ملی ساختمان


@dataclass
class AcousticMaterial:
    """اطلاعات مواد آکوستیک"""
    material_type: AcousticMaterialType
    location: Tuple[float, float]                  # موقعیت (x, y)
    dimensions: Tuple[float, float, float]         # ابعاد (عرض، ارتفاع، ضخامت)
    absorption_coefficient: float = 0.0            # ضریب جذب صدا (0-1)
    nrc_rating: float = 0.0                        # Noise Reduction Coefficient
    stc_rating: int = 0                            # Sound Transmission Class
    thickness_mm: float = 0.0                      # ضخامت به میلی‌متر
    layer: str = ""                                # لایه در DXF
    coverage_area_m2: float = 0.0                  # مساحت پوشش
    properties: Dict = field(default_factory=dict)


@dataclass
class AcousticSpace:
    """اطلاعات فضای آکوستیک"""
    space_type: AcousticSpaceType
    name: str = ""
    area_m2: float = 0.0                           # مساحت کف
    volume_m3: float = 0.0                         # حجم فضا
    height_m: float = 0.0                          # ارتفاع
    boundary: List[Tuple[float, float]] = field(default_factory=list)
    
    # پارامترهای آکوستیک
    rt60_target: float = 0.0                       # زمان پسماند هدف (ثانیه)
    rt60_actual: float = 0.0                       # زمان پسماند واقعی
    background_noise_db: float = 0.0               # نویز پس‌زمینه (dB)
    max_spl_db: float = 0.0                        # حداکثر سطح صدا (dB)
    
    # مواد نصب شده
    materials: List[AcousticMaterial] = field(default_factory=list)
    
    # استانداردها
    applicable_standards: List[AcousticStandard] = field(default_factory=list)
    
    # امتیازدهی
    acoustic_score: float = 0.0                    # امتیاز کلی (0-100)
    compliance_status: str = "unknown"             # وضعیت انطباق
    
    layer: str = ""
    properties: Dict = field(default_factory=dict)


@dataclass
class NoiseSource:
    """منبع نویز"""
    source_type: str                               # نوع منبع
    location: Tuple[float, float]
    sound_power_level_db: float                    # سطح قدرت صوتی (dB)
    frequency_range: Tuple[float, float]           # محدوده فرکانسی (Hz)
    operating_hours: str = "24/7"                  # ساعات کار
    layer: str = ""


@dataclass
class AcousticAnalysisResult:
    """نتیجه تحلیل آکوستیک"""
    spaces: List[AcousticSpace]
    materials: List[AcousticMaterial]
    noise_sources: List[NoiseSource]
    
    # آمار کلی
    total_spaces: int = 0
    total_acoustic_area_m2: float = 0.0
    total_absorber_area_m2: float = 0.0
    total_insulation_area_m2: float = 0.0
    
    # کیفیت آکوستیک
    average_acoustic_score: float = 0.0
    compliant_spaces: int = 0
    non_compliant_spaces: int = 0
    
    # هشدارها
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class AcousticAnalyzer:
    """تحلیل‌گر آکوستیک حرفه‌ای"""
    
    # استانداردهای RT60 (زمان پسماند در ثانیه)
    RT60_STANDARDS = {
        AcousticSpaceType.CONFERENCE_HALL: (0.6, 1.0),
        AcousticSpaceType.AUDITORIUM: (0.8, 1.2),
        AcousticSpaceType.LECTURE_HALL: (0.6, 0.9),
        AcousticSpaceType.CONCERT_HALL: (1.5, 2.5),
        AcousticSpaceType.THEATER: (1.0, 1.5),
        AcousticSpaceType.CINEMA: (0.8, 1.2),
        AcousticSpaceType.RECORDING_STUDIO: (0.3, 0.5),
        AcousticSpaceType.BROADCAST_STUDIO: (0.25, 0.4),
        AcousticSpaceType.CONTROL_ROOM: (0.25, 0.35),
        AcousticSpaceType.CLASSROOM: (0.4, 0.7),
        AcousticSpaceType.OFFICE: (0.4, 0.6),
        AcousticSpaceType.MEETING_ROOM: (0.4, 0.6),
        AcousticSpaceType.LIBRARY: (0.5, 0.8),
        AcousticSpaceType.RESTAURANT: (0.6, 1.0),
        AcousticSpaceType.HOSPITAL_ROOM: (0.4, 0.6),
        AcousticSpaceType.SURGERY_ROOM: (0.3, 0.5),
    }
    
    # استانداردهای نویز پس‌زمینه (dB)
    BACKGROUND_NOISE_STANDARDS = {
        AcousticSpaceType.RECORDING_STUDIO: 20,
        AcousticSpaceType.BROADCAST_STUDIO: 25,
        AcousticSpaceType.CONCERT_HALL: 25,
        AcousticSpaceType.AUDITORIUM: 30,
        AcousticSpaceType.CLASSROOM: 35,
        AcousticSpaceType.OFFICE: 40,
        AcousticSpaceType.MEETING_ROOM: 35,
        AcousticSpaceType.LIBRARY: 30,
        AcousticSpaceType.HOSPITAL_ROOM: 30,
        AcousticSpaceType.SURGERY_ROOM: 25,
    }
    
    def __init__(self):
        """مقداردهی اولیه تحلیل‌گر"""
        self.spaces: List[AcousticSpace] = []
        self.materials: List[AcousticMaterial] = []
        self.noise_sources: List[NoiseSource] = []
    
    def detect_acoustic_spaces(self, doc: ezdxf.document.Drawing) -> List[AcousticSpace]:
        """
        تشخیص فضاهای آکوستیک از روی نقشه
        
        Args:
            doc: سند DXF
            
        Returns:
            لیست فضاهای آکوستیک شناسایی شده
        """
        spaces = []
        msp = doc.modelspace()
        
        # جستجو در لایه‌های مختلف
        acoustic_layers = [
            'ACOUSTIC', 'ACOUSTICS', 'SOUND',
            'AUDITORIUM', 'STUDIO', 'HALL',
            'CONFERENCE', 'THEATER', 'CINEMA',
            'CLASSROOM', 'LECTURE'
        ]
        
        for entity in msp:
            layer_name = entity.dxf.layer.upper()
            
            # بررسی لایه
            if not any(al in layer_name for al in acoustic_layers):
                continue
            
            # تشخیص فضا از روی LWPOLYLINE یا POLYLINE
            if entity.dxftype() in ['LWPOLYLINE', 'POLYLINE']:
                if not entity.is_closed:
                    continue
                
                # استخراج مرز
                points = []
                if entity.dxftype() == 'LWPOLYLINE':
                    points = [(p[0], p[1]) for p in entity.get_points()]
                
                if len(points) < 3:
                    continue
                
                # تشخیص نوع فضا
                space_type = self._identify_space_type(layer_name, entity)
                
                # محاسبه مساحت
                area = self._calculate_polygon_area(points)
                
                # ایجاد فضای آکوستیک
                space = AcousticSpace(
                    space_type=space_type,
                    name=layer_name,
                    area_m2=area / 1000000.0,  # تبدیل به متر مربع
                    boundary=points,
                    layer=entity.dxf.layer
                )
                
                # تخمین حجم (فرض: ارتفاع 3 متر)
                space.height_m = 3.0
                space.volume_m3 = space.area_m2 * space.height_m
                
                # تنظیم استانداردها
                space.applicable_standards = [
                    AcousticStandard.ISO_3382,
                    AcousticStandard.BUILDING_CODE
                ]
                
                # تنظیم RT60 هدف
                if space_type in self.RT60_STANDARDS:
                    rt60_range = self.RT60_STANDARDS[space_type]
                    space.rt60_target = (rt60_range[0] + rt60_range[1]) / 2
                
                # تنظیم نویز پس‌زمینه
                if space_type in self.BACKGROUND_NOISE_STANDARDS:
                    space.background_noise_db = self.BACKGROUND_NOISE_STANDARDS[space_type]
                
                spaces.append(space)
        
        self.spaces = spaces
        return spaces
    
    def detect_acoustic_materials(self, doc: ezdxf.document.Drawing) -> List[AcousticMaterial]:
        """
        تشخیص مواد آکوستیک
        
        Args:
            doc: سند DXF
            
        Returns:
            لیست مواد آکوستیک
        """
        materials = []
        msp = doc.modelspace()
        
        # لایه‌های مواد آکوستیک
        material_keywords = {
            'ABSORBER': AcousticMaterialType.ABSORBER_PANEL,
            'FOAM': AcousticMaterialType.ABSORBER_FOAM,
            'INSULATION': AcousticMaterialType.INSULATION_WALL,
            'DIFFUSER': AcousticMaterialType.DIFFUSER_QRD,
            'BASS_TRAP': AcousticMaterialType.BASS_TRAP_CORNER,
            'ACOUSTIC_CEILING': AcousticMaterialType.ABSORBER_CEILING,
            'ACOUSTIC_PANEL': AcousticMaterialType.ABSORBER_PANEL,
            'SOUND_INSULATION': AcousticMaterialType.INSULATION_WALL,
        }
        
        for entity in msp:
            layer_name = entity.dxf.layer.upper()
            
            # تشخیص نوع ماده
            material_type = None
            for keyword, mat_type in material_keywords.items():
                if keyword in layer_name:
                    material_type = mat_type
                    break
            
            if material_type is None:
                continue
            
            # استخراج اطلاعات هندسی
            if entity.dxftype() == 'INSERT':  # بلوک
                location = (entity.dxf.insert.x, entity.dxf.insert.y)
                dimensions = (1000.0, 1000.0, 50.0)  # پیش‌فرض
                
            elif entity.dxftype() in ['LWPOLYLINE', 'POLYLINE']:
                points = []
                if entity.dxftype() == 'LWPOLYLINE':
                    points = [(p[0], p[1]) for p in entity.get_points()]
                
                if len(points) < 2:
                    continue
                
                location = points[0]
                
                # محاسبه ابعاد
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                width = max(xs) - min(xs)
                height = max(ys) - min(ys)
                dimensions = (width, height, 50.0)
                
            else:
                continue
            
            # ضرایب استاندارد
            absorption_coeff = self._get_absorption_coefficient(material_type)
            nrc = self._get_nrc_rating(material_type)
            stc = self._get_stc_rating(material_type)
            
            # محاسبه مساحت پوشش
            coverage_area = (dimensions[0] * dimensions[1]) / 1000000.0
            
            material = AcousticMaterial(
                material_type=material_type,
                location=location,
                dimensions=dimensions,
                absorption_coefficient=absorption_coeff,
                nrc_rating=nrc,
                stc_rating=stc,
                thickness_mm=dimensions[2],
                layer=entity.dxf.layer,
                coverage_area_m2=coverage_area
            )
            
            materials.append(material)
        
        self.materials = materials
        return materials
    
    def detect_noise_sources(self, doc: ezdxf.document.Drawing) -> List[NoiseSource]:
        """تشخیص منابع نویز"""
        noise_sources = []
        msp = doc.modelspace()
        
        noise_keywords = {
            'HVAC': (70, (100, 2000)),
            'MECHANICAL': (75, (50, 5000)),
            'ELEVATOR': (65, (125, 1000)),
            'GENERATOR': (80, (63, 8000)),
            'TRANSFORMER': (70, (100, 1000)),
            'FAN': (75, (200, 2000)),
            'PUMP': (80, (100, 2000)),
        }
        
        for entity in msp:
            layer_name = entity.dxf.layer.upper()
            
            for keyword, (spl, freq_range) in noise_keywords.items():
                if keyword in layer_name:
                    if entity.dxftype() == 'INSERT':
                        location = (entity.dxf.insert.x, entity.dxf.insert.y)
                    elif entity.dxftype() == 'CIRCLE':
                        location = (entity.dxf.center.x, entity.dxf.center.y)
                    else:
                        continue
                    
                    source = NoiseSource(
                        source_type=keyword,
                        location=location,
                        sound_power_level_db=spl,
                        frequency_range=freq_range,
                        layer=entity.dxf.layer
                    )
                    noise_sources.append(source)
                    break
        
        self.noise_sources = noise_sources
        return noise_sources
    
    def calculate_rt60(self, space: AcousticSpace) -> float:
        """
        محاسبه زمان پسماند (RT60) با فرمول Sabine
        
        RT60 = 0.161 × V / A
        V: حجم فضا (m³)
        A: مساحت جذب معادل (m²)
        """
        if space.volume_m3 <= 0:
            return 0.0
        
        # مجموع مساحت جذب
        total_absorption = 0.0
        
        for material in space.materials:
            total_absorption += material.coverage_area_m2 * material.absorption_coefficient
        
        # اگر مواد جاذب نداریم، فرض کنیم ضریب جذب کم است
        if total_absorption == 0:
            total_absorption = space.area_m2 * 0.1  # فرض: 10% جذب
        
        # فرمول Sabine
        rt60 = 0.161 * space.volume_m3 / total_absorption
        
        return rt60
    
    def calculate_acoustic_score(self, space: AcousticSpace) -> float:
        """محاسبه امتیاز آکوستیک (0-100)"""
        score = 100.0
        
        # بررسی RT60
        if space.space_type in self.RT60_STANDARDS:
            rt60_range = self.RT60_STANDARDS[space.space_type]
            rt60_actual = space.rt60_actual
            
            if rt60_actual < rt60_range[0]:
                score -= 20 * (rt60_range[0] - rt60_actual)
            elif rt60_actual > rt60_range[1]:
                score -= 20 * (rt60_actual - rt60_range[1])
        
        # بررسی نویز پس‌زمینه
        if space.space_type in self.BACKGROUND_NOISE_STANDARDS:
            max_noise = self.BACKGROUND_NOISE_STANDARDS[space.space_type]
            if space.background_noise_db > max_noise:
                score -= 2 * (space.background_noise_db - max_noise)
        
        # بررسی مواد جاذب
        if len(space.materials) == 0:
            score -= 30
        
        return max(0.0, min(100.0, score))
    
    def analyze(self, dxf_path: str) -> AcousticAnalysisResult:
        """
        تحلیل کامل آکوستیک نقشه
        
        Args:
            dxf_path: مسیر فایل DXF
            
        Returns:
            نتیجه تحلیل آکوستیک
        """
        doc = ezdxf.readfile(dxf_path)
        
        # تشخیص عناصر
        spaces = self.detect_acoustic_spaces(doc)
        materials = self.detect_acoustic_materials(doc)
        noise_sources = self.detect_noise_sources(doc)
        
        # اختصاص مواد به فضاها
        for space in spaces:
            space.materials = [
                m for m in materials
                if self._point_in_polygon(m.location, space.boundary)
            ]
            
            # محاسبه RT60
            space.rt60_actual = self.calculate_rt60(space)
            
            # محاسبه امتیاز
            space.acoustic_score = self.calculate_acoustic_score(space)
            
            # وضعیت انطباق
            if space.acoustic_score >= 80:
                space.compliance_status = "excellent"
            elif space.acoustic_score >= 60:
                space.compliance_status = "good"
            elif space.acoustic_score >= 40:
                space.compliance_status = "fair"
            else:
                space.compliance_status = "poor"
        
        # آمار کلی
        total_acoustic_area = sum(s.area_m2 for s in spaces)
        total_absorber_area = sum(
            m.coverage_area_m2 for m in materials
            if 'ABSORBER' in m.material_type.name
        )
        total_insulation_area = sum(
            m.coverage_area_m2 for m in materials
            if 'INSULATION' in m.material_type.name
        )
        
        avg_score = sum(s.acoustic_score for s in spaces) / len(spaces) if spaces else 0.0
        compliant = sum(1 for s in spaces if s.acoustic_score >= 60)
        non_compliant = len(spaces) - compliant
        
        # هشدارها و توصیه‌ها
        warnings = []
        recommendations = []
        
        for space in spaces:
            if space.acoustic_score < 60:
                warnings.append(f"فضای {space.name} نیاز به بهبود دارد (امتیاز: {space.acoustic_score:.1f})")
            
            if space.rt60_actual > 0 and space.space_type in self.RT60_STANDARDS:
                rt60_range = self.RT60_STANDARDS[space.space_type]
                if space.rt60_actual > rt60_range[1]:
                    recommendations.append(
                        f"{space.name}: افزودن جاذب صوتی برای کاهش RT60 از {space.rt60_actual:.2f}s به {rt60_range[1]:.2f}s"
                    )
                elif space.rt60_actual < rt60_range[0]:
                    recommendations.append(
                        f"{space.name}: کاهش جاذب صوتی برای افزایش RT60 از {space.rt60_actual:.2f}s به {rt60_range[0]:.2f}s"
                    )
        
        result = AcousticAnalysisResult(
            spaces=spaces,
            materials=materials,
            noise_sources=noise_sources,
            total_spaces=len(spaces),
            total_acoustic_area_m2=total_acoustic_area,
            total_absorber_area_m2=total_absorber_area,
            total_insulation_area_m2=total_insulation_area,
            average_acoustic_score=avg_score,
            compliant_spaces=compliant,
            non_compliant_spaces=non_compliant,
            warnings=warnings,
            recommendations=recommendations
        )
        
        return result
    
    def export_to_json(self, result: AcousticAnalysisResult, output_path: str):
        """خروجی JSON"""
        data = {
            'summary': {
                'total_spaces': result.total_spaces,
                'total_acoustic_area_m2': result.total_acoustic_area_m2,
                'total_absorber_area_m2': result.total_absorber_area_m2,
                'total_insulation_area_m2': result.total_insulation_area_m2,
                'average_acoustic_score': result.average_acoustic_score,
                'compliant_spaces': result.compliant_spaces,
                'non_compliant_spaces': result.non_compliant_spaces,
            },
            'spaces': [
                {
                    'type': s.space_type.value,
                    'name': s.name,
                    'area_m2': s.area_m2,
                    'volume_m3': s.volume_m3,
                    'rt60_target': s.rt60_target,
                    'rt60_actual': s.rt60_actual,
                    'background_noise_db': s.background_noise_db,
                    'acoustic_score': s.acoustic_score,
                    'compliance_status': s.compliance_status,
                    'materials_count': len(s.materials),
                }
                for s in result.spaces
            ],
            'materials': [
                {
                    'type': m.material_type.value,
                    'location': m.location,
                    'dimensions': m.dimensions,
                    'absorption_coefficient': m.absorption_coefficient,
                    'nrc_rating': m.nrc_rating,
                    'stc_rating': m.stc_rating,
                    'coverage_area_m2': m.coverage_area_m2,
                }
                for m in result.materials
            ],
            'noise_sources': [
                {
                    'type': n.source_type,
                    'location': n.location,
                    'sound_power_level_db': n.sound_power_level_db,
                    'frequency_range': n.frequency_range,
                }
                for n in result.noise_sources
            ],
            'warnings': result.warnings,
            'recommendations': result.recommendations,
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    # متدهای کمکی
    
    def _identify_space_type(self, layer_name: str, entity) -> AcousticSpaceType:
        """تشخیص نوع فضا از روی نام لایه"""
        layer_upper = layer_name.upper()
        
        if 'CONCERT' in layer_upper or 'MUSIC_HALL' in layer_upper:
            return AcousticSpaceType.CONCERT_HALL
        elif 'AUDITORIUM' in layer_upper or 'AMPHITHEATER' in layer_upper:
            return AcousticSpaceType.AUDITORIUM
        elif 'CONFERENCE' in layer_upper:
            return AcousticSpaceType.CONFERENCE_HALL
        elif 'LECTURE' in layer_upper:
            return AcousticSpaceType.LECTURE_HALL
        elif 'THEATER' in layer_upper or 'THEATRE' in layer_upper:
            return AcousticSpaceType.THEATER
        elif 'CINEMA' in layer_upper or 'MOVIE' in layer_upper:
            return AcousticSpaceType.CINEMA
        elif 'RECORDING' in layer_upper or 'STUDIO' in layer_upper:
            return AcousticSpaceType.RECORDING_STUDIO
        elif 'BROADCAST' in layer_upper:
            return AcousticSpaceType.BROADCAST_STUDIO
        elif 'CONTROL' in layer_upper:
            return AcousticSpaceType.CONTROL_ROOM
        elif 'CLASSROOM' in layer_upper or 'CLASS' in layer_upper:
            return AcousticSpaceType.CLASSROOM
        elif 'LIBRARY' in layer_upper:
            return AcousticSpaceType.LIBRARY
        elif 'OFFICE' in layer_upper:
            return AcousticSpaceType.OFFICE
        elif 'MEETING' in layer_upper:
            return AcousticSpaceType.MEETING_ROOM
        else:
            return AcousticSpaceType.UNKNOWN
    
    def _calculate_polygon_area(self, points: List[Tuple[float, float]]) -> float:
        """محاسبه مساحت چندضلعی با فرمول Shoelace"""
        if len(points) < 3:
            return 0.0
        
        area = 0.0
        for i in range(len(points)):
            j = (i + 1) % len(points)
            area += points[i][0] * points[j][1]
            area -= points[j][0] * points[i][1]
        
        return abs(area) / 2.0
    
    def _point_in_polygon(self, point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
        """بررسی قرار گرفتن نقطه در چندضلعی (Ray Casting)"""
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def _get_absorption_coefficient(self, material_type: AcousticMaterialType) -> float:
        """ضریب جذب صدا (0-1)"""
        coefficients = {
            AcousticMaterialType.ABSORBER_FOAM: 0.85,
            AcousticMaterialType.ABSORBER_PANEL: 0.75,
            AcousticMaterialType.ABSORBER_CEILING: 0.70,
            AcousticMaterialType.ABSORBER_FABRIC: 0.60,
            AcousticMaterialType.ABSORBER_WOOD: 0.40,
            AcousticMaterialType.INSULATION_WALL: 0.50,
            AcousticMaterialType.INSULATION_FLOOR: 0.30,
            AcousticMaterialType.INSULATION_CEILING: 0.50,
            AcousticMaterialType.DIFFUSER_QRD: 0.15,
            AcousticMaterialType.DIFFUSER_SKYLINE: 0.20,
            AcousticMaterialType.BASS_TRAP_CORNER: 0.80,
            AcousticMaterialType.BASS_TRAP_PANEL: 0.75,
        }
        return coefficients.get(material_type, 0.30)
    
    def _get_nrc_rating(self, material_type: AcousticMaterialType) -> float:
        """NRC Rating (Noise Reduction Coefficient)"""
        nrc_values = {
            AcousticMaterialType.ABSORBER_FOAM: 0.90,
            AcousticMaterialType.ABSORBER_PANEL: 0.80,
            AcousticMaterialType.ABSORBER_CEILING: 0.70,
            AcousticMaterialType.ABSORBER_FABRIC: 0.65,
            AcousticMaterialType.BASS_TRAP_CORNER: 0.85,
        }
        return nrc_values.get(material_type, 0.50)
    
    def _get_stc_rating(self, material_type: AcousticMaterialType) -> int:
        """STC Rating (Sound Transmission Class)"""
        stc_values = {
            AcousticMaterialType.INSULATION_WALL: 50,
            AcousticMaterialType.INSULATION_FLOOR: 55,
            AcousticMaterialType.INSULATION_CEILING: 50,
            AcousticMaterialType.INSULATION_DOOR: 45,
            AcousticMaterialType.INSULATION_WINDOW: 40,
        }
        return stc_values.get(material_type, 30)


def create_acoustic_analyzer() -> AcousticAnalyzer:
    """ایجاد تحلیل‌گر آکوستیک"""
    return AcousticAnalyzer()


if __name__ == "__main__":
    # تست سریع
    print("🎵 Acoustic Analysis System")
    print("=" * 60)
    print(f"✅ {len(AcousticSpaceType)} space types")
    print(f"✅ {len(AcousticMaterialType)} material types")
    print(f"✅ {len(AcousticStandard)} acoustic standards")
    print("\n📊 RT60 Standards:")
    analyzer = AcousticAnalyzer()
    for space_type, (min_rt, max_rt) in list(analyzer.RT60_STANDARDS.items())[:5]:
        print(f"   - {space_type.value}: {min_rt}-{max_rt}s")
    print("\n✨ Ready for acoustic analysis!")
