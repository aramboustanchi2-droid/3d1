"""
Architectural Drawing Analyzer - تحلیلگر نقشه‌های معماری
این ماژول نقشه‌های معماری را تحلیل و درک می‌کند:
- پلان‌ها (Plans)
- نماها (Elevations)
- برش‌ها (Sections)
- ابعاد و اندازه‌گیری‌ها
"""
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import math

import ezdxf
from ezdxf.entities import LWPolyline, Line, Text, MText, Dimension


class DrawingType(Enum):
    """نوع نقشه معماری و سازه‌ای"""
    PLAN = "plan"  # پلان
    ELEVATION = "elevation"  # نما
    SECTION = "section"  # برش
    DETAIL = "detail"  # جزئیات
    SITE_PLAN = "site_plan"  # پلان سایت
    # نقشه‌های سازه‌ای / Structural Drawings
    FOUNDATION = "foundation"  # نقشه فونداسیون
    STRUCTURAL_PLAN = "structural_plan"  # پلان سازه (ستون و تیر)
    BEAM_LAYOUT = "beam_layout"  # چیدمان تیرها
    COLUMN_LAYOUT = "column_layout"  # چیدمان ستون‌ها
    SLAB = "slab"  # نقشه دال و سقف
    STRUCTURAL_SECTION = "structural_section"  # برش سازه
    STRUCTURAL_DETAIL = "structural_detail"  # جزئیات سازه
    # نقشه‌های تأسیسات مکانیکی و برقی / MEP Drawings
    PLUMBING = "plumbing"  # نقشه لوله‌کشی آب و فاضلاب
    HVAC = "hvac"  # نقشه تهویه، سرمایش و گرمایش
    ELECTRICAL = "electrical"  # نقشه برق
    LIGHTING = "lighting"  # نقشه روشنایی
    FIRE_PROTECTION = "fire_protection"  # سیستم اعلام و اطفاء حریق
    MEP_COORDINATION = "mep_coordination"  # هماهنگی تأسیسات
    CONSTRUCTION_DETAIL = "construction_detail"  # جزئیات اجرایی
    UNKNOWN = "unknown"


class SpaceType(Enum):
    """انواع فضاها"""
    BEDROOM = "bedroom"  # اتاق خواب
    LIVING_ROOM = "living_room"  # اتاق نشیمن
    KITCHEN = "kitchen"  # آشپزخانه
    BATHROOM = "bathroom"  # سرویس بهداشتی
    CORRIDOR = "corridor"  # راهرو
    BALCONY = "balcony"  # بالکن
    STORAGE = "storage"  # انباری
    PARKING = "parking"  # پارکینگ
    OFFICE = "office"  # دفتر کار
    LOBBY = "lobby"  # لابی
    STAIRCASE = "staircase"  # راه پله
    ELEVATOR = "elevator"  # آسانسور
    UNKNOWN = "unknown"


class StructuralElementType(Enum):
    """انواع عناصر سازه‌ای"""
    COLUMN = "column"  # ستون
    BEAM = "beam"  # تیر
    SLAB = "slab"  # دال
    WALL_BEARING = "wall_bearing"  # دیوار باربر
    FOUNDATION = "foundation"  # فونداسیون
    FOOTING = "footing"  # پی
    PILE = "pile"  # شمع
    SHEAR_WALL = "shear_wall"  # دیوار برشی
    BRACE = "brace"  # مهاربند
    TRUSS = "truss"  # خرپا
    REBAR = "rebar"  # آرماتور


class DetailType(Enum):
    """انواع جزئیات اجرایی"""
    # جزئیات اتصالات / Connection Details
    BEAM_COLUMN_CONNECTION = "beam_column_connection"  # اتصال تیر به ستون
    WALL_FOUNDATION_CONNECTION = "wall_foundation_connection"  # اتصال دیوار به فونداسیون
    SLAB_WALL_CONNECTION = "slab_wall_connection"  # اتصال دال به دیوار
    ROOF_WALL_CONNECTION = "roof_wall_connection"  # اتصال سقف به دیوار
    EXPANSION_JOINT = "expansion_joint"  # درز انبساط
    # جزئیات درب و پنجره / Door and Window Details
    DOOR_DETAIL = "door_detail"  # جزئیات درب
    WINDOW_DETAIL = "window_detail"  # جزئیات پنجره
    DOOR_FRAME = "door_frame"  # چارچوب درب
    WINDOW_FRAME = "window_frame"  # چارچوب پنجره
    THRESHOLD = "threshold"  # آستانه
    # جزئیات نما / Facade Details
    FACADE_PANEL = "facade_panel"  # پانل نما
    CLADDING = "cladding"  # روکش نما
    CURTAIN_WALL = "curtain_wall"  # دیوار پرده‌ای
    STONE_CLADDING = "stone_cladding"  # نمای سنگ
    METAL_PANEL = "metal_panel"  # پانل فلزی
    # جزئیات کف و سقف / Floor and Ceiling Details
    FLOOR_FINISH = "floor_finish"  # پوشش کف
    CEILING_DETAIL = "ceiling_detail"  # جزئیات سقف
    FALSE_CEILING = "false_ceiling"  # سقف کاذب
    FLOORING_SECTION = "flooring_section"  # مقطع کف‌سازی
    # جزئیات عایق و آب‌بندی / Insulation and Waterproofing
    WATERPROOFING = "waterproofing"  # آب‌بندی
    INSULATION = "insulation"  # عایق حرارتی/صوتی
    VAPOR_BARRIER = "vapor_barrier"  # عایق رطوبتی
    DAMP_PROOF_COURSE = "damp_proof_course"  # لایه ضد رطوبت
    # جزئیات دیوار / Wall Details
    WALL_SECTION = "wall_section"  # مقطع دیوار
    PARTITION_WALL = "partition_wall"  # دیوار پارتیشن
    RETAINING_WALL = "retaining_wall"  # دیوار حائل
    # جزئیات پله و نرده / Stair and Railing Details
    STAIR_DETAIL = "stair_detail"  # جزئیات پله
    RAILING_DETAIL = "railing_detail"  # جزئیات نرده
    HANDRAIL = "handrail"  # دست‌انداز
    # جزئیات سقف و بام / Roof Details
    ROOF_SECTION = "roof_section"  # مقطع سقف
    ROOF_EDGE = "roof_edge"  # لبه سقف
    GUTTER = "gutter"  # ناودان
    FLASHING = "flashing"  # برگه‌های فلزی آب‌بندی
    SKYLIGHT = "skylight"  # نورگیر
    UNKNOWN = "unknown"


class MEPElementType(Enum):
    """انواع عناصر تأسیسات مکانیکی، برقی و لوله‌کشی"""
    # لوله‌کشی و آب / Plumbing
    WATER_PIPE = "water_pipe"  # لوله آب
    DRAIN_PIPE = "drain_pipe"  # لوله فاضلاب
    VENT_PIPE = "vent_pipe"  # لوله تهویه
    WATER_HEATER = "water_heater"  # آبگرمکن
    PUMP = "pump"  # پمپ
    VALVE = "valve"  # شیر
    FIXTURE = "fixture"  # شیرآلات (سینک، توالت، دوش)
    TANK = "tank"  # مخزن
    # HVAC (تهویه، سرمایش، گرمایش)
    DUCT = "duct"  # کانال هوا
    DIFFUSER = "diffuser"  # دریچه هوا
    GRILLE = "grille"  # گریل
    AIR_HANDLER = "air_handler"  # هواساز
    FAN = "fan"  # فن
    CHILLER = "chiller"  # چیلر
    BOILER = "boiler"  # بویلر
    COOLING_TOWER = "cooling_tower"  # برج خنک‌کننده
    THERMOSTAT = "thermostat"  # ترموستات
    FCU = "fcu"  # فن کویل
    VAV = "vav"  # VAV Box
    # برق / Electrical
    PANEL = "panel"  # تابلو برق
    OUTLET = "outlet"  # پریز
    SWITCH = "switch"  # کلید
    LIGHT_FIXTURE = "light_fixture"  # چراغ
    CABLE = "cable"  # کابل
    CONDUIT = "conduit"  # کاندوئیت
    TRANSFORMER = "transformer"  # ترانسفورماتور
    GENERATOR = "generator"  # ژنراتور
    UPS = "ups"  # UPS
    # اعلام و اطفاء حریق / Fire Protection
    SPRINKLER = "sprinkler"  # اسپرینکلر
    FIRE_ALARM = "fire_alarm"  # اعلام حریق
    SMOKE_DETECTOR = "smoke_detector"  # دتکتور دود
    FIRE_EXTINGUISHER = "fire_extinguisher"  # کپسول آتش‌نشانی
    FIRE_HOSE = "fire_hose"  # شیلنگ آتش‌نشانی
    UNKNOWN = "unknown"


class MaterialType(Enum):
    """انواع مصالح ساختمانی"""
    # مصالح اصلی / Primary Materials
    CONCRETE = "concrete"  # بتن
    STEEL = "steel"  # فولاد
    BRICK = "brick"  # آجر
    BLOCK = "block"  # بلوک
    STONE = "stone"  # سنگ
    WOOD = "wood"  # چوب
    # مصالح نما / Facade Materials
    GLASS = "glass"  # شیشه
    ALUMINUM = "aluminum"  # آلومینیوم
    COMPOSITE_PANEL = "composite_panel"  # پانل کامپوزیت
    CERAMIC = "ceramic"  # سرامیک
    # مصالح عایق / Insulation Materials
    FOAM_INSULATION = "foam_insulation"  # عایق فوم
    MINERAL_WOOL = "mineral_wool"  # پشم سنگ
    POLYSTYRENE = "polystyrene"  # پلی‌استایرن
    # مصالح آب‌بندی / Waterproofing Materials
    BITUMEN = "bitumen"  # قیر
    MEMBRANE = "membrane"  # غشا
    SEALANT = "sealant"  # درزگیر
    # سایر / Other
    PLASTER = "plaster"  # گچ
    MORTAR = "mortar"  # ملات
    PAINT = "paint"  # رنگ
    UNKNOWN = "unknown"


class SiteElementType(Enum):
    """انواع عناصر نقشه سایت و موقعیت"""
    # ساختمان‌ها / Buildings
    MAIN_BUILDING = "main_building"  # ساختمان اصلی
    ADJACENT_BUILDING = "adjacent_building"  # ساختمان مجاور
    EXISTING_BUILDING = "existing_building"  # ساختمان موجود
    # مرزها و محدوده‌ها / Boundaries
    PROPERTY_LINE = "property_line"  # خط مرز ملک
    SETBACK_LINE = "setback_line"  # خط عقب‌نشینی
    BUILDING_LINE = "building_line"  # خط ساختمان
    # دسترسی و حمل‌ونقل / Access & Transportation
    ROAD = "road"  # جاده
    DRIVEWAY = "driveway"  # راه ورودی
    PARKING_LOT = "parking_lot"  # پارکینگ
    PARKING_SPACE = "parking_space"  # جای پارک
    WALKWAY = "walkway"  # پیاده‌رو
    SIDEWALK = "sidewalk"  # کف پیاده
    ENTRY_GATE = "entry_gate"  # درب ورودی
    # فضای سبز / Landscape
    TREE = "tree"  # درخت
    SHRUB = "shrub"  # درختچه
    GRASS_AREA = "grass_area"  # چمن
    GARDEN = "garden"  # باغچه
    PLANTER = "planter"  # گلدان/جاگیاه
    HEDGE = "hedge"  # پرچین
    # عناصر سایت / Site Features
    FENCE = "fence"  # حصار
    WALL = "wall"  # دیوار محوطه
    RETAINING_WALL = "retaining_wall"  # دیوار حائل
    POOL = "pool"  # استخر
    FOUNTAIN = "fountain"  # فواره
    PATIO = "patio"  # پاسیو/تراس
    DECK = "deck"  # عرشه
    # تاسیسات سایت / Site Utilities
    UTILITY_LINE = "utility_line"  # خطوط تاسیسات
    MANHOLE = "manhole"  # منهول
    CATCH_BASIN = "catch_basin"  # منبع جمع‌آوری
    LIGHT_POLE = "light_pole"  # تیر چراغ
    SIGNAGE = "signage"  # تابلو
    # توپوگرافی / Topography
    CONTOUR_LINE = "contour_line"  # کانتور ارتفاعی
    SPOT_ELEVATION = "spot_elevation"  # نقطه ارتفاعی
    SLOPE = "slope"  # شیب
    # جهت‌یابی / Orientation
    NORTH_ARROW = "north_arrow"  # جهت شمال
    SCALE_BAR = "scale_bar"  # مقیاس نقشه
    UNKNOWN = "unknown"


class CivilElementType(str, Enum):
    """انواع عناصر مهندسی سایت و محوطه"""
    # شیب‌بندی و کانتور / Grading & Contouring
    CONTOUR_MAJOR = "contour_major"  # کانتور اصلی
    CONTOUR_MINOR = "contour_minor"  # کانتور فرعی
    SPOT_ELEVATION_EXISTING = "spot_elevation_existing"  # ارتفاع موجود
    SPOT_ELEVATION_PROPOSED = "spot_elevation_proposed"  # ارتفاع طرح
    SLOPE_INDICATOR = "slope_indicator"  # نشانگر شیب
    CUT_AREA = "cut_area"  # منطقه برش زمین
    FILL_AREA = "fill_area"  # منطقه خاکریزی
    # زهکشی و آبرو / Drainage Systems
    DRAINAGE_PIPE = "drainage_pipe"  # لوله زهکشی
    CATCH_BASIN_CIVIL = "catch_basin"  # حوضچه جمع‌آوری
    INLET = "inlet"  # دریچه ورودی
    OUTLET = "outlet"  # دریچه خروجی
    SWALE = "swale"  # آبرو طبیعی
    CULVERT = "culvert"  # گذرگاه آب
    DITCH = "ditch"  # کانال آب
    RETENTION_POND = "retention_pond"  # حوضچه نگهداری آب
    DETENTION_BASIN = "detention_basin"  # حوضچه تاخیری
    # فاضلاب سطحی / Surface Water
    SURFACE_FLOW_ARROW = "surface_flow_arrow"  # جهت جریان سطحی
    WATERSHED_BOUNDARY = "watershed_boundary"  # مرز حوضه آبریز
    DRAINAGE_AREA = "drainage_area"  # منطقه زهکشی
    # مسیرها و دسترسی / Access Routes
    ROAD_CENTERLINE = "road_centerline"  # محور جاده
    ROAD_EDGE = "road_edge"  # لبه جاده
    CURB = "curb"  # جدول
    GUTTER = "gutter"  # آبروی کنار جدول
    SIDEWALK_CIVIL = "sidewalk"  # پیاده‌رو
    DRIVEWAY_APRON = "driveway_apron"  # رمپ ورودی
    SPEED_BUMP = "speed_bump"  # سرعت‌گیر
    # محوطه‌سازی / Site Work
    RETAINING_WALL_CIVIL = "retaining_wall"  # دیوار حائل
    EMBANKMENT = "embankment"  # خاکریز
    EXCAVATION_LIMIT = "excavation_limit"  # محدوده گودبرداری
    STOCKPILE_AREA = "stockpile_area"  # محل انباشت خاک
    EROSION_CONTROL = "erosion_control"  # کنترل فرسایش
    SILT_FENCE = "silt_fence"  # حصار رسوب‌گیر
    # خطوط تاسیسات / Utility Lines
    WATER_LINE = "water_line"  # خط آب
    SEWER_LINE = "sewer_line"  # خط فاضلاب
    STORM_DRAIN = "storm_drain"  # زهکش طوفان
    GAS_LINE = "gas_line"  # خط گاز
    ELECTRIC_LINE = "electric_line"  # خط برق
    TELECOM_LINE = "telecom_line"  # خط مخابرات
    # نشانه‌ها و علائم / Markers & Symbols
    BENCHMARK = "benchmark"  # نقطه مبنا ارتفاعی
    SURVEY_MARKER = "survey_marker"  # نشانه نقشه‌برداری
    SECTION_CUT_LINE = "section_cut_line"  # خط برش
    GRID_LINE = "grid_line"  # خط شبکه
    UNKNOWN = "unknown"


class SafetySecurityElementType(str, Enum):
    """انواع عناصر ایمنی و امنیت"""
    # سیستم اعلام حریق / Fire Alarm System
    FIRE_ALARM_PANEL = "fire_alarm_panel"  # پنل اعلام حریق
    FIRE_ALARM_DETECTOR = "fire_alarm_detector"  # آشکارساز دود/حرارت
    FIRE_ALARM_MANUAL_STATION = "fire_alarm_manual_station"  # دکمه دستی اعلام حریق
    FIRE_ALARM_HORN = "fire_alarm_horn"  # آژیر حریق
    FIRE_ALARM_STROBE = "fire_alarm_strobe"  # چراغ چشمک‌زن
    # سیستم اطفاء حریق / Fire Suppression
    SPRINKLER_HEAD = "sprinkler_head"  # سری اسپرینکلر
    FIRE_EXTINGUISHER = "fire_extinguisher"  # کپسول آتش‌نشانی
    FIRE_HOSE_CABINET = "fire_hose_cabinet"  # جعبه شیلنگ آتش‌نشانی
    FIRE_HYDRANT = "fire_hydrant"  # هیدرانت
    FM200_NOZZLE = "fm200_nozzle"  # نازل FM200
    # خروج اضطراری / Emergency Exit
    EXIT_DOOR = "exit_door"  # درب خروج اضطراری
    EXIT_SIGN = "exit_sign"  # تابلو خروج
    EMERGENCY_LIGHT = "emergency_light"  # چراغ اضطراری
    EVACUATION_ROUTE = "evacuation_route"  # مسیر تخلیه
    ASSEMBLY_POINT = "assembly_point"  # نقطه تجمع
    # سیستم امنیتی / Security System
    CCTV_CAMERA = "cctv_camera"  # دوربین مداربسته
    CCTV_DVR = "cctv_dvr"  # دستگاه ضبط
    ACCESS_CONTROL_READER = "access_control_reader"  # کارتخوان
    ACCESS_CONTROL_PANEL = "access_control_panel"  # پنل کنترل تردد
    BIOMETRIC_SCANNER = "biometric_scanner"  # اسکنر بیومتریک
    MAGNETIC_LOCK = "magnetic_lock"  # قفل مغناطیسی
    ELECTRIC_STRIKE = "electric_strike"  # برقی ضربه‌ای
    TURNSTILE = "turnstile"  # گیت چرخشی
    BARRIER_GATE = "barrier_gate"  # راهبند
    METAL_DETECTOR = "metal_detector"  # فلزیاب
    X_RAY_SCANNER = "x_ray_scanner"  # دستگاه اشعه ایکس
    # سیستم نگهبانی / Guard System
    GUARD_BOOTH = "guard_booth"  # نگهبانی
    PANIC_BUTTON = "panic_button"  # دکمه اضطرار
    INTERCOM = "intercom"  # آیفون تصویری
    # سیستم‌های حفاظتی / Protection Systems
    MOTION_SENSOR = "motion_sensor"  # سنسور حرکتی
    DOOR_CONTACT = "door_contact"  # سنسور درب
    GLASS_BREAK_DETECTOR = "glass_break_detector"  # سنسور شکستن شیشه
    SMOKE_DETECTOR = "smoke_detector"  # آشکارساز دود
    GAS_DETECTOR = "gas_detector"  # آشکارساز گاز
    WATER_LEAK_DETECTOR = "water_leak_detector"  # سنسور نشت آب
    # هشدار و اعلام / Warning & Notification
    SIREN = "siren"  # آژیر
    BEACON = "beacon"  # چراغ هشدار
    EMERGENCY_PHONE = "emergency_phone"  # تلفن اضطراری
    

class AdvancedStructuralElementType(str, Enum):
    """انواع عناصر تحلیل پیشرفته سازه‌ای"""
    # تحلیل لرزه‌ای / Seismic Analysis
    SEISMIC_ISOLATOR = "seismic_isolator"  # جداساز لرزه‌ای
    BASE_ISOLATOR = "base_isolator"  # جداساز پایه
    DAMPER = "damper"  # میراگر
    VISCOUS_DAMPER = "viscous_damper"  # میراگر ویسکوز
    FRICTION_DAMPER = "friction_damper"  # میراگر اصطکاکی
    TUNED_MASS_DAMPER = "tuned_mass_damper"  # میراگر جرمی تنظیم‌شده
    SEISMIC_JOINT = "seismic_joint"  # درز لرزه‌ای
    SHEAR_WALL_REINFORCED = "shear_wall_reinforced"  # دیوار برشی تقویت‌شده
    
    # پی‌های ویژه / Specialized Foundations
    PILE_FOUNDATION = "pile_foundation"  # شمع
    MICRO_PILE = "micro_pile"  # ریزشمع
    CAISSON = "caisson"  # کسن
    MAT_FOUNDATION = "mat_foundation"  # پی گسترده
    RAFT_FOUNDATION = "raft_foundation"  # پی رادیه
    COMBINED_FOOTING = "combined_footing"  # فوتینگ ترکیبی
    PILE_CAP = "pile_cap"  # کلاهک شمع
    GROUND_ANCHOR = "ground_anchor"  # مهار زمین
    SOIL_NAIL = "soil_nail"  # میخ‌کوبی خاک
    
    # تحلیل خاک / Soil Analysis
    SOIL_BORING = "soil_boring"  # حفاری آزمایشی
    SOIL_LAYER = "soil_layer"  # لایه خاک
    WATER_TABLE = "water_table"  # سطح آب زیرزمینی
    BEARING_CAPACITY_ZONE = "bearing_capacity_zone"  # منطقه ظرفیت باربری
    SETTLEMENT_ZONE = "settlement_zone"  # منطقه نشست
    
    # اتصالات پیشرفته / Advanced Connections
    MOMENT_CONNECTION = "moment_connection"  # اتصال گیردار
    BOLTED_CONNECTION = "bolted_connection"  # اتصال پیچی
    WELDED_CONNECTION = "welded_connection"  # اتصال جوشی
    BASE_PLATE = "base_plate"  # صفحه پایه
    COLUMN_SPLICE = "column_splice"  # اتصال ستون
    BEAM_SPLICE = "beam_splice"  # اتصال تیر
    GUSSET_PLATE = "gusset_plate"  # ورق گوشه
    
    # تقویت و بازسازی / Strengthening & Retrofit
    FRP_REINFORCEMENT = "frp_reinforcement"  # تقویت FRP
    CARBON_FIBER = "carbon_fiber"  # الیاف کربن
    STEEL_JACKET = "steel_jacket"  # ژاکت فولادی
    CONCRETE_JACKET = "concrete_jacket"  # ژاکت بتنی
    SHOTCRETE = "shotcrete"  # شاتکریت
    POST_TENSION = "post_tension"  # پس‌کشیده
    PRE_STRESS = "pre_stress"  # پیش‌تنیده
    
    # بارگذاری و آنالیز / Loading & Analysis
    WIND_LOAD_ARROW = "wind_load_arrow"  # بار باد
    SEISMIC_LOAD = "seismic_load"  # بار زلزله
    SNOW_LOAD = "snow_load"  # بار برف
    LIVE_LOAD_ZONE = "live_load_zone"  # منطقه بار زنده
    DEAD_LOAD_ZONE = "dead_load_zone"  # منطقه بار مرده
    LOAD_PATH = "load_path"  # مسیر انتقال بار
    
    # عناصر خاص / Special Elements
    EXPANSION_JOINT = "expansion_joint"  # درز انبساط
    CONSTRUCTION_JOINT = "construction_joint"  # درز اجرایی
    CONTROL_JOINT = "control_joint"  # درز کنترل
    SHEAR_KEY = "shear_key"  # کلید برشی
    TRANSFER_GIRDER = "transfer_girder"  # تیر انتقال
    CORBEL = "corbel"  # کنسول
    CANTILEVER = "cantilever"  # کنسول
    

class SpecialEquipmentElementType(str, Enum):
    """انواع عناصر تجهیزات ویژه"""
    # حمل‌ونقل عمودی / Vertical Transportation
    ELEVATOR = "elevator"  # آسانسور
    ESCALATOR = "escalator"  # پله برقی
    MOVING_WALKWAY = "moving_walkway"  # پیاده‌رو متحرک
    DUMBWAITER = "dumbwaiter"  # آسانسور غذابر
    
    # بارگیری و جابجایی / Loading & Handling
    CRANE = "crane"  # جرثقیل
    HOIST = "hoist"  # وینچ/بالابر
    MONORAIL_HOIST = "monorail_hoist"  # بالابر خطی
    DOCK_LEVELER = "dock_leveler"  # داک لولر
    LOADING_RAMP = "loading_ramp"  # رمپ بارگیری
    
    # آشپزخانه صنعتی / Commercial Kitchen
    COOKING_RANGE = "cooking_range"  # اجاق صنعتی
    HOOD_EXHAUST = "hood_exhaust"  # هود و اگزاست
    DISHWASHER_COMMERCIAL = "dishwasher_commercial"  # ظرفشویی صنعتی
    WALKIN_FRIDGE = "walkin_fridge"  # سردخانه
    WALKIN_FREEZER = "walkin_freezer"  # یخچال فریزر اتاقک
    
    # تجهیزات پزشکی / Medical Equipment
    MRI = "mri"  # ام‌آر‌آی
    CT = "ct"  # سی‌تی اسکن
    X_RAY = "x_ray"  # رادیولوژی
    OPERATING_TABLE = "operating_table"  # تخت جراحی
    AUTOCLAVE = "autoclave"  # اتوکلاو
    
    # صحنه و تئاتر / Stage & Theater
    STAGE_LIFT = "stage_lift"  # لیفت صحنه
    RIGGING = "rigging"  # ریگینگ/آویز صحنه
    ORCHESTRA_PIT_LIFT = "orchestra_pit_lift"  # لیفت گود ارکستر
    
    # استخر و آبنما / Pool & Water Features
    POOL_PUMP = "pool_pump"  # پمپ استخر
    FILTRATION_UNIT = "filtration_unit"  # فیلتر تصفیه
    CHLORINATION_UNIT = "chlorination_unit"  # کلرزنی
    JACUZZI_EQUIPMENT = "jacuzzi_equipment"  # تجهیزات جکوزی
    
    # صنعتی خاص / Industrial Special
    BOILER = "boiler"  # دیگ بخار
    COMPRESSOR = "compressor"  # کمپرسور
    GENERATOR = "generator"  # ژنراتور


class RegulatoryComplianceElementType(str, Enum):
    """عناصر نقشه‌های ضوابط و مقررات"""
    # حریم و ضوابط شهرسازی / Zoning & Setbacks
    PROPERTY_LINE = "property_line"  # حدّ اربعه/حدود ملک
    SETBACK_LINE = "setback_line"  # حریم ساخت
    EASEMENT = "easement"  # محدودیت/حق ارتفاق
    RIGHT_OF_WAY = "right_of_way"  # حریم معبر
    BUILDING_ENVELOPE = "building_envelope"  # پوسته مجاز ساخت

    # کُد و ایمنی / Code Compliance
    FIRE_RATING_WALL = "fire_rating_wall"  # دیوار دارای حریق
    FIRE_RATING_DOOR = "fire_rating_door"  # درب حریق
    EGRESS_PATH = "egress_path"  # مسیر خروج
    EXIT_COUNT = "exit_count"  # تعداد خروجی مجاز
    STAIR_CAPACITY = "stair_capacity"  # ظرفیت پله فرار
    TRAVEL_DISTANCE = "travel_distance"  # فاصله پیمایش

    # کاربری و اشغال / Occupancy
    OCCUPANCY_TYPE = "occupancy_type"  # گروه اشغال
    OCCUPANT_LOAD = "occupant_load"  # بار اشغال
    HAZARD_CLASS = "hazard_class"  # کلاس خطر

    # دسترسی‌پذیری / Accessibility (ADA)
    ADA_RAMP = "ada_ramp"  # رمپ دسترسی
    ADA_DOOR_CLEARANCE = "ada_door_clearance"  # فضای مانور درب
    ADA_RESTROOM = "ada_restroom"  # سرویس بهداشتی قابل دسترس
    ADA_PARKING = "ada_parking"  # پارکینگ معلولین
    ADA_SIGNAGE = "ada_signage"  # علائم دسترسی

    # پارکینگ / Parking Compliance
    PARKING_STALL_COUNT = "parking_stall_count"  # تعداد جای پارک
    EV_PARKING = "ev_parking"  # پارکینگ خودرو برقی
    ACCESSIBLE_PARKING = "accessible_parking"  # پارکینگ قابل دسترس
    BICYCLE_PARKING = "bicycle_parking"  # پارکینگ دوچرخه

    # محیطی و محدودیت‌ها / Environmental
    FLOOD_ZONE = "flood_zone"  # حریم سیلاب
    WETLANDS = "wetlands"  # تالاب/حساس محیطی
    NO_BUILD_ZONE = "no_build_zone"  # محدوده غیرقابل ساخت
    HEIGHT_LIMIT = "height_limit"  # محدودیت ارتفاع
    FAR = "far"  # نسبت سطح اشغال/زیربنا (FAR)

    # مجوزها و یادداشت‌ها / Permits & Notes
    PERMIT_NOTE = "permit_note"
    CODE_REFERENCE = "code_reference"
    INSPECTION_NOTE = "inspection_note"


class SustainabilityElementType(str, Enum):
    """عناصر پایداری و انرژی"""
    # انرژی‌های تجدیدپذیر / Renewables
    SOLAR_PV_PANEL = "solar_pv_panel"
    SOLAR_INVERTER = "solar_inverter"
    SOLAR_THERMAL_COLLECTOR = "solar_thermal_collector"
    BATTERY_STORAGE = "battery_storage"

    # اندازه‌گیری و مانیتورینگ / Metering & Monitoring
    ENERGY_METER = "energy_meter"
    SUB_METER = "sub_meter"
    BMS_PANEL = "bms_panel"

    # پوسته و عایق / Envelope & Insulation
    INSULATION_ZONE = "insulation_zone"
    U_VALUE_NOTE = "u_value_note"
    R_VALUE_NOTE = "r_value_note"
    THERMAL_BREAK = "thermal_break"

    # نور روز و تهویه طبیعی / Daylighting & Natural Ventilation
    SKYLIGHT = "skylight"
    LIGHT_SHELF = "light_shelf"
    SHADING_DEVICE = "shading_device"
    LOUVER = "louver"

    # آب پایدار / Water Sustainability
    RAINWATER_TANK = "rainwater_tank"
    RAINWATER_FILTER = "rainwater_filter"
    GREYWATER_TANK = "greywater_tank"
    GREYWATER_FILTER = "greywater_filter"

    # سبزینگی / Green Features
    GREEN_ROOF = "green_roof"
    GREEN_WALL = "green_wall"

    # کارایی HVAC / HVAC Efficiency
    HEAT_RECOVERY_UNIT = "heat_recovery_unit"  # HRU/ERV/HRV
    VFD_DRIVE = "vfd_drive"
    HIGH_EFF_BOILER = "high_eff_boiler"
    EFFICIENT_CHILLER = "efficient_chiller"

    # گواهی و کد / Certifications & Codes
    LEED_NOTE = "leed_note"
    BREEAM_NOTE = "breeam_note"
    ENERGY_CODE_NOTE = "energy_code_note"

    # نواحی انرژی / Zones
    ENERGY_ZONE = "energy_zone"
    THERMAL_ZONE = "thermal_zone"

class TransportationTrafficElementType(str, Enum):
    """عناصر حمل‌ونقل و ترافیک سایت"""
    # پارکینگ / Parking
    PARKING_STALL = "parking_stall"
    PARKING_AISLE = "parking_aisle"
    ACCESSIBLE_PARKING = "accessible_parking"
    EV_PARKING = "ev_parking"
    LOADING_ZONE = "loading_zone"

    # مسیرهای سواره / Vehicle Routes
    TRAFFIC_LANE = "traffic_lane"
    TRAFFIC_FLOW_ARROW = "traffic_flow_arrow"
    TURNING_RADIUS = "turning_radius"
    DRIVEWAY_ENTRY = "driveway_entry"
    RAMP = "ramp"

    # عابر پیاده / Pedestrian
    WALKWAY = "walkway"
    CROSSWALK = "crosswalk"
    CURB_CUT = "curb_cut"

    # علائم و تابلوها / Signage & Markings
    STOP_SIGN = "stop_sign"
    YIELD_SIGN = "yield_sign"
    SPEED_LIMIT = "speed_limit"
    TRAFFIC_LIGHT = "traffic_light"
    LANE_MARKING = "lane_marking"

    # دوچرخه و اتوبوس / Bike & Bus
    BIKE_LANE = "bike_lane"
    BUS_STOP = "bus_stop"
    DROP_OFF = "drop_off"

    # آرام‌سازی ترافیک / Traffic Calming
    SPEED_BUMP = "speed_bump"

class ITNetworkElementType(str, Enum):
    """عناصر زیرساخت IT و شبکه"""
    RACK = "rack"                # رک شبکه/سرور
    PATCH_PANEL = "patch_panel"  # پچ پنل
    SERVER_EQUIPMENT = "server_equipment"  # سرور/تجهیزات اتاق سرور
    WIFI_AP = "wifi_ap"          # اکسس پوینت بی‌سیم
    DATA_PORT = "data_port"      # پریز/پورت شبکه (RJ45)
    CABLE_TRAY = "cable_tray"    # سینی کابل/ترانکینگ
    FIBER_CABLE = "fiber_cable"  # فیبر نوری/بک‌بون
    CONDUIT = "conduit"          # لوله/کاندوئیت ضعیف
    SERVER_ROOM = "server_room"  # اتاق سرور/اتاق تجهیزات

class ConstructionPhasingElementType(str, Enum):
    """عناصر مراحل پروژه و ساخت"""
    # شماره‌گذاری فاز / Phase Numbering
    PHASE_MARKER = "phase_marker"  # نشانگر فاز (PHASE 1، PHASE 2)
    PHASE_BOUNDARY = "phase_boundary"  # مرز فاز
    PHASE_LABEL = "phase_label"  # برچسب فاز
    
    # تخریب / Demolition
    DEMOLITION_ZONE = "demolition_zone"  # منطقه تخریب
    DEMOLITION_WALL = "demolition_wall"  # دیوار قابل تخریب
    SALVAGE_ITEM = "salvage_item"  # آیتم قابل بازیافت
    HAZMAT_AREA = "hazmat_area"  # منطقه مواد خطرناک
    
    # سازه‌های موقت / Temporary Structures
    TEMPORARY_WALL = "temporary_wall"  # دیوار موقت
    TEMPORARY_FENCE = "temporary_fence"  # حصار موقت
    SHORING = "shoring"  # شمع‌کوبی/نگهداری
    SCAFFOLDING = "scaffolding"  # داربست
    PROTECTION_BARRIER = "protection_barrier"  # مانع حفاظتی
    TEMPORARY_SUPPORT = "temporary_support"  # تکیه‌گاه موقت
    
    # توالی ساخت / Construction Sequence
    CONSTRUCTION_SEQUENCE = "construction_sequence"  # توالی ساخت
    POUR_SEQUENCE = "pour_sequence"  # توالی بتن‌ریزی
    ERECTION_SEQUENCE = "erection_sequence"  # توالی نصب
    
    # مناطق کاری / Work Areas
    STAGING_AREA = "staging_area"  # منطقه آماده‌سازی
    LAYDOWN_AREA = "laydown_area"  # منطقه ذخیره مصالح
    CRANE_LOCATION = "crane_location"  # موقعیت جرثقیل
    ACCESS_ROUTE = "access_route"  # مسیر دسترسی
    
    # یادداشت‌های ساخت / Construction Notes
    CONSTRUCTION_NOTE = "construction_note"  # یادداشت ساخت
    PHASING_NOTE = "phasing_note"  # یادداشت فازبندی
    COORDINATION_NOTE = "coordination_note"  # یادداشت هماهنگی

class InteriorElementType(str, Enum):
    """انواع عناصر معماری داخلی و دکوراسیون"""
    # کفپوش‌ها / Flooring
    FLOORING_TILE = "flooring_tile"  # کاشی کف
    FLOORING_WOOD = "flooring_wood"  # پارکت چوبی
    FLOORING_CARPET = "flooring_carpet"  # موکت/فرش
    FLOORING_STONE = "flooring_stone"  # سنگ کف
    FLOORING_VINYL = "flooring_vinyl"  # وینیل
    FLOORING_LAMINATE = "flooring_laminate"  # لمینت
    FLOORING_MARBLE = "flooring_marble"  # مرمر
    FLOORING_CERAMIC = "flooring_ceramic"  # سرامیک
    # پوشش دیوار / Wall Finishes
    WALL_PAINT = "wall_paint"  # رنگ دیوار
    WALL_TILE = "wall_tile"  # کاشی دیوار
    WALL_WALLPAPER = "wall_wallpaper"  # کاغذ دیواری
    WALL_STONE = "wall_stone"  # سنگ دیوار
    WALL_WOOD_PANEL = "wall_wood_panel"  # پانل چوبی
    WALL_GYPSUM = "wall_gypsum"  # گچ
    WALL_FABRIC = "wall_fabric"  # پارچه دیواری
    # سقف / Ceiling
    CEILING_GYPSUM = "ceiling_gypsum"  # سقف گچی
    CEILING_SUSPENDED = "ceiling_suspended"  # سقف کاذب
    CEILING_ACOUSTIC = "ceiling_acoustic"  # سقف آکوستیک
    CEILING_DECORATIVE = "ceiling_decorative"  # سقف تزئینی
    # نورپردازی داخلی / Interior Lighting
    LIGHT_CHANDELIER = "light_chandelier"  # لوستر
    LIGHT_PENDANT = "light_pendant"  # آویز
    LIGHT_RECESSED = "light_recessed"  # چراغ توکار
    LIGHT_TRACK = "light_track"  # ریل نوری
    LIGHT_WALL_SCONCE = "light_wall_sconce"  # آپلیک دیواری
    LIGHT_FLOOR_LAMP = "light_floor_lamp"  # چراغ ایستاده
    LIGHT_TABLE_LAMP = "light_table_lamp"  # آباژور
    LIGHT_STRIP = "light_strip"  # نوار LED
    LIGHT_SPOTLIGHT = "light_spotlight"  # اسپات لایت
    # مبلمان / Furniture
    FURNITURE_BED = "furniture_bed"  # تخت
    FURNITURE_SOFA = "furniture_sofa"  # مبل
    FURNITURE_CHAIR = "furniture_chair"  # صندلی
    FURNITURE_TABLE = "furniture_table"  # میز
    FURNITURE_DESK = "furniture_desk"  # میز کار
    FURNITURE_CABINET = "furniture_cabinet"  # کابینت
    FURNITURE_SHELF = "furniture_shelf"  # قفسه
    FURNITURE_WARDROBE = "furniture_wardrobe"  # کمد
    FURNITURE_DRESSER = "furniture_dresser"  # دراور
    FURNITURE_NIGHTSTAND = "furniture_nightstand"  # پاتختی
    # دکوراسیون / Decoration
    DECOR_CURTAIN = "decor_curtain"  # پرده
    DECOR_BLINDS = "decor_blinds"  # پرده کرکره‌ای
    DECOR_MIRROR = "decor_mirror"  # آینه
    DECOR_ARTWORK = "decor_artwork"  # تابلو/اثر هنری
    DECOR_PLANT = "decor_plant"  # گیاه تزئینی
    DECOR_VASE = "decor_vase"  # گلدان
    DECOR_RUG = "decor_rug"  # قالیچه
    DECOR_CUSHION = "decor_cushion"  # کوسن
    # آشپزخانه / Kitchen
    KITCHEN_COUNTER = "kitchen_counter"  # پیشخوان
    KITCHEN_ISLAND = "kitchen_island"  # جزیره
    KITCHEN_SINK = "kitchen_sink"  # سینک
    KITCHEN_STOVE = "kitchen_stove"  # اجاق
    KITCHEN_OVEN = "kitchen_oven"  # فر
    KITCHEN_REFRIGERATOR = "kitchen_refrigerator"  # یخچال
    KITCHEN_DISHWASHER = "kitchen_dishwasher"  # ماشین ظرفشویی
    KITCHEN_HOOD = "kitchen_hood"  # هود
    # حمام / Bathroom
    BATHROOM_VANITY = "bathroom_vanity"  # روشویی
    BATHROOM_TOILET = "bathroom_toilet"  # توالت فرنگی
    BATHROOM_SHOWER = "bathroom_shower"  # دوش
    BATHROOM_BATHTUB = "bathroom_bathtub"  # وان
    BATHROOM_MIRROR = "bathroom_mirror"  # آینه حمام
    UNKNOWN = "unknown"


@dataclass
class RoomInfo:
    """اطلاعات یک اتاق یا فضا"""
    name: str
    space_type: SpaceType
    area: float  # مساحت به متر مربع
    perimeter: float  # محیط
    width: float
    length: float
    layer: str
    boundary_vertices: List[Tuple[float, float]]
    center: Tuple[float, float]
    text_entities: List[str]  # متن‌های داخل فضا


@dataclass
class StructuralElementInfo:
    """اطلاعات یک عنصر سازه‌ای"""
    element_type: StructuralElementType
    layer: str
    position: Tuple[float, float, float]  # x, y, z
    dimensions: Tuple[float, float, float]  # width, depth, height/length
    size_designation: str  # مثلاً "C30x40" برای ستون، "B25x50" برای تیر
    material: Optional[str] = "concrete"  # بتن، فولاد، و...
    reinforcement: Optional[str] = None  # آرماتورگذاری
    load_capacity: Optional[float] = None  # ظرفیت باربری (اگر مشخص باشد)


@dataclass
class MEPElementInfo:
    """اطلاعات یک عنصر تأسیساتی (MEP)"""
    element_type: MEPElementType
    layer: str
    position: Tuple[float, float, float]  # x, y, z
    size_designation: str  # مثلاً "Ø100mm" برای لوله، "2000W" برای چراغ
    system: Optional[str] = None  # نام سیستم (مثلاً "DHW", "CW", "HVAC-1")
    flow_rate: Optional[float] = None  # دبی جریان (برای لوله‌ها)
    power: Optional[float] = None  # توان (برای تجهیزات برقی)
    voltage: Optional[str] = None  # ولتاژ (برای برق)
    capacity: Optional[float] = None  # ظرفیت (برای تجهیزات)
    connection_points: Optional[List[Tuple[float, float, float]]] = None  # نقاط اتصال


@dataclass
class ConstructionDetailInfo:
    """اطلاعات جزئیات اجرایی"""
    detail_type: DetailType
    layer: str
    position: Tuple[float, float, float]  # x, y, z
    scale: Optional[str] = None  # مقیاس جزئیات (مثلاً "1:5", "1:10")
    materials: Optional[List[MaterialType]] = None  # مصالح استفاده شده
    dimensions: Optional[Dict[str, float]] = None  # ابعاد کلیدی
    annotations: Optional[List[str]] = None  # یادداشت‌ها و توضیحات
    section_cut: Optional[str] = None  # خط برش (اگر مقطع باشد)
    reference_drawing: Optional[str] = None  # ارجاع به نقشه اصلی
    construction_notes: Optional[List[str]] = None  # یادداشت‌های اجرایی
    material_specs: Optional[Dict[str, str]] = None  # مشخصات مصالح


@dataclass
class SiteElementInfo:
    """اطلاعات عناصر نقشه سایت"""
    element_type: SiteElementType
    layer: str
    position: Tuple[float, float, float]  # x, y, z
    geometry_type: str  # "point", "line", "polyline", "polygon", "circle"
    area: Optional[float] = None  # مساحت (برای ساختمان، پارکینگ، چمن)
    length: Optional[float] = None  # طول (برای جاده، پیاده‌رو)
    width: Optional[float] = None  # عرض (برای جاده، پیاده‌رو)
    boundary_vertices: Optional[List[Tuple[float, float]]] = None  # رئوس محدوده
    label: Optional[str] = None  # برچسب/نام
    distance_to_main: Optional[float] = None  # فاصله تا ساختمان اصلی
    orientation: Optional[float] = None  # جهت (درجه)
    elevation: Optional[float] = None  # ارتفاع
    specifications: Optional[Dict[str, str]] = None  # مشخصات (نوع درخت، عرض جاده، ...)


@dataclass
class CivilElementInfo:
    """اطلاعات عناصر مهندسی سایت و محوطه"""
    element_type: CivilElementType
    layer: str
    position: Tuple[float, float, float]
    geometry_type: str  # line, polyline, polygon, circle, point
    elevation: Optional[float] = None  # ارتفاع (متر)
    slope: Optional[float] = None  # شیب (درصد)
    length: Optional[float] = None  # طول (متر)
    diameter: Optional[float] = None  # قطر لوله (میلی‌متر)
    area: Optional[float] = None  # مساحت (متر مربع)
    volume: Optional[float] = None  # حجم خاکبرداری/خاکریزی (متر مکعب)
    flow_direction: Optional[float] = None  # جهت جریان (درجه)
    invert_elevation: Optional[float] = None  # ارتفاع کف لوله
    rim_elevation: Optional[float] = None  # ارتفاع لبه منهول
    label: Optional[str] = None
    specifications: Optional[str] = None


@dataclass
class InteriorElementInfo:
    """اطلاعات عناصر معماری داخلی و دکوراسیون"""
    element_type: InteriorElementType
    layer: str
    position: Tuple[float, float, float]
    room_name: Optional[str] = None  # نام اتاق/فضا
    geometry_type: Optional[str] = None  # rectangle, circle, polygon, etc.
    dimensions: Optional[Tuple[float, float, float]] = None  # width, depth, height (mm)
    area: Optional[float] = None  # مساحت (متر مربع) - برای کفپوش
    material: Optional[str] = None  # جنس (چوب، سنگ، ...)
    color: Optional[str] = None  # رنگ
    finish: Optional[str] = None  # نوع پرداخت/فینیش
    brand: Optional[str] = None  # برند
    model: Optional[str] = None  # مدل
    quantity: Optional[int] = None  # تعداد
    label: Optional[str] = None
    specifications: Optional[str] = None


@dataclass
class SafetySecurityElementInfo:
    """اطلاعات عناصر ایمنی و امنیت"""
    element_type: SafetySecurityElementType
    layer: str
    position: Tuple[float, float, float]
    geometry_type: Optional[str] = None  # point, block, line, polygon
    zone: Optional[str] = None  # منطقه‌بندی امنیتی/حریق
    coverage_area: Optional[float] = None  # m² - محدوده پوشش
    device_id: Optional[str] = None  # شناسه دستگاه
    manufacturer: Optional[str] = None  # سازنده
    model: Optional[str] = None  # مدل
    capacity: Optional[str] = None  # ظرفیت (برای کپسول، مخزن و ...)
    direction: Optional[float] = None  # جهت (برای دوربین، خروج و ...)
    height: Optional[float] = None  # ارتفاع نصب (متر)
    is_addressable: Optional[bool] = None  # قابلیت آدرس‌پذیری
    connection_type: Optional[str] = None  # نوع اتصال (wired, wireless)
    label: Optional[str] = None
    specifications: Optional[str] = None


@dataclass
class AdvancedStructuralElementInfo:
    """اطلاعات عناصر تحلیل پیشرفته سازه‌ای"""
    element_type: AdvancedStructuralElementType
    layer: str
    position: Tuple[float, float, float]
    geometry_type: Optional[str] = None  # point, line, polygon, block
    dimensions: Optional[Tuple[float, float, float]] = None  # طول، عرض، ارتفاع (میلی‌متر)
    capacity: Optional[float] = None  # ظرفیت باربری (تن، کیلونیوتن)
    material: Optional[str] = None  # جنس (فولاد، بتن، FRP)
    grade: Optional[str] = None  # درجه مقاومت (مثلاً St37، C30)
    diameter: Optional[float] = None  # قطر (میلی‌متر) - برای شمع، میراگر
    depth: Optional[float] = None  # عمق (متر) - برای پی، حفاری
    length: Optional[float] = None  # طول (میلی‌متر)
    load_value: Optional[float] = None  # مقدار بار (کیلونیوتن)
    load_direction: Optional[float] = None  # جهت بار (درجه)
    reinforcement: Optional[str] = None  # آرماتور/تقویت
    connection_type: Optional[str] = None  # نوع اتصال
    analysis_note: Optional[str] = None  # یادداشت تحلیل
    label: Optional[str] = None
    specifications: Optional[str] = None

@dataclass
class SpecialEquipmentElementInfo:
    """اطلاعات تجهیزات ویژه"""
    element_type: SpecialEquipmentElementType
    layer: str
    position: Tuple[float, float, float]
    geometry_type: Optional[str] = None  # point, block, circle, polygon
    dimensions: Optional[Tuple[float, float, float]] = None  # w, d, h (mm)
    manufacturer: Optional[str] = None
    model: Optional[str] = None
    capacity: Optional[float] = None  # ظرفیت (kg, L/s, kW)
    power: Optional[float] = None  # توان kW
    voltage: Optional[float] = None  # ولتاژ V
    room: Optional[str] = None  # مکان/اتاق
    note: Optional[str] = None
    label: Optional[str] = None
    specifications: Optional[str] = None


@dataclass
class RegulatoryComplianceElementInfo:
    """اطلاعات عناصر ضوابط و مقررات"""
    element_type: RegulatoryComplianceElementType
    layer: str
    position: Tuple[float, float, float]
    geometry_type: Optional[str] = None  # text, line, polygon, block
    text: Optional[str] = None
    value: Optional[float] = None
    units: Optional[str] = None
    requirement: Optional[str] = None
    provided: Optional[str] = None
    status: Optional[str] = None  # pass / fail / unknown
    note: Optional[str] = None
    label: Optional[str] = None
    related_layers: Optional[List[str]] = None


@dataclass
class SustainabilityElementInfo:
    """اطلاعات عناصر پایداری و انرژی"""
    element_type: SustainabilityElementType
    layer: str
    position: Tuple[float, float, float]
    geometry_type: Optional[str] = None  # block, text, polygon, circle
    capacity: Optional[float] = None  # kW, m3, etc.
    efficiency: Optional[float] = None  # % or COP/EER
    area: Optional[float] = None  # m² for zones/arrays
    orientation: Optional[float] = None  # degrees (azimuth)
    tilt: Optional[float] = None  # degrees
    note: Optional[str] = None
    label: Optional[str] = None
    specifications: Optional[str] = None

@dataclass
class TransportationElementInfo:
    """اطلاعات عناصر حمل‌ونقل و ترافیک سایت"""
    element_type: TransportationTrafficElementType
    layer: str
    position: Tuple[float, float, float]
    geometry_type: Optional[str] = None  # line, polyline, polygon, block, text, circle
    direction: Optional[float] = None  # degrees
    radius: Optional[float] = None  # for turning radii
    width: Optional[float] = None
    length: Optional[float] = None
    count: Optional[int] = None
    slope: Optional[float] = None  # for ramps
    speed_limit: Optional[float] = None
    vehicle_type: Optional[str] = None
    sign_text: Optional[str] = None
    lane_count: Optional[int] = None
    aisle_width: Optional[float] = None
    turning_radius: Optional[float] = None
    note: Optional[str] = None
    label: Optional[str] = None


@dataclass
class ITNetworkElementInfo:
    """اطلاعات عناصر زیرساخت IT و شبکه"""
    element_type: ITNetworkElementType
    layer: str
    position: Tuple[float, float, float]
    geometry_type: Optional[str] = None  # نوع هندسه
    dimensions: Optional[Tuple[float, float, float]] = None  # طول، عرض، ارتفاع
    label: Optional[str] = None  # برچسب/نام
    note: Optional[str] = None  # یادداشت


@dataclass
class ConstructionPhasingElementInfo:
    """اطلاعات عناصر مراحل پروژه و ساخت"""
    element_type: ConstructionPhasingElementType
    layer: str
    position: Tuple[float, float, float]
    geometry_type: Optional[str] = None  # text, line, polygon, block
    phase_number: Optional[int] = None  # شماره فاز (1، 2، 3، ...)
    sequence_order: Optional[int] = None  # ترتیب در توالی
    status: Optional[str] = None  # existing, demolish, new, temporary
    start_date: Optional[str] = None  # تاریخ شروع
    end_date: Optional[str] = None  # تاریخ پایان
    area: Optional[float] = None  # مساحت (m²)
    volume: Optional[float] = None  # حجم (m³)
    note: Optional[str] = None  # یادداشت
    label: Optional[str] = None  # برچسب
    specifications: Optional[str] = None  # مشخصات
@dataclass
class WallInfo:
    """اطلاعات یک دیوار"""
    start_point: Tuple[float, float]
    end_point: Tuple[float, float]
    length: float
    thickness: float
    layer: str
    is_load_bearing: bool = False  # دیوار باربر
    is_exterior: bool = False  # دیوار خارجی


@dataclass
class DimensionInfo:
    """اطلاعات ابعاد"""
    value: float
    text: str
    start_point: Tuple[float, float]
    end_point: Tuple[float, float]
    layer: str


@dataclass
class ArchitecturalAnalysis:
    """نتیجه تحلیل کامل نقشه معماری و سازه‌ای"""
    drawing_type: DrawingType
    rooms: List[RoomInfo]
    walls: List[WallInfo]
    dimensions: List[DimensionInfo]
    structural_elements: List[StructuralElementInfo]  # عناصر سازه‌ای
    mep_elements: List[MEPElementInfo]  # عناصر تأسیساتی
    construction_details: List[ConstructionDetailInfo]  # جزئیات اجرایی
    site_elements: List[SiteElementInfo]  # عناصر سایت و موقعیت
    civil_elements: List[CivilElementInfo]  # عناصر مهندسی سایت و محوطه
    interior_elements: List[InteriorElementInfo]  # عناصر معماری داخلی و دکوراسیون
    safety_security_elements: List[SafetySecurityElementInfo]  # عناصر ایمنی و امنیت
    advanced_structural_elements: List[AdvancedStructuralElementInfo]  # عناصر تحلیل پیشرفته سازه‌ای
    special_equipment_elements: List['SpecialEquipmentElementInfo']  # تجهیزات ویژه
    regulatory_elements: List['RegulatoryComplianceElementInfo']  # عناصر ضوابط و مقررات
    sustainability_elements: List['SustainabilityElementInfo']  # عناصر پایداری و انرژی
    transportation_elements: List['TransportationElementInfo']  # عناصر حمل‌ونقل و ترافیک
    it_network_elements: List['ITNetworkElementInfo']  # عناصر زیرساخت IT و شبکه
    construction_phasing_elements: List['ConstructionPhasingElementInfo']  # عناصر مراحل پروژه و ساخت
    total_area: float
    building_footprint: Tuple[float, float, float, float]  # bbox: min_x, min_y, max_x, max_y
    layers_info: Dict[str, int]  # تعداد entity در هر لایه
    scale: Optional[float] = None
    unit: str = "mm"
    metadata: Dict[str, any] = None


class ArchitecturalAnalyzer:
    """تحلیلگر نقشه‌های معماری"""
    
    # کلمات کلیدی فارسی و انگلیسی برای شناسایی فضاها
    SPACE_KEYWORDS = {
        SpaceType.BEDROOM: [
            "bedroom", "bed room", "اتاق خواب", "خواب", "master", "اتاق",
            "room", "BR", "مستر"
        ],
        SpaceType.LIVING_ROOM: [
            "living", "living room", "hall", "پذیرایی", "نشیمن", "هال",
            "salon", "سالن", "LR"
        ],
        SpaceType.KITCHEN: [
            "kitchen", "آشپزخانه", "اشپزخانه", "کیچن", "pantry", "KT"
        ],
        SpaceType.BATHROOM: [
            "bathroom", "bath", "wc", "toilet", "سرویس", "دستشویی",
            "حمام", "توالت", "WC", "سرویس بهداشتی"
        ],
        SpaceType.CORRIDOR: [
            "corridor", "hallway", "passage", "راهرو", "گذر", "ورودی"
        ],
        SpaceType.BALCONY: [
            "balcony", "terrace", "بالکن", "تراس", "بالکون"
        ],
        SpaceType.PARKING: [
            "parking", "garage", "پارکینگ", "گاراژ", "پارکنگ", "PK"
        ],
        SpaceType.STORAGE: [
            "storage", "store", "انباری", "انبار", "ST"
        ],
        SpaceType.STAIRCASE: [
            "stair", "staircase", "راه پله", "پله", "stairs"
        ],
        SpaceType.ELEVATOR: [
            "elevator", "lift", "آسانسور", "EL", "لیفت"
        ],
        SpaceType.LOBBY: [
            "lobby", "entrance", "لابی", "ورودی", "entrance hall"
        ],
    }
    
    # لایه‌های معمول در نقشه‌های معماری
    WALL_LAYERS = ["WALLS", "WALL", "دیوار", "دیوارها", "A-WALL", "ARCH-WALL"]
    DIMENSION_LAYERS = ["DIMS", "DIMENSION", "اندازه", "ابعاد", "A-ANNO-DIMS"]
    TEXT_LAYERS = ["TEXT", "NOTES", "متن", "یادداشت", "A-ANNO-TEXT"]
    ROOM_LAYERS = ["ROOMS", "SPACES", "اتاق", "فضا", "A-AREA"]
    
    # لایه‌های سازه‌ای
    STRUCTURAL_LAYERS = ["STRUCTURAL", "STRUCTURE", "سازه", "S-", "STR"]
    COLUMN_LAYERS = ["COLUMN", "COLUMNS", "ستون", "ستون‌ها", "S-COLS", "S-COLUMN", "STR-COL"]
    BEAM_LAYERS = ["BEAM", "BEAMS", "تیر", "تیرها", "S-BEAM", "S-BEAMS", "STR-BEAM", "JOIST"]
    SLAB_LAYERS = ["SLAB", "SLABS", "دال", "سقف", "S-SLAB", "S-DECK", "STR-SLAB", "DECK"]
    FOUNDATION_LAYERS = ["FOUNDATION", "FOOTING", "فونداسیون", "پی", "S-FOUND", "S-FDN", "STR-FDN"]
    
    # کلمات کلیدی سازه‌ای (فارسی + انگلیسی)
    COLUMN_KEYWORDS = [
        "ستون", "column", "col", "C", "c", "ستون‌", "COLUMN", "COL"
    ]
    BEAM_KEYWORDS = [
        "تیر", "beam", "B", "b", "joist", "تیر", "BEAM", "BM"
    ]
    SLAB_KEYWORDS = [
        "دال", "سقف", "slab", "deck", "S", "s", "SLAB", "DECK"
    ]
    FOUNDATION_KEYWORDS = [
        "فونداسیون", "پی", "foundation", "footing", "F", "f", "FDN", "FOUND"
    ]
    REBAR_KEYWORDS = [
        "آرماتور", "میلگرد", "rebar", "reinforcement", "R", "Ø", "φ", "ϕ"
    ]
    
    # لایه‌های تأسیسات MEP
    PLUMBING_LAYERS = ["PLUMBING", "PLUMB", "P-", "لوله‌کشی", "آب", "فاضلاب", "P-WATER", "P-DRAIN", "P-SEWER"]
    HVAC_LAYERS = ["HVAC", "H-", "تهویه", "تاسیسات", "H-DUCT", "H-SUPPLY", "H-RETURN", "MECHANICAL"]
    ELECTRICAL_LAYERS = ["ELECTRICAL", "ELEC", "E-", "برق", "E-POWER", "E-LIGHT", "E-PANEL"]
    LIGHTING_LAYERS = ["LIGHTING", "LIGHT", "L-", "روشنایی", "چراغ", "E-LITE"]
    FIRE_LAYERS = ["FIRE", "FP-", "حریق", "اطفاء", "FP-SPRNK", "FP-ALARM"]
    
    # کلمات کلیدی تأسیسات (فارسی + انگلیسی)
    PLUMBING_KEYWORDS = [
        "لوله", "pipe", "آب", "water", "فاضلاب", "drain", "sewer", "CW", "HW", "DHW",
        "شیر", "valve", "پمپ", "pump", "WC", "توالت", "سینک", "sink"
    ]
    HVAC_KEYWORDS = [
        "کانال", "duct", "هوا", "air", "تهویه", "vent", "فن", "fan", "VAV", "FCU",
        "دریچه", "diffuser", "grille", "AHU", "chiller", "boiler", "HVAC"
    ]
    ELECTRICAL_KEYWORDS = [
        "تابلو", "panel", "پریز", "outlet", "socket", "کلید", "switch",
        "کابل", "cable", "برق", "electrical", "power", "V", "A", "W", "KW"
    ]
    LIGHTING_KEYWORDS = [
        "چراغ", "light", "لامپ", "lamp", "روشنایی", "lighting", "LED", "fixture"
    ]
    FIRE_KEYWORDS = [
        "اسپرینکلر", "sprinkler", "حریق", "fire", "دود", "smoke", "detector",
        "آلارم", "alarm", "کپسول", "extinguisher"
    ]
    
    # لایه‌های جزئیات اجرایی
    DETAIL_LAYERS = [
        "DETAIL", "DET", "DTL", "D-", "جزئیات", "SECTION", "SEC",
        "CONNECTION", "CONN", "اتصال", "CALLOUT", "ASSEMBLY"
    ]
    
    # کلمات کلیدی جزئیات اجرایی
    DETAIL_KEYWORDS = {
        "connection": ["اتصال", "connection", "joint", "conn", "attachment"],
        "door": ["درب", "door", "د", "DR"],
        "window": ["پنجره", "window", "پ", "WIN", "WN"],
        "facade": ["نما", "facade", "cladding", "curtain wall"],
        "waterproofing": ["آب‌بندی", "waterproof", "ایزوگام", "membrane"],
        "insulation": ["عایق", "insulation", "thermal", "acoustic"],
        "flooring": ["کف", "floor", "کف‌پوش", "flooring", "tile"],
        "ceiling": ["سقف", "ceiling", "کاذب", "false ceiling"],
        "stair": ["پله", "stair", "راه پله", "step"],
        "railing": ["نرده", "railing", "handrail", "دست‌انداز"],
        "roof": ["بام", "roof", "سقف", "flashing", "gutter"],
        "wall": ["دیوار", "wall", "partition", "پارتیشن"],
        "frame": ["چارچوب", "frame", "قاب"],
        "threshold": ["آستانه", "threshold", "sill"],
    }
    
    # کلمات کلیدی مصالح
    MATERIAL_KEYWORDS = {
        MaterialType.CONCRETE: ["بتن", "concrete", "C20", "C25", "C30"],
        MaterialType.STEEL: ["فولاد", "steel", "استیل", "ST37", "ST52"],
        MaterialType.BRICK: ["آجر", "brick"],
        MaterialType.BLOCK: ["بلوک", "block", "CMU"],
        MaterialType.STONE: ["سنگ", "stone", "granite", "marble"],
        MaterialType.WOOD: ["چوب", "wood", "timber", "lumber"],
        MaterialType.GLASS: ["شیشه", "glass"],
        MaterialType.ALUMINUM: ["آلومینیوم", "aluminum", "aluminium"],
        MaterialType.CERAMIC: ["سرامیک", "ceramic", "tile"],
        MaterialType.BITUMEN: ["قیر", "bitumen", "ایزوگام"],
        MaterialType.MEMBRANE: ["غشا", "membrane"],
        MaterialType.FOAM_INSULATION: ["فوم", "foam", "پلی‌اورتان", "polyurethane"],
        MaterialType.POLYSTYRENE: ["پلی‌استایرن", "polystyrene", "EPS", "XPS"],
    }
    
    # لایه‌های نقشه سایت
    SITE_LAYERS = [
        "SITE", "S-", "موقعیت", "سایت", "PROPERTY", "BOUNDARY",
        "LANDSCAPE", "TOPO", "CONTOUR", "ACCESS", "PARKING"
    ]
    
    # کلمات کلیدی عناصر سایت
    SITE_KEYWORDS = {
        "building": ["ساختمان", "building", "بنا", "structure"],
        "boundary": ["مرز", "boundary", "ملک", "property", "خط"],
        "road": ["جاده", "road", "خیابان", "street", "راه"],
        "parking": ["پارکینگ", "parking", "پارک", "park"],
        "tree": ["درخت", "tree", "کاج", "pine", "نخل", "palm"],
        "grass": ["چمن", "grass", "چمنکاری", "lawn"],
        "walkway": ["پیاده‌رو", "walkway", "sidewalk", "مسیر"],
        "fence": ["حصار", "fence", "نرده", "railing"],
        "pool": ["استخر", "pool", "آب"],
        "north": ["شمال", "north", "N"],
        "contour": ["کانتور", "contour", "elevation", "ارتفاع"],
        "gate": ["درب", "gate", "دروازه", "entrance"],
        "utility": ["تاسیسات", "utility", "خط"],
    }
    
    # لایه‌های نقشه مهندسی سایت و محوطه
    CIVIL_LAYERS = ["CIVIL", "C-", "GRADING", "GRADE", "CONTOUR", "TOPO", "DRAIN", "DRAINAGE", 
                    "UTILITY", "STORM", "SEWER", "WATER", "شیب", "زهکشی", "کانتور", "محوطه"]
    
    CIVIL_KEYWORDS = {
        "contour": ["کانتور", "contour", "elevation", "ارتفاع", "elev"],
        "grading": ["شیب", "grade", "grading", "slope", "شیب‌بندی"],
        "drainage": ["زهکشی", "drain", "drainage", "آبرو", "swale"],
        "sewer": ["فاضلاب", "sewer", "sani", "ww"],
        "storm": ["طوفان", "storm", "sd", "surface"],
        "water": ["آب", "water", "wm", "شبکه آب"],
        "manhole": ["منهول", "manhole", "mh", "cb"],
        "inlet": ["دریچه", "inlet", "grate"],
        "pipe": ["لوله", "pipe", "line"],
        "culvert": ["گذرگاه", "culvert"],
        "curb": ["جدول", "curb"],
        "gutter": ["آبرو", "gutter"],
        "cut": ["برش", "cut", "excavation", "گودبرداری"],
        "fill": ["خاکریزی", "fill", "embankment"],
    }
    
    # لایه‌های معماری داخلی و دکوراسیون
    INTERIOR_LAYERS = ["INTERIOR", "INT", "I-", "FURNITURE", "FURN", "FF&E", "FLOORING", "FLOOR", 
                       "CEILING", "LIGHTING", "LIGHT", "مبلمان", "کفپوش", "دکور", "نور"]
    
    INTERIOR_KEYWORDS = {
        "flooring": ["کفپوش", "floor", "tile", "کاشی", "پارکت", "parquet", "carpet", "موکت", "سرامیک", "ceramic"],
        "wall_finish": ["دیوار", "wall", "paint", "رنگ", "wallpaper", "کاغذ دیواری", "stone", "سنگ"],
        "ceiling": ["سقف", "ceiling", "gypsum", "گچ", "suspended", "کاذب"],
        "lighting": ["نور", "light", "چراغ", "لوستر", "chandelier", "lamp", "آباژور"],
        "furniture": ["مبل", "furniture", "sofa", "chair", "صندلی", "table", "میز", "bed", "تخت", "desk"],
        "kitchen": ["آشپزخانه", "kitchen", "counter", "پیشخوان", "sink", "سینک", "cabinet", "کابینت"],
        "bathroom": ["حمام", "bathroom", "vanity", "روشویی", "toilet", "توالت", "shower", "دوش", "bath"],
        "curtain": ["پرده", "curtain", "blinds", "کرکره"],
        "decoration": ["دکور", "decor", "mirror", "آینه", "plant", "گیاه", "artwork", "تابلو"],
    }
    
    # لایه‌های ایمنی و امنیت
    SAFETY_SECURITY_LAYERS = ["SAFETY", "SECURITY", "FIRE", "ALARM", "CCTV", "ACCESS", "EMERGENCY", 
                              "EXIT", "SPRINKLER", "EXTINGUISHER", "ایمنی", "امنیت", "حریق", "اعلام"]
    
    SAFETY_SECURITY_KEYWORDS = {
        "fire_alarm": ["اعلام حریق", "fire alarm", "detector", "آشکارساز", "smoke", "دود", "heat", "حرارت"],
        "fire_suppression": ["اطفاء", "sprinkler", "اسپرینکلر", "extinguisher", "کپسول", "hose", "شیلنگ", "hydrant"],
        "exit": ["خروج", "exit", "emergency", "اضطراری", "evacuation", "تخلیه", "escape"],
        "security": ["امنیت", "security", "cctv", "camera", "دوربین", "access", "دسترسی"],
        "access_control": ["کنترل تردد", "access control", "card reader", "کارتخوان", "biometric", "بیومتریک"],
        "detection": ["سنسور", "sensor", "motion", "حرکتی", "door contact", "glass break"],
        "alarm": ["آژیر", "siren", "alarm", "هشدار", "beacon", "چراغ"],
        "guard": ["نگهبانی", "guard", "booth", "پست", "panic", "اضطرار"],
    }
    
    # لایه‌های تحلیل پیشرفته سازه‌ای
    ADVANCED_STRUCTURAL_LAYERS = ["SEISMIC", "DAMPER", "ISOLATOR", "PILE", "FOUNDATION", "SOIL", 
                                  "CONNECTION", "MOMENT", "RETROFIT", "STRENGTHEN", "لرزه", "میراگر", 
                                  "جداساز", "شمع", "پی", "خاک", "تقویت"]
    
    ADVANCED_STRUCTURAL_KEYWORDS = {
        "seismic": ["لرزه", "seismic", "earthquake", "زلزله", "isolator", "جداساز", "damper", "میراگر"],
        "foundation": ["پی", "foundation", "pile", "شمع", "footing", "فوتینگ", "caisson", "کسن"],
        "soil": ["خاک", "soil", "boring", "حفاری", "layer", "لایه", "bearing", "باربری"],
        "connection": ["اتصال", "connection", "moment", "گیردار", "bolt", "پیچ", "weld", "جوش"],
        "reinforcement": ["تقویت", "strengthen", "retrofit", "frp", "jacket", "ژاکت", "fiber", "الیاف"],
        "loading": ["بار", "load", "wind", "باد", "snow", "برف", "seismic", "لرزه"],
        "joint": ["درز", "joint", "expansion", "انبساط", "construction", "اجرایی"],
        "advanced": ["پیشرفته", "advanced", "analysis", "تحلیل", "special", "ویژه"],
    }

    # لایه‌ها و کلمات کلیدی ضوابط و مقررات
    REGULATORY_LAYERS = [
        "ZONING", "SETBACK", "CODE", "FIRE-RATING", "EGRESS", "OCCUPANCY", "ADA", "ACCESSIBILITY",
        "PARKING-REQ", "ENV", "ENVIRONMENT", "PERMIT", "LEGAL", "RIGHT-OF-WAY", "EASEMENT",
        "HEIGHT-LIMIT", "FAR", "NO-BUILD",
        "ضوابط", "مقررات", "حریم", "کاربری", "اشغال", "کد", "دسترسی", "پارکینگ", "محیطی", "مجوز"
    ]

    REGULATORY_KEYWORDS = {
        "zoning": ["zoning", "setback", "right of way", "easement", "envelope", "حد", "حریم", "پوسته"],
        "code": ["code", "fire", "rating", "egress", "exit", "stair", "travel", "ضوابط", "حریق", "خروج"],
        "occupancy": ["occupancy", "load", "hazard", "اشغال", "گروه"],
        "ada": ["ada", "accessible", "ramp", "door", "restroom", "signage", "معلول", "دسترسی"],
        "parking": ["parking", "stall", "ev", "accessible", "bicycle", "پارکینگ"],
        "env": ["flood", "wetland", "no build", "height", "limit", "far", "محیط", "ارتفاع"],
        "permits": ["permit", "inspection", "code", "مجوز", "بازرسی"],
    }

    # لایه‌ها و کلمات کلیدی پایداری و انرژی
    SUSTAINABILITY_LAYERS = [
        "SUSTAIN", "SUSTAINABILITY", "ENERGY", "SOLAR", "PV", "GREEN", "LEED", "BREEAM",
        "RAINWATER", "GREYWATER", "DAYLIGHT", "SHADING", "INSULATION", "U-VALUE", "R-VALUE",
        "METER", "SUB-METER", "BMS", "HRU", "ERV", "HRV", "VFD", "THERMAL-ZONE", "ENERGY-ZONE",
        "پایداری", "انرژی", "خورشیدی", "سبز", "باران", "آب خاکستری", "روشنایی روز", "سایه بان", "عایق"
    ]

    SUSTAINABILITY_KEYWORDS = {
        "renewables": ["solar", "pv", "photovoltaic", "inverter", "collector", "thermal", "battery", "خورشید"],
        "metering": ["meter", "sub", "bms", "monitor"],
        "envelope": ["insulation", "u-value", "r-value", "thermal", "break", "عایق"],
        "daylight": ["daylight", "skylight", "light shelf", "shading", "louver", "سایه"],
        "water": ["rainwater", "greywater", "tank", "filter", "storage"],
        "green": ["green roof", "green wall", "بام سبز", "دیوار سبز"],
        "hvac": ["erv", "hrv", "hru", "vfd", "boiler", "chiller"],
        "cert": ["leed", "breeam", "energy code"],
        "zones": ["energy zone", "thermal zone"],
    }

    # لایه‌ها و کلمات کلیدی تجهیزات ویژه
    SPECIAL_EQUIPMENT_LAYERS = [
        "SPECIAL-EQUIP", "ELEVATOR", "LIFT", "ESCALATOR", "TRAVELATOR", "DUMBWAITER",
        "CRANE", "HOIST", "MONORAIL", "DOCK", "KITCHEN-EQUIP", "MEDICAL-EQUIP",
        "MRI", "CT", "X-RAY", "STAGE", "POOL-EQUIP", "BOILER", "GENERATOR", "COMPRESSOR",
        "تجهیزات", "آسانسور", "پله برقی", "کاشف", "جرثقیل", "بالابر", "آشپزخانه صنعتی", "پزشکی", "استخر"
    ]

    SPECIAL_EQUIPMENT_KEYWORDS = {
        "vertical": ["elevator", "lift", "آسانسور", "escalator", "پله برقی", "moving walkway", "travelator", "dumbwaiter"],
        "loading": ["crane", "جرثقیل", "hoist", "بالابر", "monorail", "dock", "leveler", "ramp"],
        "kitchen": ["kitchen", "آشپزخانه", "range", "اجاق", "hood", "هود", "exhaust", "dishwasher", "سردخانه", "walk in", "freezer"],
        "medical": ["medical", "پزشکی", "mri", "ct", "x-ray", "radiology", "operating", "جراحی", "autoclave"],
        "stage": ["stage", "صحنه", "rigging", "orchestra", "pit", "lift"],
        "pool": ["pool", "استخر", "pump", "پمپ", "filter", "فیلتر", "chlor", "کلر", "jacuzzi", "جکوزی"],
        "industrial": ["boiler", "دیگ", "compressor", "کمپرسور", "generator", "ژنراتور"],
    }

    # لایه‌ها و کلمات کلیدی زیرساخت IT و شبکه
    IT_LAYERS = [
        "IT", "NETWORK", "DATA", "LAN", "TEL", "COMM", "TRAY", "CABLE", "CAT6", "FIBER", "BACKBONE",
        "WIFI", "AP", "SERVER", "RACK", "PATCH", "IDF", "MDF",
        "شبکه", "رک", "سرور", "دیتا", "اتاق سرور"
    ]

    IT_KEYWORDS = {
        "racks": ["rack", "رک"],
        "patch": ["patch", "panel", "پچ"],
        "ap": ["ap", "access point", "wifi", "وای فای"],
        "port": ["data", "rj45", "cat6", "lan", "پریز", "دیتا"],
        "cable": ["tray", "trunk", "trunking", "cable", "cat6"],
        "fiber": ["fiber", "fibre", "optic", "backbone", "فیبر"],
        "server": ["server", "سرور", "idf", "mdf", "اتاق سرور"],
    }

    # لایه‌ها و کلمات کلیدی حمل‌ونقل و ترافیک
    TRANSPORT_LAYERS = [
        "TRAFFIC", "ROAD", "PARKING", "LANE", "SIGN", "SIGNAGE", "ARROW", "LOADING", "DRIVEWAY",
        "RAMP", "CROSSWALK", "WALKWAY", "SIDEWALK", "BIKE", "BUS", "DROP", "CURB", "SPEED", "TURN",
        "ترافیک", "راه", "پارکینگ", "تابلو", "رمپ", "پیاده", "دوچرخه", "اتوبوس"
    ]

    TRANSPORT_KEYWORDS = {
        "parking": ["parking", "پارکینگ", "stall", "space", "aisle", "ev", "accessible", "ada"],
        "vehicle": ["lane", "drive", "driveway", "arrow", "turn", "radius", "ramp", "entry"],
        "pedestrian": ["walk", "walkway", "sidewalk", "crosswalk", "curb", "curb cut", "پیاده"],
        "signage": ["sign", "stop", "yield", "speed", "limit", "traffic light", "signal", "mark"],
        "bike_bus": ["bike", "bicycle", "bus", "stop", "drop", "drop-off"],
        "calming": ["speed bump", "speed table", "bump", "table"],
    }

    # لایه‌ها و کلمات کلیدی مراحل پروژه و ساخت
    PHASING_LAYERS = [
        "PHASE", "PHASING", "DEMO", "DEMOLITION", "TEMP", "TEMPORARY", "CONSTRUCTION",
        "SEQUENCE", "STAGING", "SHORING", "SCAFFOLD", "PROTECTION", "EXIST", "NEW",
        "LAYDOWN", "CRANE", "ACCESS",
        "فاز", "مرحله", "تخریب", "موقت", "ساخت", "توالی", "نگهداری"
    ]

    PHASING_KEYWORDS = {
        "phase": ["phase", "فاز", "مرحله", "stage"],
        "demolition": ["demo", "demolish", "remove", "تخریب", "salvage", "hazmat"],
        "temporary": ["temp", "temporary", "موقت", "shoring", "scaffold", "داربست", "protection", "barrier"],
        "sequence": ["sequence", "توالی", "pour", "بتن‌ریزی", "erection", "نصب"],
        "staging": ["staging", "laydown", "crane", "جرثقیل", "access", "دسترسی"],
        "notes": ["construction note", "phasing", "coordination", "یادداشت", "هماهنگی"],
        "status": ["existing", "new", "موجود", "جدید", "future", "آینده"],
    }
    
    def __init__(self, dxf_path: str):
        """
        آغاز تحلیلگر
        
        Args:
            dxf_path: مسیر فایل DXF
        """
        self.doc = ezdxf.readfile(dxf_path)
        self.msp = self.doc.modelspace()
        self.dxf_path = dxf_path
        
    def analyze(self) -> ArchitecturalAnalysis:
        """
        تحلیل کامل نقشه معماری
        
        Returns:
            ArchitecturalAnalysis: نتایج تحلیل
        """
        # شناسایی نوع نقشه
        drawing_type = self._detect_drawing_type()
        
        # استخراج اطلاعات لایه‌ها
        layers_info = self._analyze_layers()
        
        # تشخیص دیوارها
        walls = self._detect_walls()
        
        # تشخیص ابعاد
        dimensions = self._extract_dimensions()
        
        # تشخیص فضاها و اتاق‌ها
        rooms = self._detect_rooms()
        
        # تشخیص عناصر سازه‌ای
        structural_elements = self._detect_structural_elements()
        
        # تشخیص عناصر تأسیساتی MEP
        mep_elements = self._detect_mep_elements()
        
        # تشخیص جزئیات اجرایی
        construction_details = self._detect_construction_details()
        
        # تشخیص عناصر سایت (فقط اگر نوع نقشه SITE_PLAN باشد)
        site_elements = []
        if drawing_type == DrawingType.SITE_PLAN:
            site_elements = self._detect_site_elements()
        
        # تشخیص عناصر مهندسی سایت و محوطه
        civil_elements = []
        if drawing_type == DrawingType.SITE_PLAN:
            civil_elements = self._detect_civil_elements()
        
        # تشخیص عناصر معماری داخلی و دکوراسیون
        interior_elements = []
        if drawing_type == DrawingType.PLAN:  # فقط برای پلان‌های معماری
            interior_elements = self._detect_interior_elements()
        
        # تشخیص عناصر ایمنی و امنیت
        safety_security_elements = self._detect_safety_security_elements()
        
        # تشخیص عناصر تحلیل پیشرفته سازه‌ای
        advanced_structural_elements = self._detect_advanced_structural_elements()

        # تشخیص تجهیزات ویژه
        special_equipment_elements = self._detect_special_equipment_elements()

        # تشخیص ضوابط و مقررات
        regulatory_elements = self._detect_regulatory_elements()

        # تشخیص پایداری و انرژی
        sustainability_elements = self._detect_sustainability_elements()
        
        # تشخیص عناصر حمل‌ونقل و ترافیک (برای پلان سایت و پارکینگ‌ها نیز مفید)
        transportation_elements = self._detect_transportation_elements()

        # تشخیص زیرساخت IT و شبکه
        it_network_elements = self._detect_it_network_elements()

        # تشخیص مراحل پروژه و ساخت
        construction_phasing_elements = self._detect_construction_phasing_elements()
        
        # محاسبه مساحت کل
        total_area = sum(room.area for room in rooms)
        
        # محاسبه کادر محیطی ساختمان
        footprint = self._calculate_building_footprint()
        
        return ArchitecturalAnalysis(
            drawing_type=drawing_type,
            rooms=rooms,
            walls=walls,
            dimensions=dimensions,
            structural_elements=structural_elements,
            mep_elements=mep_elements,
            construction_details=construction_details,
            site_elements=site_elements,
            civil_elements=civil_elements,
            interior_elements=interior_elements,
            safety_security_elements=safety_security_elements,
            advanced_structural_elements=advanced_structural_elements,
            special_equipment_elements=special_equipment_elements,
            regulatory_elements=regulatory_elements,
            sustainability_elements=sustainability_elements,
            transportation_elements=transportation_elements,
            it_network_elements=it_network_elements,
            construction_phasing_elements=construction_phasing_elements,
            total_area=total_area,
            building_footprint=footprint,
            layers_info=layers_info,
            unit="mm",
            metadata={
                "file_path": self.dxf_path,
                "num_rooms": len(rooms),
                "num_walls": len(walls),
                "num_dimensions": len(dimensions),
                "num_structural_elements": len(structural_elements),
                "num_mep_elements": len(mep_elements),
                "num_construction_details": len(construction_details),
                "num_site_elements": len(site_elements),
                "num_civil_elements": len(civil_elements),
                "num_interior_elements": len(interior_elements),
                "num_safety_security_elements": len(safety_security_elements),
                "num_advanced_structural_elements": len(advanced_structural_elements),
                "num_special_equipment_elements": len(special_equipment_elements),
                "num_regulatory_elements": len(regulatory_elements),
                "num_sustainability_elements": len(sustainability_elements),
                "num_transportation_elements": len(transportation_elements),
                "num_it_network_elements": len(it_network_elements),
                "num_construction_phasing_elements": len(construction_phasing_elements),
            }
        )
    
    def _detect_drawing_type(self) -> DrawingType:
        """تشخیص نوع نقشه (پلان، نما، برش، سازه‌ای، تأسیسات، جزئیات، سایت)"""
        # بررسی نام فایل و لایه‌ها
        filename_lower = self.dxf_path.lower()
        
        # بررسی نقشه سایت (اولویت بالا - قبل از plan)
        if "site" in filename_lower or "سایت" in filename_lower or "موقعیت" in filename_lower:
            return DrawingType.SITE_PLAN
        
        # بررسی لایه‌ها برای نقشه سایت
        site_layer_count = 0
        for entity in self.msp:
            layer = entity.dxf.layer.upper() if hasattr(entity.dxf, 'layer') else ""
            if any(kw in layer for kw in ["SITE", "PROPERTY", "BOUNDARY", "PARKING", "TREE", "LANDSCAPE"]):
                site_layer_count += 1
                if site_layer_count > 5:  # اگر بیش از 5 entity با لایه سایت داشت
                    return DrawingType.SITE_PLAN
        
        # بررسی نقشه‌های تأسیسات MEP (اولویت بالا - قبل از plan)
        if "plumbing" in filename_lower or "لوله‌کشی" in filename_lower:
            return DrawingType.PLUMBING
        elif "hvac" in filename_lower or ("تهویه" in filename_lower) or ("duct" in filename_lower):
            return DrawingType.HVAC
        elif "electrical" in filename_lower or ("برق" in filename_lower and "power" in filename_lower):
            return DrawingType.ELECTRICAL
        elif "lighting" in filename_lower or ("روشنایی" in filename_lower):
            return DrawingType.LIGHTING
        elif "fire" in filename_lower or "حریق" in filename_lower or "sprinkler" in filename_lower:
            return DrawingType.FIRE_PROTECTION
        
        # بررسی نقشه‌های سازه‌ای (قبل از plan)
        if "foundation" in filename_lower or "فونداسیون" in filename_lower or "footing" in filename_lower:
            return DrawingType.FOUNDATION
        elif "beam" in filename_lower and "layout" in filename_lower:
            return DrawingType.BEAM_LAYOUT
        elif "column" in filename_lower and "layout" in filename_lower:
            return DrawingType.COLUMN_LAYOUT
        elif ("slab" in filename_lower or "دال" in filename_lower) and not "detail" in filename_lower:
            return DrawingType.SLAB
        elif "structural" in filename_lower or ("سازه" in filename_lower and not "detail" in filename_lower):
            return DrawingType.STRUCTURAL_PLAN
        
        # بررسی جزئیات اجرایی (اولویت متوسط - قبل از plan)
        if "detail" in filename_lower or "جزئیات" in filename_lower or "connection" in filename_lower:
            return DrawingType.CONSTRUCTION_DETAIL
        
        # بررسی نقشه‌های معماری
        if "plan" in filename_lower or "پلان" in filename_lower or "floor" in filename_lower:
            return DrawingType.PLAN
        elif "elevation" in filename_lower or "نما" in filename_lower:
            return DrawingType.ELEVATION
        elif "section" in filename_lower or "برش" in filename_lower:
            return DrawingType.SECTION
        
        # بررسی بر اساس محتوای نقشه
        has_3d_elements = any(
            hasattr(e.dxf, 'elevation') and e.dxf.elevation != 0
            for e in self.msp
        )
        
        if has_3d_elements:
            return DrawingType.SECTION
        
        return DrawingType.PLAN  # پیش‌فرض
    
    def _analyze_layers(self) -> Dict[str, int]:
        """تحلیل لایه‌ها و شمارش entity ها"""
        layers_info = {}
        for entity in self.msp:
            layer = entity.dxf.layer
            layers_info[layer] = layers_info.get(layer, 0) + 1
        return layers_info
    
    def _detect_walls(self) -> List[WallInfo]:
        """تشخیص دیوارها"""
        walls = []
        
        # جستجوی خطوط و polyline ها در لایه‌های دیوار
        for entity in self.msp:
            layer = entity.dxf.layer.upper()
            
            # بررسی اینکه آیا در لایه دیوار است
            is_wall_layer = any(wl.upper() in layer for wl in self.WALL_LAYERS)
            
            if not is_wall_layer:
                continue
            
            if entity.dxftype() == 'LINE':
                line: Line = entity
                start = (line.dxf.start.x, line.dxf.start.y)
                end = (line.dxf.end.x, line.dxf.end.y)
                length = math.hypot(end[0] - start[0], end[1] - start[1])
                
                walls.append(WallInfo(
                    start_point=start,
                    end_point=end,
                    length=length,
                    thickness=0,  # باید از polyline های موازی محاسبه شود
                    layer=entity.dxf.layer,
                ))
            
            elif entity.dxftype() == 'LWPOLYLINE':
                lwpoly: LWPolyline = entity
                points = list(lwpoly.get_points(format='xy'))
                
                # هر لبه را به عنوان دیوار در نظر بگیریم
                for i in range(len(points) - 1):
                    start = points[i]
                    end = points[i + 1]
                    length = math.hypot(end[0] - start[0], end[1] - start[1])
                    
                    walls.append(WallInfo(
                        start_point=start,
                        end_point=end,
                        length=length,
                        thickness=0,
                        layer=entity.dxf.layer,
                    ))
        
        return walls
    
    def _extract_dimensions(self) -> List[DimensionInfo]:
        """استخراج ابعاد از نقشه"""
        dimensions = []
        
        for entity in self.msp.query('DIMENSION'):
            dim: Dimension = entity
            try:
                measurement = dim.get_measurement()
                text = dim.dxf.get('text', str(measurement))
                
                # استخراج نقاط ابتدا و انتها (ساده‌سازی شده)
                dimensions.append(DimensionInfo(
                    value=measurement,
                    text=text,
                    start_point=(0, 0),  # نیاز به پردازش دقیق‌تر
                    end_point=(0, 0),
                    layer=entity.dxf.layer,
                ))
            except:
                continue
        
        return dimensions
    
    def _detect_rooms(self) -> List[RoomInfo]:
        """تشخیص اتاق‌ها و فضاها"""
        rooms = []
        
        # جستجوی polyline های بسته که می‌توانند فضا باشند
        for entity in self.msp.query('LWPOLYLINE'):
            poly: LWPolyline = entity
            
            if not poly.closed:
                continue
            
            # محاسبه مساحت و محیط
            points = list(poly.get_points(format='xy'))
            area = self._calculate_area(points)
            perimeter = self._calculate_perimeter(points)
            
            # فیلتر: فقط فضاهای معقول (> 1 متر مربع و < 500 متر مربع)
            area_m2 = area / 1_000_000  # تبدیل mm² به m²
            if area_m2 < 1 or area_m2 > 500:
                continue
            
            # محاسبه عرض و طول تقریبی
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            width = max(xs) - min(xs)
            length = max(ys) - min(ys)
            center = (sum(xs) / len(xs), sum(ys) / len(ys))
            
            # جستجوی متن‌ها داخل فضا
            texts = self._find_texts_in_space(points)
            
            # تشخیص نوع فضا
            space_type = self._identify_space_type(texts, area_m2)
            
            # نام فضا
            room_name = texts[0] if texts else f"Space_{len(rooms)+1}"
            
            rooms.append(RoomInfo(
                name=room_name,
                space_type=space_type,
                area=area_m2,
                perimeter=perimeter / 1000,  # تبدیل mm به m
                width=width / 1000,
                length=length / 1000,
                layer=entity.dxf.layer,
                boundary_vertices=points,
                center=center,
                text_entities=texts,
            ))
        
        return rooms
    
    def _calculate_area(self, points: List[Tuple[float, float]]) -> float:
        """محاسبه مساحت با فرمول Shoelace"""
        n = len(points)
        if n < 3:
            return 0
        
        area = 0
        for i in range(n):
            j = (i + 1) % n
            area += points[i][0] * points[j][1]
            area -= points[j][0] * points[i][1]
        
        return abs(area) / 2
    
    def _calculate_perimeter(self, points: List[Tuple[float, float]]) -> float:
        """محاسبه محیط"""
        perimeter = 0
        n = len(points)
        for i in range(n):
            j = (i + 1) % n
            dx = points[j][0] - points[i][0]
            dy = points[j][1] - points[i][1]
            perimeter += math.hypot(dx, dy)
        return perimeter
    
    def _find_texts_in_space(self, boundary: List[Tuple[float, float]]) -> List[str]:
        """یافتن متن‌های داخل یک فضا"""
        texts = []
        
        # محاسبه کادر محیطی
        xs = [p[0] for p in boundary]
        ys = [p[1] for p in boundary]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        # جستجوی متن‌ها
        for entity in self.msp.query('TEXT MTEXT'):
            try:
                if entity.dxftype() == 'TEXT':
                    text_content = entity.dxf.text
                    pos = entity.dxf.insert
                else:  # MTEXT
                    text_content = entity.text
                    pos = entity.dxf.insert
                
                # بررسی اینکه آیا در کادر است
                if min_x <= pos.x <= max_x and min_y <= pos.y <= max_y:
                    texts.append(text_content.strip())
            except:
                continue
        
        return texts
    
    def _identify_space_type(self, texts: List[str], area_m2: float) -> SpaceType:
        """تشخیص نوع فضا بر اساس متن‌ها و مساحت"""
        # ترکیب همه متن‌ها
        combined_text = " ".join(texts).lower()
        
        # جستجوی کلمات کلیدی
        for space_type, keywords in self.SPACE_KEYWORDS.items():
            for keyword in keywords:
                if keyword.lower() in combined_text:
                    return space_type
        
        # حدس بر اساس مساحت
        if area_m2 < 3:
            return SpaceType.BATHROOM
        elif area_m2 < 8:
            return SpaceType.STORAGE
        elif area_m2 < 15:
            return SpaceType.BEDROOM
        elif area_m2 < 30:
            return SpaceType.LIVING_ROOM
        
        return SpaceType.UNKNOWN
    
    def _detect_structural_elements(self) -> List[StructuralElementInfo]:
        """
        تشخیص عناصر سازه‌ای (ستون، تیر، دال، فونداسیون)
        
        Returns:
            لیست عناصر سازه‌ای شناسایی شده
        """
        elements = []
        
        # تشخیص ستون‌ها
        elements.extend(self._detect_columns())
        
        # تشخیص تیرها
        elements.extend(self._detect_beams())
        
        # تشخیص دال‌ها
        elements.extend(self._detect_slabs())
        
        # تشخیص فونداسیون‌ها
        elements.extend(self._detect_foundations())
        
        return elements
    
    def _detect_columns(self) -> List[StructuralElementInfo]:
        """تشخیص ستون‌ها از دایره‌ها، مربع‌ها و برچسب‌های متنی"""
        columns = []
        
        # جستجوی در لایه‌های ستون
        for entity in self.msp:
            layer_name_upper = entity.dxf.layer.upper()
            
            # بررسی لایه - چک کردن اینکه آیا نام لایه شامل کلمات کلیدی ستون است
            is_column_layer = any(col_layer.upper() in layer_name_upper for col_layer in self.COLUMN_LAYERS)
            
            # تشخیص دایره‌ها (ستون‌های دایره‌ای)
            if entity.dxftype() == 'CIRCLE' and is_column_layer:
                try:
                    center = entity.dxf.center
                    radius = entity.dxf.radius
                    diameter = radius * 2
                    
                    # جستجوی برچسب نزدیک
                    label = self._find_nearby_text(center, search_radius=radius*3)
                    size_designation = self._extract_column_size(label) if label else f"C{int(diameter)}"
                    
                    columns.append(StructuralElementInfo(
                        element_type=StructuralElementType.COLUMN,
                        position=(center.x, center.y, 0),
                        dimensions=(diameter, diameter, 0),  # width, depth, height (unknown)
                        size_designation=size_designation,
                        layer=entity.dxf.layer,
                        material="concrete",
                        reinforcement=None
                    ))
                except Exception as e:
                    # Skip problematic entities
                    continue
            
            # تشخیص مربع/مستطیل (ستون‌های مستطیلی)
            elif entity.dxftype() == 'LWPOLYLINE' and is_column_layer:
                try:
                    if not entity.is_closed:
                        continue
                    
                    points = list(entity.get_points('xy'))
                    # یک polyline بسته می‌تواند 4 یا 5 نقطه داشته باشد (نقطه آخر ممکن است تکرار اول باشد)
                    if len(points) < 4 or len(points) > 5:
                        continue
                    
                    # محاسبه ابعاد
                    xs = [p[0] for p in points]
                    ys = [p[1] for p in points]
                    width = max(xs) - min(xs)
                    depth = max(ys) - min(ys)
                    center_x = (max(xs) + min(xs)) / 2
                    center_y = (max(ys) + min(ys)) / 2
                    
                    # فقط اشکال کوچک (ستون‌ها معمولاً کوچک هستند)
                    if width < 2000 and depth < 2000 and width > 100 and depth > 100:
                        label = self._find_nearby_text((center_x, center_y, 0), search_radius=max(width, depth))
                        size_designation = self._extract_column_size(label) if label else f"C{int(width)}x{int(depth)}"
                        
                        columns.append(StructuralElementInfo(
                            element_type=StructuralElementType.COLUMN,
                            position=(center_x, center_y, 0),
                            dimensions=(width, depth, 0),
                            size_designation=size_designation,
                            layer=entity.dxf.layer,
                            material="concrete",
                            reinforcement=None
                        ))
                except:
                    continue
        
        # جستجوی برچسب‌های ستون (TEXT/MTEXT با کلمات کلیدی)
        for entity in self.msp.query('TEXT MTEXT'):
            try:
                text_content = entity.dxf.text if entity.dxftype() == 'TEXT' else entity.text
                text_upper = text_content.upper()
                
                # بررسی کلمات کلیدی ستون
                if any(keyword.upper() in text_upper for keyword in self.COLUMN_KEYWORDS):
                    pos = entity.dxf.insert
                    size_designation = self._extract_column_size(text_content)
                    
                    # اگر ستونی با همین موقعیت نداریم، اضافه کن
                    if not any(math.hypot(col.position[0]-pos.x, col.position[1]-pos.y) < 1000 for col in columns):
                        # استخراج ابعاد از برچسب
                        dims = self._parse_structural_dimensions(text_content)
                        
                        columns.append(StructuralElementInfo(
                            element_type=StructuralElementType.COLUMN,
                            position=(pos.x, pos.y, 0),
                            dimensions=dims if dims else (0, 0, 0),
                            size_designation=size_designation,
                            layer=entity.dxf.layer,
                            material="concrete",
                            reinforcement=None
                        ))
            except:
                continue
        
        return columns
    
    def _detect_beams(self) -> List[StructuralElementInfo]:
        """تشخیص تیرها از خطوط و برچسب‌های متنی"""
        beams = []
        
        # جستجوی در لایه‌های تیر
        for entity in self.msp:
            layer_name_upper = entity.dxf.layer.upper()
            is_beam_layer = any(beam_layer.upper() in layer_name_upper for beam_layer in self.BEAM_LAYERS)
            
            # تشخیص خطوط و polyline‌ها
            if (entity.dxftype() in ['LINE', 'LWPOLYLINE']) and is_beam_layer:
                try:
                    if entity.dxftype() == 'LINE':
                        start = entity.dxf.start
                        end = entity.dxf.end
                        length = math.hypot(end.x - start.x, end.y - start.y)
                        center_x = (start.x + end.x) / 2
                        center_y = (start.y + end.y) / 2
                    else:  # LWPOLYLINE
                        points = list(entity.get_points('xy'))
                        if len(points) < 2:
                            continue
                        center_x = sum(p[0] for p in points) / len(points)
                        center_y = sum(p[1] for p in points) / len(points)
                        length = sum(math.hypot(points[i+1][0]-points[i][0], points[i+1][1]-points[i][1]) 
                                   for i in range(len(points)-1))
                    
                    # جستجوی برچسب نزدیک
                    label = self._find_nearby_text((center_x, center_y, 0), search_radius=500)
                    size_designation = self._extract_beam_size(label) if label else f"B{int(length)}"
                    dims = self._parse_structural_dimensions(label) if label else (0, 0, length)
                    
                    beams.append(StructuralElementInfo(
                        element_type=StructuralElementType.BEAM,
                        position=(center_x, center_y, 0),
                        dimensions=dims,
                        size_designation=size_designation,
                        layer=entity.dxf.layer,
                        material="concrete",
                        reinforcement=None
                    ))
                except:
                    continue
        
        # جستجوی برچسب‌های تیر
        for entity in self.msp.query('TEXT MTEXT'):
            try:
                text_content = entity.dxf.text if entity.dxftype() == 'TEXT' else entity.text
                text_upper = text_content.upper()
                
                if any(keyword.upper() in text_upper for keyword in self.BEAM_KEYWORDS):
                    pos = entity.dxf.insert
                    size_designation = self._extract_beam_size(text_content)
                    dims = self._parse_structural_dimensions(text_content)
                    
                    if not any(math.hypot(beam.position[0]-pos.x, beam.position[1]-pos.y) < 1000 for beam in beams):
                        beams.append(StructuralElementInfo(
                            element_type=StructuralElementType.BEAM,
                            position=(pos.x, pos.y, 0),
                            dimensions=dims if dims else (0, 0, 0),
                            size_designation=size_designation,
                            layer=entity.dxf.layer,
                            material="concrete",
                            reinforcement=None
                        ))
            except:
                continue
        
        return beams
    
    def _detect_slabs(self) -> List[StructuralElementInfo]:
        """تشخیص دال‌ها از polyline‌های بسته و برچسب‌ها"""
        slabs = []
        
        # جستجوی در لایه‌های دال
        for entity in self.msp:
            layer_name_upper = entity.dxf.layer.upper()
            is_slab_layer = any(slab_layer.upper() in layer_name_upper for slab_layer in self.SLAB_LAYERS)
            
            if entity.dxftype() == 'LWPOLYLINE' and entity.is_closed and is_slab_layer:
                try:
                    points = list(entity.get_points('xy'))
                    
                    # محاسبه مساحت و مرکز
                    area = abs(self._calculate_area(points))
                    if area < 1000000:  # فقط دال‌های بزرگتر از 1 متر مربع (1,000,000 mm²)
                        continue
                    
                    xs = [p[0] for p in points]
                    ys = [p[1] for p in points]
                    center_x = sum(xs) / len(xs)
                    center_y = sum(ys) / len(ys)
                    
                    # جستجوی برچسب
                    label = self._find_nearby_text((center_x, center_y, 0), search_radius=2000)
                    thickness = self._extract_slab_thickness(label) if label else 0
                    
                    slabs.append(StructuralElementInfo(
                        element_type=StructuralElementType.SLAB,
                        position=(center_x, center_y, 0),
                        dimensions=(max(xs)-min(xs), max(ys)-min(ys), thickness),
                        size_designation=f"S-{int(area/1000000)}m2-t{thickness}mm" if thickness else f"S-{int(area/1000000)}m2",
                        layer=entity.dxf.layer,
                        material="concrete",
                        reinforcement=None
                    ))
                except Exception as e:
                    # Skip problematic entities
                    continue
        
        return slabs
    
    def _detect_foundations(self) -> List[StructuralElementInfo]:
        """تشخیص فونداسیون‌ها"""
        foundations = []
        
        # جستجوی در لایه‌های فونداسیون
        for entity in self.msp:
            layer_name_upper = entity.dxf.layer.upper()
            is_foundation_layer = any(fdn_layer.upper() in layer_name_upper for fdn_layer in self.FOUNDATION_LAYERS)
            
            if entity.dxftype() in ['LWPOLYLINE', 'CIRCLE'] and is_foundation_layer:
                try:
                    if entity.dxftype() == 'CIRCLE':
                        center = entity.dxf.center
                        radius = entity.dxf.radius
                        diameter = radius * 2
                        position = (center.x, center.y, 0)
                        dimensions = (diameter, diameter, 0)
                    else:  # LWPOLYLINE
                        if not entity.is_closed:
                            continue
                        points = list(entity.get_points('xy'))
                        xs = [p[0] for p in points]
                        ys = [p[1] for p in points]
                        position = (sum(xs)/len(xs), sum(ys)/len(ys), 0)
                        dimensions = (max(xs)-min(xs), max(ys)-min(ys), 0)
                    
                    label = self._find_nearby_text(position, search_radius=1000)
                    
                    foundations.append(StructuralElementInfo(
                        element_type=StructuralElementType.FOUNDATION,
                        position=position,
                        dimensions=dimensions,
                        size_designation=label if label else "FDN",
                        layer=entity.dxf.layer,
                        material="concrete",
                        reinforcement=None
                    ))
                except:
                    continue
        
        return foundations
    
    def _find_nearby_text(self, position: Tuple[float, float, float], search_radius: float) -> Optional[str]:
        """یافتن نزدیکترین متن به یک موقعیت"""
        nearest_text = None
        nearest_distance = search_radius
        
        for entity in self.msp.query('TEXT MTEXT'):
            try:
                text_content = entity.dxf.text if entity.dxftype() == 'TEXT' else entity.text
                pos = entity.dxf.insert
                
                distance = math.hypot(pos.x - position[0], pos.y - position[1])
                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest_text = text_content.strip()
            except:
                continue
        
        return nearest_text
    
    def _extract_column_size(self, text: str) -> str:
        """استخراج اندازه ستون از متن (مثال: C30x40 یا ستون 30*40)"""
        if not text:
            return "C"
        
        import re
        # الگوهای معمول: C30x40, 30x40, 30*40, ۳۰×۴۰
        patterns = [
            r'[Cc][-_]?(\d+)[x×*](\d+)',
            r'(\d+)[x×*](\d+)',
            r'[Cc](\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                if len(match.groups()) == 2:
                    return f"C{match.group(1)}x{match.group(2)}"
                else:
                    return f"C{match.group(1)}"
        
        return text[:20]  # بازگشت قسمتی از متن
    
    def _extract_beam_size(self, text: str) -> str:
        """استخراج اندازه تیر از متن"""
        if not text:
            return "B"
        
        import re
        patterns = [
            r'[Bb][-_]?(\d+)[x×*](\d+)',
            r'(\d+)[x×*](\d+)',
            r'[Bb](\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                if len(match.groups()) == 2:
                    return f"B{match.group(1)}x{match.group(2)}"
                else:
                    return f"B{match.group(1)}"
        
        return text[:20]
    
    def _extract_slab_thickness(self, text: str) -> int:
        """استخراج ضخامت دال از متن (میلی‌متر)"""
        if not text:
            return 0
        
        import re
        # الگوهای معمول: t=200, 200mm, ضخامت 20, S200
        patterns = [
            r't[=\s]?(\d+)',
            r'(\d+)\s*mm',
            r'[Ss](\d+)',
            r'ضخامت\s*(\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return int(match.group(1))
        
        return 0
    
    def _parse_structural_dimensions(self, text: str) -> Tuple[float, float, float]:
        """استخراج ابعاد سازه‌ای از متن (width, depth, height/length)"""
        if not text:
            return (0, 0, 0)
        
        import re
        # الگوی: 30x40 یا 300*400
        pattern = r'(\d+)[x×*](\d+)(?:[x×*](\d+))?'
        match = re.search(pattern, text)
        
        if match:
            width = float(match.group(1))
            depth = float(match.group(2))
            height = float(match.group(3)) if match.group(3) else 0
            return (width, depth, height)
        
        return (0, 0, 0)
    
    def _detect_mep_elements(self) -> List[MEPElementInfo]:
        """
        تشخیص عناصر تأسیسات MEP (لوله‌کشی، HVAC، برق، روشنایی)
        
        Returns:
            لیست عناصر تأسیساتی شناسایی شده
        """
        elements = []
        
        # تشخیص لوله‌کشی و سیستم‌های آب
        elements.extend(self._detect_plumbing())
        
        # تشخیص سیستم HVAC
        elements.extend(self._detect_hvac())
        
        # تشخیص سیستم برق
        elements.extend(self._detect_electrical())
        
        # تشخیص روشنایی
        elements.extend(self._detect_lighting())
        
        # تشخیص سیستم اطفاء حریق
        elements.extend(self._detect_fire_protection())
        
        return elements
    
    def _detect_plumbing(self) -> List[MEPElementInfo]:
        """تشخیص لوله‌های آب و فاضلاب"""
        plumbing = []
        
        for entity in self.msp:
            layer_name_upper = entity.dxf.layer.upper()
            is_plumbing_layer = any(p_layer.upper() in layer_name_upper for p_layer in self.PLUMBING_LAYERS)
            
            # تشخیص لوله‌ها (خطوط و polyline‌ها)
            if (entity.dxftype() in ['LINE', 'LWPOLYLINE']) and is_plumbing_layer:
                try:
                    if entity.dxftype() == 'LINE':
                        start = entity.dxf.start
                        end = entity.dxf.end
                        center_x = (start.x + end.x) / 2
                        center_y = (start.y + end.y) / 2
                        length = math.hypot(end.x - start.x, end.y - start.y)
                    else:  # LWPOLYLINE
                        points = list(entity.get_points('xy'))
                        if len(points) < 2:
                            continue
                        center_x = sum(p[0] for p in points) / len(points)
                        center_y = sum(p[1] for p in points) / len(points)
                        length = sum(math.hypot(points[i+1][0]-points[i][0], points[i+1][1]-points[i][1]) 
                                   for i in range(len(points)-1))
                    
                    # جستجوی برچسب نزدیک
                    label = self._find_nearby_text((center_x, center_y, 0), search_radius=500)
                    
                    # تشخیص نوع لوله (آب، فاضلاب، تهویه)
                    if label and any(kw in label.upper() for kw in ["CW", "COLD", "سرد", "آب سرد"]):
                        elem_type = MEPElementType.WATER_PIPE
                        system = "CW"
                    elif label and any(kw in label.upper() for kw in ["HW", "HOT", "گرم", "آب گرم"]):
                        elem_type = MEPElementType.WATER_PIPE
                        system = "HW"
                    elif label and any(kw in label.upper() for kw in ["DRAIN", "فاضلاب", "SEWER"]):
                        elem_type = MEPElementType.DRAIN_PIPE
                        system = "DRAIN"
                    elif label and any(kw in label.upper() for kw in ["VENT", "تهویه"]):
                        elem_type = MEPElementType.VENT_PIPE
                        system = "VENT"
                    else:
                        elem_type = MEPElementType.WATER_PIPE
                        system = None
                    
                    # استخراج قطر لوله
                    size = self._extract_pipe_size(label) if label else "Ø??"
                    
                    plumbing.append(MEPElementInfo(
                        element_type=elem_type,
                        layer=entity.dxf.layer,
                        position=(center_x, center_y, 0),
                        size_designation=size,
                        system=system
                    ))
                except Exception as e:
                    continue
            
            # تشخیص شیرآلات و تجهیزات (از بلوک‌ها یا دایره‌ها)
            elif entity.dxftype() in ['CIRCLE', 'INSERT'] and is_plumbing_layer:
                try:
                    if entity.dxftype() == 'CIRCLE':
                        pos = entity.dxf.center
                    else:  # INSERT (block)
                        pos = entity.dxf.insert
                    
                    label = self._find_nearby_text((pos.x, pos.y, 0), search_radius=500)
                    
                    # تشخیص نوع تجهیز
                    if label:
                        label_upper = label.upper()
                        if any(kw in label_upper for kw in ["VALVE", "شیر"]):
                            elem_type = MEPElementType.VALVE
                        elif any(kw in label_upper for kw in ["PUMP", "پمپ"]):
                            elem_type = MEPElementType.PUMP
                        elif any(kw in label_upper for kw in ["TANK", "مخزن"]):
                            elem_type = MEPElementType.TANK
                        elif any(kw in label_upper for kw in ["HEATER", "آبگرمکن"]):
                            elem_type = MEPElementType.WATER_HEATER
                        elif any(kw in label_upper for kw in ["WC", "TOILET", "SINK", "توالت", "سینک"]):
                            elem_type = MEPElementType.FIXTURE
                        else:
                            continue
                        
                        plumbing.append(MEPElementInfo(
                            element_type=elem_type,
                            layer=entity.dxf.layer,
                            position=(pos.x, pos.y, 0),
                            size_designation=label[:30]
                        ))
                except:
                    continue
        
        return plumbing
    
    def _detect_hvac(self) -> List[MEPElementInfo]:
        """تشخیص سیستم‌های HVAC (کانال‌ها، دریچه‌ها، تجهیزات)"""
        hvac = []
        
        for entity in self.msp:
            layer_name_upper = entity.dxf.layer.upper()
            is_hvac_layer = any(h_layer.upper() in layer_name_upper for h_layer in self.HVAC_LAYERS)
            
            # تشخیص کانال‌های هوا (مستطیل‌ها)
            if entity.dxftype() == 'LWPOLYLINE' and is_hvac_layer:
                try:
                    if not entity.is_closed:
                        # کانال‌ها معمولاً مستطیل‌های بسته هستند یا خطوط مسیر
                        points = list(entity.get_points('xy'))
                        if len(points) >= 2:
                            center_x = sum(p[0] for p in points) / len(points)
                            center_y = sum(p[1] for p in points) / len(points)
                            
                            label = self._find_nearby_text((center_x, center_y, 0), search_radius=800)
                            size = self._extract_duct_size(label) if label else "??"
                            
                            hvac.append(MEPElementInfo(
                                element_type=MEPElementType.DUCT,
                                layer=entity.dxf.layer,
                                position=(center_x, center_y, 0),
                                size_designation=size,
                                system="HVAC"
                            ))
                    else:
                        # مستطیل بسته - ممکن است AHU یا تجهیزات باشد
                        points = list(entity.get_points('xy'))
                        xs = [p[0] for p in points]
                        ys = [p[1] for p in points]
                        center_x = sum(xs) / len(xs)
                        center_y = sum(ys) / len(ys)
                        
                        label = self._find_nearby_text((center_x, center_y, 0), search_radius=1000)
                        if label:
                            label_upper = label.upper()
                            if "AHU" in label_upper or "هواساز" in label:
                                elem_type = MEPElementType.AIR_HANDLER
                            elif "FCU" in label_upper or "فن کویل" in label:
                                elem_type = MEPElementType.FCU
                            elif "VAV" in label_upper:
                                elem_type = MEPElementType.VAV
                            else:
                                elem_type = MEPElementType.DUCT
                            
                            hvac.append(MEPElementInfo(
                                element_type=elem_type,
                                layer=entity.dxf.layer,
                                position=(center_x, center_y, 0),
                                size_designation=label[:30]
                            ))
                except:
                    continue
            
            # تشخیص دریچه‌ها و گریل‌ها (دایره‌ها و مربع‌های کوچک)
            elif entity.dxftype() in ['CIRCLE', 'INSERT'] and is_hvac_layer:
                try:
                    pos = entity.dxf.center if entity.dxftype() == 'CIRCLE' else entity.dxf.insert
                    label = self._find_nearby_text((pos.x, pos.y, 0), search_radius=500)
                    
                    if label:
                        label_upper = label.upper()
                        if "DIFFUSER" in label_upper or "دریچه" in label:
                            elem_type = MEPElementType.DIFFUSER
                        elif "GRILLE" in label_upper or "گریل" in label:
                            elem_type = MEPElementType.GRILLE
                        elif "FAN" in label_upper or "فن" in label:
                            elem_type = MEPElementType.FAN
                        else:
                            elem_type = MEPElementType.DIFFUSER
                        
                        hvac.append(MEPElementInfo(
                            element_type=elem_type,
                            layer=entity.dxf.layer,
                            position=(pos.x, pos.y, 0),
                            size_designation=label[:30]
                        ))
                except:
                    continue
        
        return hvac
    
    def _detect_electrical(self) -> List[MEPElementInfo]:
        """تشخیص سیستم برق (تابلو، پریز، کلید، کابل)"""
        electrical = []
        
        for entity in self.msp:
            layer_name_upper = entity.dxf.layer.upper()
            is_electrical_layer = any(e_layer.upper() in layer_name_upper for e_layer in self.ELECTRICAL_LAYERS)
            
            # تشخیص کابل‌ها (خطوط)
            if entity.dxftype() in ['LINE', 'LWPOLYLINE'] and is_electrical_layer:
                try:
                    if entity.dxftype() == 'LINE':
                        start = entity.dxf.start
                        end = entity.dxf.end
                        center_x = (start.x + end.x) / 2
                        center_y = (start.y + end.y) / 2
                    else:
                        points = list(entity.get_points('xy'))
                        if len(points) < 2:
                            continue
                        center_x = sum(p[0] for p in points) / len(points)
                        center_y = sum(p[1] for p in points) / len(points)
                    
                    label = self._find_nearby_text((center_x, center_y, 0), search_radius=500)
                    
                    electrical.append(MEPElementInfo(
                        element_type=MEPElementType.CABLE,
                        layer=entity.dxf.layer,
                        position=(center_x, center_y, 0),
                        size_designation=label[:30] if label else "Cable"
                    ))
                except:
                    continue
            
            # تشخیص تجهیزات برقی (سیمبل‌ها)
            elif entity.dxftype() in ['CIRCLE', 'INSERT', 'TEXT'] and is_electrical_layer:
                try:
                    if entity.dxftype() == 'CIRCLE':
                        pos = entity.dxf.center
                    elif entity.dxftype() == 'INSERT':
                        pos = entity.dxf.insert
                    else:  # TEXT
                        pos = entity.dxf.insert
                        text_content = entity.dxf.text
                        # فقط متن‌هایی که سیمبل هستند
                        if not any(kw in text_content.upper() for kw in self.ELECTRICAL_KEYWORDS):
                            continue
                    
                    label = self._find_nearby_text((pos.x, pos.y, 0), search_radius=500)
                    if not label and entity.dxftype() == 'TEXT':
                        label = text_content
                    
                    if label:
                        label_upper = label.upper()
                        # تشخیص نوع تجهیز
                        if any(kw in label_upper for kw in ["PANEL", "تابلو", "DB", "MDB"]):
                            elem_type = MEPElementType.PANEL
                        elif any(kw in label_upper for kw in ["OUTLET", "پریز", "SOCKET"]):
                            elem_type = MEPElementType.OUTLET
                        elif any(kw in label_upper for kw in ["SWITCH", "کلید", "SW"]):
                            elem_type = MEPElementType.SWITCH
                        elif any(kw in label_upper for kw in ["TRANS", "ترانس"]):
                            elem_type = MEPElementType.TRANSFORMER
                        elif any(kw in label_upper for kw in ["GEN", "ژنراتور"]):
                            elem_type = MEPElementType.GENERATOR
                        elif any(kw in label_upper for kw in ["UPS"]):
                            elem_type = MEPElementType.UPS
                        else:
                            continue
                        
                        # استخراج ولتاژ و توان
                        voltage = self._extract_voltage(label)
                        power = self._extract_power(label)
                        
                        electrical.append(MEPElementInfo(
                            element_type=elem_type,
                            layer=entity.dxf.layer,
                            position=(pos.x, pos.y, 0),
                            size_designation=label[:30],
                            voltage=voltage,
                            power=power
                        ))
                except:
                    continue
        
        return electrical
    
    def _detect_lighting(self) -> List[MEPElementInfo]:
        """تشخیص سیستم روشنایی (چراغ‌ها)"""
        lighting = []
        
        for entity in self.msp:
            layer_name_upper = entity.dxf.layer.upper()
            is_lighting_layer = any(l_layer.upper() in layer_name_upper for l_layer in self.LIGHTING_LAYERS)
            
            if entity.dxftype() in ['CIRCLE', 'INSERT', 'POINT'] and is_lighting_layer:
                try:
                    if entity.dxftype() == 'CIRCLE':
                        pos = entity.dxf.center
                    elif entity.dxftype() == 'INSERT':
                        pos = entity.dxf.insert
                    else:  # POINT
                        pos = entity.dxf.location
                    
                    label = self._find_nearby_text((pos.x, pos.y, 0), search_radius=500)
                    power = self._extract_power(label) if label else None
                    
                    lighting.append(MEPElementInfo(
                        element_type=MEPElementType.LIGHT_FIXTURE,
                        layer=entity.dxf.layer,
                        position=(pos.x, pos.y, 0),
                        size_designation=label[:30] if label else "Light",
                        power=power
                    ))
                except:
                    continue
        
        return lighting
    
    def _detect_fire_protection(self) -> List[MEPElementInfo]:
        """تشخیص سیستم اطفاء حریق (اسپرینکلر، آلارم، دتکتور)"""
        fire_protection = []
        
        for entity in self.msp:
            layer_name_upper = entity.dxf.layer.upper()
            is_fire_layer = any(f_layer.upper() in layer_name_upper for f_layer in self.FIRE_LAYERS)
            
            if entity.dxftype() in ['CIRCLE', 'INSERT', 'POINT'] and is_fire_layer:
                try:
                    if entity.dxftype() == 'CIRCLE':
                        pos = entity.dxf.center
                    elif entity.dxftype() == 'INSERT':
                        pos = entity.dxf.insert
                    else:  # POINT
                        pos = entity.dxf.location
                    
                    label = self._find_nearby_text((pos.x, pos.y, 0), search_radius=500)
                    
                    if label:
                        label_upper = label.upper()
                        if any(kw in label_upper for kw in ["SPRINKLER", "اسپرینکلر", "SPK"]):
                            elem_type = MEPElementType.SPRINKLER
                        elif any(kw in label_upper for kw in ["ALARM", "آلارم", "اعلام"]):
                            elem_type = MEPElementType.FIRE_ALARM
                        elif any(kw in label_upper for kw in ["SMOKE", "دود", "DETECTOR"]):
                            elem_type = MEPElementType.SMOKE_DETECTOR
                        elif any(kw in label_upper for kw in ["EXTINGUISHER", "کپسول"]):
                            elem_type = MEPElementType.FIRE_EXTINGUISHER
                        elif any(kw in label_upper for kw in ["HOSE", "شیلنگ"]):
                            elem_type = MEPElementType.FIRE_HOSE
                        else:
                            elem_type = MEPElementType.SPRINKLER  # default
                        
                        fire_protection.append(MEPElementInfo(
                            element_type=elem_type,
                            layer=entity.dxf.layer,
                            position=(pos.x, pos.y, 0),
                            size_designation=label[:30]
                        ))
                except:
                    continue
        
        return fire_protection
    
    def _detect_construction_details(self) -> List[ConstructionDetailInfo]:
        """تشخیص جزئیات اجرایی"""
        details = []
        
        # جمع‌آوری تمام متن‌ها و موقعیت‌های آنها
        texts_with_position = []
        for entity in self.msp:
            if entity.dxftype() in ["TEXT", "MTEXT"]:
                try:
                    text_content = entity.dxf.text if entity.dxftype() == "TEXT" else entity.text
                    position = entity.dxf.insert if hasattr(entity.dxf, 'insert') else (0, 0, 0)
                    layer = entity.dxf.layer if hasattr(entity.dxf, 'layer') else ""
                    texts_with_position.append({
                        'text': text_content.lower(),
                        'position': position,
                        'layer': layer,
                        'entity': entity
                    })
                except:
                    continue
        
        # بررسی هر متن برای شناسایی نوع جزئیات
        for text_info in texts_with_position:
            text_lower = text_info['text']
            position = text_info['position']
            layer = text_info['layer']
            
            # تشخیص نوع جزئیات بر اساس کلمات کلیدی
            detail_type = self._identify_detail_type(text_lower)
            
            if detail_type != DetailType.UNKNOWN:
                # استخراج مقیاس
                scale = self._extract_scale(text_lower)
                
                # تشخیص مصالح
                materials = self._identify_materials(text_lower)
                
                # جمع‌آوری یادداشت‌های نزدیک
                annotations = self._collect_nearby_annotations(position, texts_with_position)
                
                details.append(ConstructionDetailInfo(
                    detail_type=detail_type,
                    layer=layer,
                    position=position,
                    scale=scale,
                    materials=materials,
                    annotations=annotations[:5],  # حداکثر 5 یادداشت
                ))
        
        # تشخیص جزئیات بر اساس لایه
        for layer in self.doc.layers:
            layer_name = layer.dxf.name
            layer_upper = layer_name.upper()
            
            if any(keyword in layer_upper for keyword in self.DETAIL_LAYERS):
                # شمارش entity ها در این لایه
                entities_in_layer = [e for e in self.msp if hasattr(e.dxf, 'layer') and e.dxf.layer == layer_name]
                
                if len(entities_in_layer) > 0:
                    # اگر جزئیات مشخصی تشخیص نداده‌ایم، یک جزئیات عمومی اضافه کن
                    if not any(d.layer == layer_name for d in details):
                        # میانگین موقعیت entity ها
                        avg_x = avg_y = 0
                        count = 0
                        for e in entities_in_layer[:10]:  # نمونه از 10 entity اول
                            try:
                                if hasattr(e.dxf, 'insert'):
                                    avg_x += e.dxf.insert.x
                                    avg_y += e.dxf.insert.y
                                    count += 1
                            except:
                                continue
                        
                        if count > 0:
                            details.append(ConstructionDetailInfo(
                                detail_type=DetailType.UNKNOWN,
                                layer=layer_name,
                                position=(avg_x/count, avg_y/count, 0),
                                annotations=[f"تعداد عناصر: {len(entities_in_layer)}"]
                            ))
        
        return details
    
    def _identify_detail_type(self, text: str) -> DetailType:
        """تشخیص نوع جزئیات از روی متن"""
        text_lower = text.lower()
        
        # بررسی اتصالات
        if any(kw in text_lower for kw in ["beam", "column", "تیر", "ستون", "connection"]):
            if any(kw in text_lower for kw in ["beam", "تیر"]) and any(kw in text_lower for kw in ["column", "ستون"]):
                return DetailType.BEAM_COLUMN_CONNECTION
        
        # بررسی درب و پنجره
        if any(kw in text_lower for kw in self.DETAIL_KEYWORDS["door"]):
            if any(kw in text_lower for kw in ["frame", "چارچوب", "قاب"]):
                return DetailType.DOOR_FRAME
            return DetailType.DOOR_DETAIL
        
        if any(kw in text_lower for kw in self.DETAIL_KEYWORDS["window"]):
            if any(kw in text_lower for kw in ["frame", "چارچوب", "قاب"]):
                return DetailType.WINDOW_FRAME
            return DetailType.WINDOW_DETAIL
        
        # بررسی نما
        if any(kw in text_lower for kw in self.DETAIL_KEYWORDS["facade"]):
            if any(kw in text_lower for kw in ["stone", "سنگ"]):
                return DetailType.STONE_CLADDING
            elif any(kw in text_lower for kw in ["curtain", "پرده"]):
                return DetailType.CURTAIN_WALL
            elif any(kw in text_lower for kw in ["panel", "پانل"]):
                return DetailType.FACADE_PANEL
            return DetailType.CLADDING
        
        # بررسی آب‌بندی
        if any(kw in text_lower for kw in self.DETAIL_KEYWORDS["waterproofing"]):
            return DetailType.WATERPROOFING
        
        # بررسی عایق
        if any(kw in text_lower for kw in self.DETAIL_KEYWORDS["insulation"]):
            return DetailType.INSULATION
        
        # بررسی کف
        if any(kw in text_lower for kw in self.DETAIL_KEYWORDS["flooring"]):
            return DetailType.FLOOR_FINISH
        
        # بررسی سقف
        if any(kw in text_lower for kw in self.DETAIL_KEYWORDS["ceiling"]):
            if any(kw in text_lower for kw in ["false", "کاذب"]):
                return DetailType.FALSE_CEILING
            return DetailType.CEILING_DETAIL
        
        # بررسی پله و نرده
        if any(kw in text_lower for kw in self.DETAIL_KEYWORDS["stair"]):
            return DetailType.STAIR_DETAIL
        
        if any(kw in text_lower for kw in self.DETAIL_KEYWORDS["railing"]):
            return DetailType.RAILING_DETAIL
        
        # بررسی سقف/بام
        if any(kw in text_lower for kw in self.DETAIL_KEYWORDS["roof"]):
            if any(kw in text_lower for kw in ["edge", "لبه"]):
                return DetailType.ROOF_EDGE
            elif any(kw in text_lower for kw in ["gutter", "ناودان"]):
                return DetailType.GUTTER
            return DetailType.ROOF_SECTION
        
        # بررسی دیوار
        if any(kw in text_lower for kw in self.DETAIL_KEYWORDS["wall"]):
            if any(kw in text_lower for kw in ["partition", "پارتیشن"]):
                return DetailType.PARTITION_WALL
            return DetailType.WALL_SECTION
        
        return DetailType.UNKNOWN
    
    def _identify_materials(self, text: str) -> List[MaterialType]:
        """تشخیص مصالح از روی متن"""
        materials = []
        text_lower = text.lower()
        
        for material_type, keywords in self.MATERIAL_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                materials.append(material_type)
        
        return materials if materials else None
    
    def _extract_scale(self, text: str) -> Optional[str]:
        """استخراج مقیاس نقشه"""
        import re
        # الگوهای معمول: 1:5, 1:10, 1:20, SCALE 1:10
        pattern = r'1\s*:\s*(\d+)'
        match = re.search(pattern, text)
        
        if match:
            return f"1:{match.group(1)}"
        
        return None
    
    def _collect_nearby_annotations(self, position: Tuple[float, float, float], 
                                     all_texts: List[Dict], max_distance: float = 500) -> List[str]:
        """جمع‌آوری یادداشت‌های نزدیک به یک نقطه"""
        annotations = []
        px, py, pz = position
        
        for text_info in all_texts:
            tx, ty, tz = text_info['position']
            distance = math.sqrt((tx - px)**2 + (ty - py)**2)
            
            if distance < max_distance and distance > 1:  # نه خیلی نزدیک (خود متن)
                text_content = text_info['text']
                if len(text_content) > 3:  # حداقل 3 کاراکتر
                    annotations.append(text_content[:100])  # حداکثر 100 کاراکتر
        
        return annotations
    
    def _detect_site_elements(self) -> List[SiteElementInfo]:
        """تشخیص عناصر نقشه سایت و موقعیت"""
        site_elements = []
        
        # جمع‌آوری متن‌ها
        texts_with_position = []
        for entity in self.msp:
            if entity.dxftype() in ["TEXT", "MTEXT"]:
                try:
                    text_content = entity.dxf.text if entity.dxftype() == "TEXT" else entity.text
                    position = entity.dxf.insert if hasattr(entity.dxf, 'insert') else (0, 0, 0)
                    layer = entity.dxf.layer if hasattr(entity.dxf, 'layer') else ""
                    texts_with_position.append({
                        'text': text_content.lower(),
                        'position': position,
                        'layer': layer
                    })
                except:
                    continue
        
        # تشخیص ساختمان اصلی و مجاور
        buildings = self._detect_buildings()
        site_elements.extend(buildings)
        
        # محاسبه مرکز ساختمان اصلی
        main_building_center = None
        for building in buildings:
            if building.element_type == SiteElementType.MAIN_BUILDING:
                if building.boundary_vertices:
                    xs = [v[0] for v in building.boundary_vertices]
                    ys = [v[1] for v in building.boundary_vertices]
                    main_building_center = ((min(xs) + max(xs))/2, (min(ys) + max(ys))/2, 0)
                break
        
        # تشخیص مرزها و خطوط ملک
        boundaries = self._detect_boundaries()
        site_elements.extend(boundaries)
        
        # تشخیص جاده‌ها و مسیرها
        roads = self._detect_roads_and_access()
        site_elements.extend(roads)
        
        # تشخیص پارکینگ
        parking = self._detect_parking()
        site_elements.extend(parking)
        
        # تشخیص فضای سبز و درختان
        landscape = self._detect_landscape(texts_with_position)
        site_elements.extend(landscape)
        
        # تشخیص عناصر دیگر (حصار، استخر، ...)
        other_elements = self._detect_other_site_elements(texts_with_position)
        site_elements.extend(other_elements)
        
        # محاسبه فاصله تا ساختمان اصلی
        if main_building_center:
            for elem in site_elements:
                if elem.element_type != SiteElementType.MAIN_BUILDING:
                    dx = elem.position[0] - main_building_center[0]
                    dy = elem.position[1] - main_building_center[1]
                    elem.distance_to_main = math.sqrt(dx*dx + dy*dy)
        
        return site_elements
    
    def _detect_buildings(self) -> List[SiteElementInfo]:
        """تشخیص ساختمان‌ها"""
        buildings = []
        building_layers = ["BUILDING", "BLDG", "B-", "ساختمان", "بنا", "BUILD"]
        
        for entity in self.msp:
            if entity.dxftype() == "LWPOLYLINE":
                layer = entity.dxf.layer if hasattr(entity.dxf, 'layer') else ""
                
                # بررسی لایه ساختمان
                is_building_layer = any(kw.upper() in layer.upper() for kw in building_layers)
                
                if is_building_layer:
                    try:
                        points = list(entity.get_points(format='xy'))
                        if len(points) >= 3:
                            # بررسی اینکه بسته است (یا نقطه اول و آخر یکسان است)
                            is_closed = entity.is_closed or (len(points) >= 4 and points[0] == points[-1])
                            if is_closed:
                                area = abs(self._calculate_area(points))
                                # حداقل مساحت 1 متر مربع (1,000,000 mm²)
                                if area > 1_000_000:
                                    xs = [p[0] for p in points]
                                    ys = [p[1] for p in points]
                                    center = ((min(xs) + max(xs))/2, (min(ys) + max(ys))/2, 0)
                                    
                                    # تعیین نوع (اصلی یا مجاور)
                                    elem_type = SiteElementType.MAIN_BUILDING
                                    if "EXIST" in layer.upper() or "موجود" in layer:
                                        elem_type = SiteElementType.EXISTING_BUILDING
                                    elif "ADJACENT" in layer.upper() or "مجاور" in layer:
                                        elem_type = SiteElementType.ADJACENT_BUILDING
                                    
                                    buildings.append(SiteElementInfo(
                                        element_type=elem_type,
                                        layer=layer,
                                        position=center,
                                        geometry_type="polygon",
                                        area=area / 1_000_000,  # تبدیل به متر مربع
                                        boundary_vertices=points,
                                        label=layer
                                    ))
                    except Exception as e:
                        continue
        
        return buildings
    
    def _detect_boundaries(self) -> List[SiteElementInfo]:
        """تشخیص مرزها و خطوط ملک"""
        boundaries = []
        boundary_layers = ["PROPERTY", "BOUNDARY", "LIMIT", "مرز", "ملک"]
        
        for entity in self.msp:
            layer = entity.dxf.layer if hasattr(entity.dxf, 'layer') else ""
            
            if any(kw in layer.upper() for kw in boundary_layers):
                if entity.dxftype() == "LINE":
                    try:
                        start = (entity.dxf.start.x, entity.dxf.start.y, entity.dxf.start.z)
                        end = (entity.dxf.end.x, entity.dxf.end.y, entity.dxf.end.z)
                        length = math.sqrt((end[0]-start[0])**2 + (end[1]-start[1])**2)
                        
                        boundaries.append(SiteElementInfo(
                            element_type=SiteElementType.PROPERTY_LINE,
                            layer=layer,
                            position=start,
                            geometry_type="line",
                            length=length / 1000,  # به متر
                        ))
                    except:
                        continue
                
                elif entity.dxftype() == "LWPOLYLINE":
                    try:
                        points = list(entity.get_points(format='xy'))
                        if len(points) >= 2:
                            total_length = 0
                            for i in range(len(points)-1):
                                dx = points[i+1][0] - points[i][0]
                                dy = points[i+1][1] - points[i][1]
                                total_length += math.sqrt(dx*dx + dy*dy)
                            
                            boundaries.append(SiteElementInfo(
                                element_type=SiteElementType.PROPERTY_LINE,
                                layer=layer,
                                position=(points[0][0], points[0][1], 0),
                                geometry_type="polyline",
                                length=total_length / 1000,
                                boundary_vertices=points
                            ))
                    except:
                        continue
        
        return boundaries
    
    def _detect_roads_and_access(self) -> List[SiteElementInfo]:
        """تشخیص جاده‌ها و مسیرهای دسترسی"""
        roads = []
        road_layers = ["ROAD", "STREET", "DRIVE", "ACCESS", "جاده", "خیابان", "مسیر"]
        
        for entity in self.msp:
            layer = entity.dxf.layer if hasattr(entity.dxf, 'layer') else ""
            
            if any(kw in layer.upper() for kw in road_layers):
                if entity.dxftype() == "LWPOLYLINE":
                    try:
                        points = list(entity.get_points(format='xy'))
                        if len(points) >= 2:
                            total_length = 0
                            for i in range(len(points)-1):
                                dx = points[i+1][0] - points[i][0]
                                dy = points[i+1][1] - points[i][1]
                                total_length += math.sqrt(dx*dx + dy*dy)
                            
                            elem_type = SiteElementType.ROAD
                            if "DRIVE" in layer.upper() or "ورودی" in layer:
                                elem_type = SiteElementType.DRIVEWAY
                            elif "WALK" in layer.upper() or "پیاده" in layer:
                                elem_type = SiteElementType.WALKWAY
                            
                            roads.append(SiteElementInfo(
                                element_type=elem_type,
                                layer=layer,
                                position=(points[0][0], points[0][1], 0),
                                geometry_type="polyline",
                                length=total_length / 1000,
                                boundary_vertices=points
                            ))
                    except:
                        continue
        
        return roads
    
    def _detect_parking(self) -> List[SiteElementInfo]:
        """تشخیص پارکینگ"""
        parking = []
        parking_layers = ["PARKING", "PARK", "P-", "پارکینگ"]
        
        for entity in self.msp:
            layer = entity.dxf.layer if hasattr(entity.dxf, 'layer') else ""
            
            if any(kw in layer.upper() for kw in parking_layers):
                if entity.dxftype() == "LWPOLYLINE":
                    try:
                        points = list(entity.get_points(format='xy'))
                        if len(points) >= 3:
                            # بررسی اینکه بسته است
                            is_closed = entity.is_closed or (points[0] == points[-1])
                            if is_closed:
                                area = self._calculate_area(points)
                                xs = [p[0] for p in points]
                                ys = [p[1] for p in points]
                                center = ((min(xs) + max(xs))/2, (min(ys) + max(ys))/2, 0)
                                
                                # تشخیص نوع (محوطه یا جای پارک منفرد)
                                elem_type = SiteElementType.PARKING_LOT
                                if area < 50_000_000:  # کمتر از 50 متر مربع
                                    elem_type = SiteElementType.PARKING_SPACE
                                
                                parking.append(SiteElementInfo(
                                    element_type=elem_type,
                                    layer=layer,
                                    position=center,
                                    geometry_type="polygon",
                                    area=area / 1_000_000,
                                    boundary_vertices=points
                                ))
                    except:
                        continue
        
        return parking
    
    def _detect_landscape(self, texts: List[Dict]) -> List[SiteElementInfo]:
        """تشخیص فضای سبز و درختان"""
        landscape = []
        
        # تشخیص درختان (معمولاً دایره)
        tree_layers = ["TREE", "PLANT", "درخت", "LANDSCAPE", "L-"]
        for entity in self.msp:
            if entity.dxftype() == "CIRCLE":
                layer = entity.dxf.layer if hasattr(entity.dxf, 'layer') else ""
                
                if any(kw in layer.upper() for kw in tree_layers):
                    try:
                        center = entity.dxf.center
                        radius = entity.dxf.radius
                        
                        landscape.append(SiteElementInfo(
                            element_type=SiteElementType.TREE,
                            layer=layer,
                            position=(center.x, center.y, center.z),
                            geometry_type="circle",
                            width=radius * 2 / 1000  # قطر به متر
                        ))
                    except:
                        continue
        
        # تشخیص فضای سبز (پلی‌گون)
        grass_layers = ["GRASS", "LAWN", "GREEN", "چمن", "فضای سبز"]
        for entity in self.msp:
            layer = entity.dxf.layer if hasattr(entity.dxf, 'layer') else ""
            
            if any(kw in layer.upper() for kw in grass_layers):
                if entity.dxftype() == "LWPOLYLINE":
                    try:
                        points = list(entity.get_points(format='xy'))
                        if len(points) >= 3:
                            # بررسی اینکه بسته است
                            is_closed = entity.is_closed or (points[0] == points[-1])
                            if is_closed:
                                area = self._calculate_area(points)
                                xs = [p[0] for p in points]
                                ys = [p[1] for p in points]
                                center = ((min(xs) + max(xs))/2, (min(ys) + max(ys))/2, 0)
                                
                                landscape.append(SiteElementInfo(
                                    element_type=SiteElementType.GRASS_AREA,
                                    layer=layer,
                                    position=center,
                                    geometry_type="polygon",
                                    area=area / 1_000_000,
                                    boundary_vertices=points
                                ))
                    except:
                        continue
        
        return landscape
    
    def _detect_other_site_elements(self, texts: List[Dict]) -> List[SiteElementInfo]:
        """تشخیص سایر عناصر سایت"""
        elements = []
        
        # تشخیص حصار
        fence_layers = ["FENCE", "حصار", "WALL"]
        for entity in self.msp:
            layer = entity.dxf.layer if hasattr(entity.dxf, 'layer') else ""
            
            if any(kw in layer.upper() for kw in fence_layers) and "RETAINING" not in layer.upper():
                if entity.dxftype() == "LINE":
                    try:
                        start = (entity.dxf.start.x, entity.dxf.start.y, entity.dxf.start.z)
                        end = (entity.dxf.end.x, entity.dxf.end.y, entity.dxf.end.z)
                        length = math.sqrt((end[0]-start[0])**2 + (end[1]-start[1])**2)
                        
                        elements.append(SiteElementInfo(
                            element_type=SiteElementType.FENCE,
                            layer=layer,
                            position=start,
                            geometry_type="line",
                            length=length / 1000
                        ))
                    except:
                        continue
                elif entity.dxftype() == "LWPOLYLINE":
                    try:
                        points = list(entity.get_points(format='xy'))
                        if len(points) >= 2:
                            total_length = 0
                            for i in range(len(points)-1):
                                dx = points[i+1][0] - points[i][0]
                                dy = points[i+1][1] - points[i][1]
                                total_length += math.sqrt(dx*dx + dy*dy)
                            
                            elements.append(SiteElementInfo(
                                element_type=SiteElementType.FENCE,
                                layer=layer,
                                position=(points[0][0], points[0][1], 0),
                                geometry_type="polyline",
                                length=total_length / 1000
                            ))
                    except:
                        continue
        
        # تشخیص استخر
        pool_layers = ["POOL", "WATER", "استخر"]
        for entity in self.msp:
            layer = entity.dxf.layer if hasattr(entity.dxf, 'layer') else ""
            
            if any(kw in layer.upper() for kw in pool_layers):
                if entity.dxftype() == "LWPOLYLINE":
                    try:
                        points = list(entity.get_points(format='xy'))
                        if len(points) >= 3:
                            # بررسی اینکه بسته است
                            is_closed = entity.is_closed or (points[0] == points[-1])
                            if is_closed:
                                area = self._calculate_area(points)
                                xs = [p[0] for p in points]
                                ys = [p[1] for p in points]
                                center = ((min(xs) + max(xs))/2, (min(ys) + max(ys))/2, 0)
                                
                                elements.append(SiteElementInfo(
                                    element_type=SiteElementType.POOL,
                                    layer=layer,
                                    position=center,
                                    geometry_type="polygon",
                                    area=area / 1_000_000,
                                    boundary_vertices=points
                                ))
                    except:
                        continue
        
        # تشخیص تأسیسات (خط، منهول، چراغ)
        utility_layers = ["UTILITY", "تاسیسات", "برق"]
        for entity in self.msp:
            layer = entity.dxf.layer if hasattr(entity.dxf, 'layer') else ""
            
            if any(kw in layer.upper() for kw in utility_layers):
                if entity.dxftype() == "LINE":
                    try:
                        start = (entity.dxf.start.x, entity.dxf.start.y, entity.dxf.start.z)
                        end = (entity.dxf.end.x, entity.dxf.end.y, entity.dxf.end.z)
                        length = math.sqrt((end[0]-start[0])**2 + (end[1]-start[1])**2)
                        
                        elements.append(SiteElementInfo(
                            element_type=SiteElementType.UTILITY_LINE,
                            layer=layer,
                            position=start,
                            geometry_type="line",
                            length=length / 1000
                        ))
                    except:
                        continue
        
        manhole_layers = ["MANHOLE", "MH", "منهول"]
        for entity in self.msp:
            layer = entity.dxf.layer if hasattr(entity.dxf, 'layer') else ""
            
            if any(kw in layer.upper() for kw in manhole_layers):
                if entity.dxftype() == "CIRCLE":
                    try:
                        center = entity.dxf.center
                        
                        elements.append(SiteElementInfo(
                            element_type=SiteElementType.MANHOLE,
                            layer=layer,
                            position=(center.x, center.y, center.z),
                            geometry_type="circle"
                        ))
                    except:
                        continue
        
        light_layers = ["LIGHT", "چراغ", "LAMP"]
        for entity in self.msp:
            layer = entity.dxf.layer if hasattr(entity.dxf, 'layer') else ""
            
            if any(kw in layer.upper() for kw in light_layers):
                if entity.dxftype() == "CIRCLE":
                    try:
                        center = entity.dxf.center
                        
                        elements.append(SiteElementInfo(
                            element_type=SiteElementType.LIGHT_POLE,
                            layer=layer,
                            position=(center.x, center.y, center.z),
                            geometry_type="circle"
                        ))
                    except:
                        continue
        
        # تشخیص جهت شمال
        for text_info in texts:
            text_lower = text_info['text']
            # بررسی با حروف کوچک و بزرگ
            if any(kw.lower() in text_lower for kw in self.SITE_KEYWORDS["north"]):
                elements.append(SiteElementInfo(
                    element_type=SiteElementType.NORTH_ARROW,
                    layer=text_info['layer'],
                    position=text_info['position'],
                    geometry_type="point",
                    label=text_lower[:50]
                ))
                break  # فقط یک جهت شمال
        
        return elements
    
    def _detect_civil_elements(self) -> List[CivilElementInfo]:
        """تشخیص عناصر مهندسی سایت و محوطه"""
        civil_elements = []
        
        # جمع‌آوری متن‌ها
        texts_with_position = []
        for entity in self.msp:
            if entity.dxftype() in ["TEXT", "MTEXT"]:
                try:
                    text_content = entity.dxf.text if entity.dxftype() == "TEXT" else entity.text
                    position = entity.dxf.insert if hasattr(entity.dxf, 'insert') else (0, 0, 0)
                    layer = entity.dxf.layer if hasattr(entity.dxf, 'layer') else ""
                    texts_with_position.append({
                        'text': text_content.lower(),
                        'position': position,
                        'layer': layer
                    })
                except:
                    continue
        
        # تشخیص کانتورها و شیب‌بندی
        contours = self._detect_contours()
        civil_elements.extend(contours)
        
        # تشخیص سیستم‌های زهکشی
        drainage = self._detect_drainage_systems()
        civil_elements.extend(drainage)
        
        # تشخیص لوله‌های فاضلاب و آب
        utilities = self._detect_civil_utilities()
        civil_elements.extend(utilities)
        
        # تشخیص مناطق برش و خاکریزی
        earthwork = self._detect_earthwork()
        civil_elements.extend(earthwork)
        
        return civil_elements
    
    def _detect_contours(self) -> List[CivilElementInfo]:
        """تشخیص کانتورها و خطوط ارتفاعی"""
        contours = []
        contour_layers = ["CONTOUR", "TOPO", "ELEVATION", "ELEV", "کانتور", "ارتفاع"]
        
        for entity in self.msp:
            layer = entity.dxf.layer if hasattr(entity.dxf, 'layer') else ""
            
            if any(kw in layer.upper() for kw in contour_layers):
                if entity.dxftype() in ["LINE", "LWPOLYLINE", "POLYLINE"]:
                    try:
                        if entity.dxftype() == "LINE":
                            start = (entity.dxf.start.x, entity.dxf.start.y, entity.dxf.start.z)
                            length = math.sqrt((entity.dxf.end.x - start[0])**2 + (entity.dxf.end.y - start[1])**2)
                        else:
                            points = list(entity.get_points(format='xy'))
                            if len(points) < 2:
                                continue
                            start = (points[0][0], points[0][1], 0)
                            length = sum(math.sqrt((points[i+1][0]-points[i][0])**2 + (points[i+1][1]-points[i][1])**2)
                                       for i in range(len(points)-1))
                        
                        # شناسایی ارتفاع از نام لایه یا متن‌های نزدیک
                        elevation = self._extract_elevation_from_layer(layer)
                        
                        # تشخیص نوع (اصلی یا فرعی)
                        elem_type = CivilElementType.CONTOUR_MAJOR
                        if "MINOR" in layer.upper() or "فرعی" in layer:
                            elem_type = CivilElementType.CONTOUR_MINOR
                        
                        contours.append(CivilElementInfo(
                            element_type=elem_type,
                            layer=layer,
                            position=start,
                            geometry_type="polyline",
                            length=length / 1000,  # به متر
                            elevation=elevation
                        ))
                    except:
                        continue
        
        return contours
    
    def _detect_drainage_systems(self) -> List[CivilElementInfo]:
        """تشخیص سیستم‌های زهکشی"""
        drainage = []
        drainage_layers = ["DRAIN", "DRAINAGE", "STORM", "SD", "زهکشی", "آبرو"]
        
        for entity in self.msp:
            layer = entity.dxf.layer if hasattr(entity.dxf, 'layer') else ""
            
            if any(kw in layer.upper() for kw in drainage_layers):
                if entity.dxftype() in ["LINE", "LWPOLYLINE"]:
                    try:
                        if entity.dxftype() == "LINE":
                            start = (entity.dxf.start.x, entity.dxf.start.y, entity.dxf.start.z)
                            end = (entity.dxf.end.x, entity.dxf.end.y, entity.dxf.end.z)
                            length = math.sqrt((end[0]-start[0])**2 + (end[1]-start[1])**2)
                        else:
                            points = list(entity.get_points(format='xy'))
                            if len(points) < 2:
                                continue
                            start = (points[0][0], points[0][1], 0)
                            length = sum(math.sqrt((points[i+1][0]-points[i][0])**2 + (points[i+1][1]-points[i][1])**2)
                                       for i in range(len(points)-1))
                        
                        drainage.append(CivilElementInfo(
                            element_type=CivilElementType.DRAINAGE_PIPE,
                            layer=layer,
                            position=start,
                            geometry_type="line",
                            length=length / 1000
                        ))
                    except:
                        continue
                
                # تشخیص حوضچه‌های جمع‌آوری
                elif entity.dxftype() == "CIRCLE":
                    try:
                        center = entity.dxf.center
                        
                        drainage.append(CivilElementInfo(
                            element_type=CivilElementType.CATCH_BASIN_CIVIL,
                            layer=layer,
                            position=(center.x, center.y, center.z),
                            geometry_type="circle"
                        ))
                    except:
                        continue
        
        return drainage
    
    def _detect_civil_utilities(self) -> List[CivilElementInfo]:
        """تشخیص خطوط تاسیسات (آب، فاضلاب، برق، گاز)"""
        utilities = []
        
        utility_types = {
            "WATER": CivilElementType.WATER_LINE,
            "SEWER": CivilElementType.SEWER_LINE,
            "STORM": CivilElementType.STORM_DRAIN,
            "GAS": CivilElementType.GAS_LINE,
            "ELECTRIC": CivilElementType.ELECTRIC_LINE,
            "TELECOM": CivilElementType.TELECOM_LINE,
        }
        
        for entity in self.msp:
            layer = entity.dxf.layer if hasattr(entity.dxf, 'layer') else ""
            layer_upper = layer.upper()
            
            for keyword, element_type in utility_types.items():
                if keyword in layer_upper:
                    if entity.dxftype() in ["LINE", "LWPOLYLINE"]:
                        try:
                            if entity.dxftype() == "LINE":
                                start = (entity.dxf.start.x, entity.dxf.start.y, entity.dxf.start.z)
                                end = (entity.dxf.end.x, entity.dxf.end.y, entity.dxf.end.z)
                                length = math.sqrt((end[0]-start[0])**2 + (end[1]-start[1])**2)
                            else:
                                points = list(entity.get_points(format='xy'))
                                if len(points) < 2:
                                    continue
                                start = (points[0][0], points[0][1], 0)
                                length = sum(math.sqrt((points[i+1][0]-points[i][0])**2 + (points[i+1][1]-points[i][1])**2)
                                           for i in range(len(points)-1))
                            
                            utilities.append(CivilElementInfo(
                                element_type=element_type,
                                layer=layer,
                                position=start,
                                geometry_type="line",
                                length=length / 1000
                            ))
                        except:
                            continue
                    break
        
        return utilities
    
    def _detect_earthwork(self) -> List[CivilElementInfo]:
        """تشخیص مناطق برش و خاکریزی"""
        earthwork = []
        
        cut_layers = ["CUT", "EXCAVATION", "برش", "گودبرداری"]
        fill_layers = ["FILL", "EMBANKMENT", "خاکریزی"]
        
        for entity in self.msp:
            layer = entity.dxf.layer if hasattr(entity.dxf, 'layer') else ""
            layer_upper = layer.upper()
            
            is_cut = any(kw in layer_upper for kw in cut_layers)
            is_fill = any(kw in layer_upper for kw in fill_layers)
            
            if (is_cut or is_fill) and entity.dxftype() == "LWPOLYLINE":
                try:
                    points = list(entity.get_points(format='xy'))
                    if len(points) >= 3:
                        is_closed = entity.is_closed or (points[0] == points[-1])
                        if is_closed:
                            area = abs(self._calculate_area(points))
                            xs = [p[0] for p in points]
                            ys = [p[1] for p in points]
                            center = ((min(xs) + max(xs))/2, (min(ys) + max(ys))/2, 0)
                            
                            elem_type = CivilElementType.CUT_AREA if is_cut else CivilElementType.FILL_AREA
                            
                            earthwork.append(CivilElementInfo(
                                element_type=elem_type,
                                layer=layer,
                                position=center,
                                geometry_type="polygon",
                                area=area / 1_000_000
                            ))
                except:
                    continue
        
        return earthwork
    
    def _extract_elevation_from_layer(self, layer: str) -> Optional[float]:
        """استخراج ارتفاع از نام لایه"""
        import re
        # جستجوی الگوهای مثل "CONTOUR-100" یا "ELEV_50"
        match = re.search(r'[-_](\d+\.?\d*)', layer)
        if match:
            try:
                return float(match.group(1))
            except:
                pass
        return None
    
    def _detect_interior_elements(self) -> List[InteriorElementInfo]:
        """تشخیص عناصر معماری داخلی و دکوراسیون"""
        interior_elements = []
        
        # جمع‌آوری متن‌ها
        texts_with_position = []
        for entity in self.msp:
            if entity.dxftype() in ["TEXT", "MTEXT"]:
                try:
                    text_content = entity.dxf.text if entity.dxftype() == "TEXT" else entity.text
                    position = entity.dxf.insert if hasattr(entity.dxf, 'insert') else (0, 0, 0)
                    layer = entity.dxf.layer if hasattr(entity.dxf, 'layer') else ""
                    texts_with_position.append({
                        'text': text_content.lower(),
                        'position': position,
                        'layer': layer
                    })
                except:
                    continue
        
        # تشخیص کفپوش‌ها
        flooring = self._detect_flooring()
        interior_elements.extend(flooring)
        
        # تشخیص مبلمان
        furniture = self._detect_furniture()
        interior_elements.extend(furniture)
        
        # تشخیص نورپردازی
        lighting = self._detect_interior_lighting()
        interior_elements.extend(lighting)
        
        # تشخیص عناصر آشپزخانه و حمام
        kitchen_bath = self._detect_kitchen_bathroom()
        interior_elements.extend(kitchen_bath)
        
        return interior_elements
    
    def _detect_flooring(self) -> List[InteriorElementInfo]:
        """تشخیص کفپوش‌ها"""
        flooring = []
        flooring_layers = ["FLOOR", "FLOORING", "TILE", "کفپوش", "کاشی", "پارکت"]
        
        for entity in self.msp:
            layer = entity.dxf.layer if hasattr(entity.dxf, 'layer') else ""
            
            if any(kw in layer.upper() for kw in flooring_layers):
                if entity.dxftype() == "LWPOLYLINE":
                    try:
                        points = list(entity.get_points(format='xy'))
                        if len(points) >= 3:
                            is_closed = entity.is_closed or (len(points) >= 4 and points[0] == points[-1])
                            if is_closed:
                                area = abs(self._calculate_area(points))
                                xs = [p[0] for p in points]
                                ys = [p[1] for p in points]
                                center = ((min(xs) + max(xs))/2, (min(ys) + max(ys))/2, 0)
                                
                                # تشخیص نوع کفپوش از نام لایه
                                elem_type = InteriorElementType.FLOORING_TILE
                                if "WOOD" in layer.upper() or "پارکت" in layer:
                                    elem_type = InteriorElementType.FLOORING_WOOD
                                elif "CARPET" in layer.upper() or "موکت" in layer:
                                    elem_type = InteriorElementType.FLOORING_CARPET
                                elif "STONE" in layer.upper() or "سنگ" in layer:
                                    elem_type = InteriorElementType.FLOORING_STONE
                                elif "MARBLE" in layer.upper() or "مرمر" in layer:
                                    elem_type = InteriorElementType.FLOORING_MARBLE
                                
                                flooring.append(InteriorElementInfo(
                                    element_type=elem_type,
                                    layer=layer,
                                    position=center,
                                    geometry_type="polygon",
                                    area=area / 1_000_000  # به متر مربع
                                ))
                    except:
                        continue
        
        return flooring
    
    def _detect_furniture(self) -> List[InteriorElementInfo]:
        """تشخیص مبلمان"""
        furniture = []
        furniture_layers = ["FURNITURE", "FURN", "FF&E", "مبلمان", "مبل"]
        
        for entity in self.msp:
            layer = entity.dxf.layer if hasattr(entity.dxf, 'layer') else ""
            
            if any(kw in layer.upper() for kw in furniture_layers):
                if entity.dxftype() in ["INSERT", "BLOCK", "LWPOLYLINE", "CIRCLE"]:
                    try:
                        if entity.dxftype() == "INSERT":
                            position = (entity.dxf.insert.x, entity.dxf.insert.y, entity.dxf.insert.z)
                            block_name = entity.dxf.name.lower()
                            
                            # تشخیص نوع از نام بلوک
                            elem_type = InteriorElementType.FURNITURE_TABLE
                            if "bed" in block_name or "تخت" in block_name:
                                elem_type = InteriorElementType.FURNITURE_BED
                            elif "sofa" in block_name or "مبل" in block_name:
                                elem_type = InteriorElementType.FURNITURE_SOFA
                            elif "chair" in block_name or "صندلی" in block_name:
                                elem_type = InteriorElementType.FURNITURE_CHAIR
                            elif "desk" in block_name or "میز کار" in block_name:
                                elem_type = InteriorElementType.FURNITURE_DESK
                            elif "cabinet" in block_name or "کابینت" in block_name:
                                elem_type = InteriorElementType.FURNITURE_CABINET
                            
                            furniture.append(InteriorElementInfo(
                                element_type=elem_type,
                                layer=layer,
                                position=position,
                                geometry_type="block",
                                label=entity.dxf.name
                            ))
                        
                        elif entity.dxftype() == "LWPOLYLINE":
                            points = list(entity.get_points(format='xy'))
                            if len(points) >= 3:
                                xs = [p[0] for p in points]
                                ys = [p[1] for p in points]
                                center = ((min(xs) + max(xs))/2, (min(ys) + max(ys))/2, 0)
                                width = (max(xs) - min(xs)) / 1000
                                depth = (max(ys) - min(ys)) / 1000
                                
                                furniture.append(InteriorElementInfo(
                                    element_type=InteriorElementType.FURNITURE_TABLE,
                                    layer=layer,
                                    position=center,
                                    geometry_type="rectangle",
                                    dimensions=(width, depth, 0)
                                ))
                    except:
                        continue
        
        return furniture
    
    def _detect_interior_lighting(self) -> List[InteriorElementInfo]:
        """تشخیص نورپردازی داخلی"""
        lighting = []
        lighting_layers = ["LIGHT", "LIGHTING", "FIXTURE", "نور", "چراغ", "لوستر"]
        
        for entity in self.msp:
            layer = entity.dxf.layer if hasattr(entity.dxf, 'layer') else ""
            
            if any(kw in layer.upper() for kw in lighting_layers):
                if entity.dxftype() in ["CIRCLE", "INSERT"]:
                    try:
                        if entity.dxftype() == "CIRCLE":
                            center = entity.dxf.center
                            position = (center.x, center.y, center.z)
                        else:
                            position = (entity.dxf.insert.x, entity.dxf.insert.y, entity.dxf.insert.z)
                        
                        # تشخیص نوع از لایه
                        elem_type = InteriorElementType.LIGHT_RECESSED
                        if "CHANDELIER" in layer.upper() or "لوستر" in layer:
                            elem_type = InteriorElementType.LIGHT_CHANDELIER
                        elif "PENDANT" in layer.upper() or "آویز" in layer:
                            elem_type = InteriorElementType.LIGHT_PENDANT
                        elif "TRACK" in layer.upper() or "ریل" in layer:
                            elem_type = InteriorElementType.LIGHT_TRACK
                        elif "SPOT" in layer.upper():
                            elem_type = InteriorElementType.LIGHT_SPOTLIGHT
                        
                        lighting.append(InteriorElementInfo(
                            element_type=elem_type,
                            layer=layer,
                            position=position,
                            geometry_type="point"
                        ))
                    except:
                        continue
        
        return lighting
    
    def _detect_kitchen_bathroom(self) -> List[InteriorElementInfo]:
        """تشخیص عناصر آشپزخانه و حمام"""
        elements = []
        
        # آشپزخانه
        kitchen_layers = ["KITCHEN", "آشپزخانه", "SINK", "COUNTER"]
        for entity in self.msp:
            layer = entity.dxf.layer if hasattr(entity.dxf, 'layer') else ""
            
            if any(kw in layer.upper() for kw in kitchen_layers):
                if entity.dxftype() in ["LWPOLYLINE", "INSERT"]:
                    try:
                        if entity.dxftype() == "INSERT":
                            position = (entity.dxf.insert.x, entity.dxf.insert.y, entity.dxf.insert.z)
                            block_name = entity.dxf.name.lower()
                            
                            elem_type = InteriorElementType.KITCHEN_COUNTER
                            if "sink" in block_name or "سینک" in block_name:
                                elem_type = InteriorElementType.KITCHEN_SINK
                            elif "stove" in block_name or "اجاق" in block_name:
                                elem_type = InteriorElementType.KITCHEN_STOVE
                            elif "fridge" in block_name or "یخچال" in block_name:
                                elem_type = InteriorElementType.KITCHEN_REFRIGERATOR
                            
                            elements.append(InteriorElementInfo(
                                element_type=elem_type,
                                layer=layer,
                                position=position,
                                geometry_type="block"
                            ))
                    except:
                        continue
        
        # حمام
        bathroom_layers = ["BATHROOM", "BATH", "حمام", "WC", "TOILET"]
        for entity in self.msp:
            layer = entity.dxf.layer if hasattr(entity.dxf, 'layer') else ""
            
            if any(kw in layer.upper() for kw in bathroom_layers):
                if entity.dxftype() in ["INSERT", "CIRCLE"]:
                    try:
                        if entity.dxftype() == "INSERT":
                            position = (entity.dxf.insert.x, entity.dxf.insert.y, entity.dxf.insert.z)
                            block_name = entity.dxf.name.lower()
                            
                            elem_type = InteriorElementType.BATHROOM_TOILET
                            if "vanity" in block_name or "روشویی" in block_name:
                                elem_type = InteriorElementType.BATHROOM_VANITY
                            elif "shower" in block_name or "دوش" in block_name:
                                elem_type = InteriorElementType.BATHROOM_SHOWER
                            elif "bath" in block_name or "وان" in block_name:
                                elem_type = InteriorElementType.BATHROOM_BATHTUB
                            
                            elements.append(InteriorElementInfo(
                                element_type=elem_type,
                                layer=layer,
                                position=position,
                                geometry_type="block"
                            ))
                    except:
                        continue
        
        return elements
    
    def _detect_safety_security_elements(self) -> List[SafetySecurityElementInfo]:
        """تشخیص عناصر ایمنی و امنیت"""
        elements = []
        
        # تشخیص سیستم اعلام حریق
        fire_alarm = self._detect_fire_alarm_system()
        elements.extend(fire_alarm)
        
        # تشخیص سیستم اطفاء حریق
        fire_suppression = self._detect_fire_suppression()
        elements.extend(fire_suppression)
        
        # تشخیص خروج اضطراری
        emergency_exit = self._detect_emergency_exits()
        elements.extend(emergency_exit)
        
        # تشخیص سیستم امنیتی (دوربین، کنترل تردد)
        security = self._detect_security_systems()
        elements.extend(security)
        
        return elements
    
    def _detect_fire_alarm_system(self) -> List[SafetySecurityElementInfo]:
        """تشخیص سیستم اعلام حریق"""
        elements = []
        fire_alarm_layers = ["FIRE", "ALARM", "FA", "F-ALARM", "حریق", "اعلام"]
        
        for entity in self.msp:
            layer = entity.dxf.layer if hasattr(entity.dxf, 'layer') else ""
            
            if any(kw in layer.upper() for kw in fire_alarm_layers):
                if entity.dxftype() in ["INSERT", "CIRCLE", "POINT"]:
                    try:
                        if entity.dxftype() == "INSERT":
                            position = (entity.dxf.insert.x, entity.dxf.insert.y, entity.dxf.insert.z)
                            block_name = entity.dxf.name.lower()
                            
                            elem_type = SafetySecurityElementType.FIRE_ALARM_DETECTOR
                            if "panel" in block_name or "پنل" in block_name:
                                elem_type = SafetySecurityElementType.FIRE_ALARM_PANEL
                            elif "manual" in block_name or "دستی" in block_name or "push" in block_name:
                                elem_type = SafetySecurityElementType.FIRE_ALARM_MANUAL_STATION
                            elif "horn" in block_name or "آژیر" in block_name or "bell" in block_name:
                                elem_type = SafetySecurityElementType.FIRE_ALARM_HORN
                            elif "strobe" in block_name or "چشمک" in block_name:
                                elem_type = SafetySecurityElementType.FIRE_ALARM_STROBE
                            elif "detector" in block_name or "آشکار" in block_name or "smoke" in block_name:
                                elem_type = SafetySecurityElementType.FIRE_ALARM_DETECTOR
                            
                            elements.append(SafetySecurityElementInfo(
                                element_type=elem_type,
                                layer=layer,
                                position=position,
                                geometry_type="block",
                                label=entity.dxf.name
                            ))
                        elif entity.dxftype() == "CIRCLE":
                            center = entity.dxf.center
                            position = (center.x, center.y, center.z)
                            
                            elements.append(SafetySecurityElementInfo(
                                element_type=SafetySecurityElementType.FIRE_ALARM_DETECTOR,
                                layer=layer,
                                position=position,
                                geometry_type="point"
                            ))
                    except:
                        continue
        
        return elements
    
    def _detect_fire_suppression(self) -> List[SafetySecurityElementInfo]:
        """تشخیص سیستم اطفاء حریق"""
        elements = []
        suppression_layers = ["SPRINKLER", "EXTINGUISHER", "HOSE", "HYDRANT", "اطفاء", "اسپرینکلر"]
        
        for entity in self.msp:
            layer = entity.dxf.layer if hasattr(entity.dxf, 'layer') else ""
            
            if any(kw in layer.upper() for kw in suppression_layers):
                if entity.dxftype() in ["INSERT", "CIRCLE"]:
                    try:
                        if entity.dxftype() == "INSERT":
                            position = (entity.dxf.insert.x, entity.dxf.insert.y, entity.dxf.insert.z)
                            block_name = entity.dxf.name.lower()
                            
                            elem_type = SafetySecurityElementType.SPRINKLER_HEAD
                            if "extinguisher" in block_name or "کپسول" in block_name:
                                elem_type = SafetySecurityElementType.FIRE_EXTINGUISHER
                            elif "hose" in block_name or "شیلنگ" in block_name or "cabinet" in block_name:
                                elem_type = SafetySecurityElementType.FIRE_HOSE_CABINET
                            elif "hydrant" in block_name or "هیدرانت" in block_name:
                                elem_type = SafetySecurityElementType.FIRE_HYDRANT
                            elif "fm200" in block_name or "nozzle" in block_name:
                                elem_type = SafetySecurityElementType.FM200_NOZZLE
                            
                            elements.append(SafetySecurityElementInfo(
                                element_type=elem_type,
                                layer=layer,
                                position=position,
                                geometry_type="block",
                                label=entity.dxf.name
                            ))
                        else:  # CIRCLE
                            center = entity.dxf.center
                            position = (center.x, center.y, center.z)
                            
                            elements.append(SafetySecurityElementInfo(
                                element_type=SafetySecurityElementType.SPRINKLER_HEAD,
                                layer=layer,
                                position=position,
                                geometry_type="point"
                            ))
                    except:
                        continue
        
        return elements
    
    def _detect_emergency_exits(self) -> List[SafetySecurityElementInfo]:
        """تشخیص خروج اضطراری و مسیرهای تخلیه"""
        elements = []
        exit_layers = ["EXIT", "EMERGENCY", "EVACUATION", "خروج", "اضطراری", "تخلیه"]
        
        for entity in self.msp:
            layer = entity.dxf.layer if hasattr(entity.dxf, 'layer') else ""
            
            if any(kw in layer.upper() for kw in exit_layers):
                if entity.dxftype() in ["INSERT", "LWPOLYLINE", "LINE"]:
                    try:
                        if entity.dxftype() == "INSERT":
                            position = (entity.dxf.insert.x, entity.dxf.insert.y, entity.dxf.insert.z)
                            block_name = entity.dxf.name.lower()
                            
                            elem_type = SafetySecurityElementType.EXIT_SIGN
                            if "door" in block_name or "درب" in block_name:
                                elem_type = SafetySecurityElementType.EXIT_DOOR
                            elif "light" in block_name or "lamp" in block_name or "چراغ" in block_name:
                                elem_type = SafetySecurityElementType.EMERGENCY_LIGHT
                            elif "assembly" in block_name or "تجمع" in block_name:
                                elem_type = SafetySecurityElementType.ASSEMBLY_POINT
                            
                            elements.append(SafetySecurityElementInfo(
                                element_type=elem_type,
                                layer=layer,
                                position=position,
                                geometry_type="block",
                                label=entity.dxf.name
                            ))
                        elif entity.dxftype() in ["LINE", "LWPOLYLINE"]:
                            # مسیر تخلیه
                            if entity.dxftype() == "LINE":
                                start = entity.dxf.start
                                end = entity.dxf.end
                                mid_x = (start.x + end.x) / 2
                                mid_y = (start.y + end.y) / 2
                                position = (mid_x, mid_y, 0)
                            else:
                                points = list(entity.get_points(format='xy'))
                                if points:
                                    mid_x = sum(p[0] for p in points) / len(points)
                                    mid_y = sum(p[1] for p in points) / len(points)
                                    position = (mid_x, mid_y, 0)
                                else:
                                    continue
                            
                            elements.append(SafetySecurityElementInfo(
                                element_type=SafetySecurityElementType.EVACUATION_ROUTE,
                                layer=layer,
                                position=position,
                                geometry_type="line"
                            ))
                    except:
                        continue
        
        return elements
    
    def _detect_security_systems(self) -> List[SafetySecurityElementInfo]:
        """تشخیص سیستم‌های امنیتی (دوربین، کنترل تردد، حسگرها)"""
        elements = []
        security_layers = ["SECURITY", "CCTV", "CAMERA", "ACCESS", "SENSOR", "امنیت", "دوربین"]
        
        for entity in self.msp:
            layer = entity.dxf.layer if hasattr(entity.dxf, 'layer') else ""
            
            if any(kw in layer.upper() for kw in security_layers):
                if entity.dxftype() in ["INSERT", "CIRCLE", "POINT"]:
                    try:
                        if entity.dxftype() == "INSERT":
                            position = (entity.dxf.insert.x, entity.dxf.insert.y, entity.dxf.insert.z)
                            block_name = entity.dxf.name.lower()
                            
                            # بررسی با اولویت از خاص به عام
                            elem_type = SafetySecurityElementType.CCTV_CAMERA
                            if "door contact" in block_name or "door-contact" in block_name:
                                elem_type = SafetySecurityElementType.DOOR_CONTACT
                            elif "metal detect" in block_name or "فلزیاب" in block_name:
                                elem_type = SafetySecurityElementType.METAL_DETECTOR
                            elif "panel" in block_name and "access" in block_name:
                                elem_type = SafetySecurityElementType.ACCESS_CONTROL_PANEL
                            elif "dvr" in block_name or "nvr" in block_name or "recorder" in block_name:
                                elem_type = SafetySecurityElementType.CCTV_DVR
                            elif "reader" in block_name or "کارتخوان" in block_name:
                                elem_type = SafetySecurityElementType.ACCESS_CONTROL_READER
                            elif "biometric" in block_name or "finger" in block_name:
                                elem_type = SafetySecurityElementType.BIOMETRIC_SCANNER
                            elif "lock" in block_name or "قفل" in block_name:
                                elem_type = SafetySecurityElementType.MAGNETIC_LOCK
                            elif "turnstile" in block_name or "گیت" in block_name:
                                elem_type = SafetySecurityElementType.TURNSTILE
                            elif "barrier" in block_name or "راهبند" in block_name:
                                elem_type = SafetySecurityElementType.BARRIER_GATE
                            elif "motion" in block_name or "pir" in block_name:
                                elem_type = SafetySecurityElementType.MOTION_SENSOR
                            elif "panic" in block_name or "sos" in block_name:
                                elem_type = SafetySecurityElementType.PANIC_BUTTON
                            
                            elements.append(SafetySecurityElementInfo(
                                element_type=elem_type,
                                layer=layer,
                                position=position,
                                geometry_type="block",
                                label=entity.dxf.name
                            ))
                        else:  # CIRCLE or POINT
                            if entity.dxftype() == "CIRCLE":
                                center = entity.dxf.center
                                position = (center.x, center.y, center.z)
                            else:
                                position = (entity.dxf.location.x, entity.dxf.location.y, entity.dxf.location.z)
                            
                            elements.append(SafetySecurityElementInfo(
                                element_type=SafetySecurityElementType.CCTV_CAMERA,
                                layer=layer,
                                position=position,
                                geometry_type="point"
                            ))
                    except:
                        continue
        
        return elements
    
    def _detect_advanced_structural_elements(self) -> List[AdvancedStructuralElementInfo]:
        """تشخیص عناصر تحلیل پیشرفته سازه‌ای"""
        elements = []
        
        # تشخیص عناصر لرزه‌ای
        seismic = self._detect_seismic_elements()
        elements.extend(seismic)
        
        # تشخیص پی‌های ویژه
        foundations = self._detect_specialized_foundations()
        elements.extend(foundations)
        
        # تشخیص اتصالات پیشرفته
        connections = self._detect_advanced_connections()
        elements.extend(connections)
        
        # تشخیص تقویت و بازسازی
        retrofit = self._detect_retrofit_elements()
        elements.extend(retrofit)
        
        return elements
    
    def _detect_seismic_elements(self) -> List[AdvancedStructuralElementInfo]:
        """تشخیص عناصر لرزه‌ای (جداساز، میراگر، دیوار برشی)"""
        elements = []
        seismic_layers = ["SEISMIC", "ISOLATOR", "DAMPER", "SHEAR", "لرزه", "جداساز", "میراگر"]
        
        for entity in self.msp:
            layer = entity.dxf.layer if hasattr(entity.dxf, 'layer') else ""
            
            if any(kw in layer.upper() for kw in seismic_layers):
                if entity.dxftype() in ["INSERT", "CIRCLE", "LWPOLYLINE", "RECTANGLE"]:
                    try:
                        if entity.dxftype() == "INSERT":
                            position = (entity.dxf.insert.x, entity.dxf.insert.y, entity.dxf.insert.z)
                            block_name = entity.dxf.name.lower()
                            
                            elem_type = AdvancedStructuralElementType.SEISMIC_ISOLATOR
                            if "base" in block_name and "isolator" in block_name:
                                elem_type = AdvancedStructuralElementType.BASE_ISOLATOR
                            elif "viscous" in block_name or "ویسکوز" in block_name:
                                elem_type = AdvancedStructuralElementType.VISCOUS_DAMPER
                            elif "friction" in block_name or "اصطکاک" in block_name:
                                elem_type = AdvancedStructuralElementType.FRICTION_DAMPER
                            elif "tmd" in block_name or "tuned mass" in block_name:
                                elem_type = AdvancedStructuralElementType.TUNED_MASS_DAMPER
                            elif "damper" in block_name or "میراگر" in block_name:
                                elem_type = AdvancedStructuralElementType.DAMPER
                            elif "joint" in block_name or "درز" in block_name:
                                elem_type = AdvancedStructuralElementType.SEISMIC_JOINT
                            
                            elements.append(AdvancedStructuralElementInfo(
                                element_type=elem_type,
                                layer=layer,
                                position=position,
                                geometry_type="block",
                                label=entity.dxf.name
                            ))
                        elif entity.dxftype() in ["LWPOLYLINE", "RECTANGLE"]:
                            # دیوار برشی تقویت‌شده
                            points = list(entity.get_points(format='xy'))
                            if points:
                                xs = [p[0] for p in points]
                                ys = [p[1] for p in points]
                                center = ((min(xs) + max(xs))/2, (min(ys) + max(ys))/2, 0)
                                
                                elements.append(AdvancedStructuralElementInfo(
                                    element_type=AdvancedStructuralElementType.SHEAR_WALL_REINFORCED,
                                    layer=layer,
                                    position=center,
                                    geometry_type="polygon"
                                ))
                    except:
                        continue
        
        return elements
    
    def _detect_specialized_foundations(self) -> List[AdvancedStructuralElementInfo]:
        """تشخیص پی‌های ویژه (شمع، کسن، رادیه)"""
        elements = []
        foundation_layers = [
            "PILE", "FOUNDATION", "FOOTING", "CAISSON", "RAFT", "MAT",
            "شمع", "پی", "فونداسیون", "رادیه"
        ]
        
        for entity in self.msp:
            layer = entity.dxf.layer if hasattr(entity.dxf, 'layer') else ""
            
            if any(kw in layer.upper() for kw in foundation_layers):
                if entity.dxftype() in ["INSERT", "CIRCLE", "LWPOLYLINE"]:
                    try:
                        if entity.dxftype() == "INSERT":
                            position = (entity.dxf.insert.x, entity.dxf.insert.y, entity.dxf.insert.z)
                            block_name = entity.dxf.name.lower()
                            
                            elem_type = AdvancedStructuralElementType.PILE_FOUNDATION
                            if "micro" in block_name or "ریز" in block_name:
                                elem_type = AdvancedStructuralElementType.MICRO_PILE
                            elif "caisson" in block_name or "کسن" in block_name:
                                elem_type = AdvancedStructuralElementType.CAISSON
                            elif "cap" in block_name or "کلاهک" in block_name:
                                elem_type = AdvancedStructuralElementType.PILE_CAP
                            elif "anchor" in block_name or "مهار" in block_name:
                                elem_type = AdvancedStructuralElementType.GROUND_ANCHOR
                            elif "nail" in block_name or "میخ" in block_name:
                                elem_type = AdvancedStructuralElementType.SOIL_NAIL
                            elif "combined" in block_name or "ترکیبی" in block_name:
                                elem_type = AdvancedStructuralElementType.COMBINED_FOOTING
                            
                            elements.append(AdvancedStructuralElementInfo(
                                element_type=elem_type,
                                layer=layer,
                                position=position,
                                geometry_type="block",
                                label=entity.dxf.name
                            ))
                        elif entity.dxftype() == "CIRCLE":
                            # شمع به صورت دایره
                            center = entity.dxf.center
                            position = (center.x, center.y, center.z)
                            diameter = entity.dxf.radius * 2
                            
                            elements.append(AdvancedStructuralElementInfo(
                                element_type=AdvancedStructuralElementType.PILE_FOUNDATION,
                                layer=layer,
                                position=position,
                                geometry_type="circle",
                                diameter=diameter * 1000  # به میلی‌متر
                            ))
                        elif entity.dxftype() == "LWPOLYLINE":
                            # پی رادیه یا گسترده
                            points = list(entity.get_points(format='xy'))
                            if len(points) >= 4:
                                xs = [p[0] for p in points]
                                ys = [p[1] for p in points]
                                center = ((min(xs) + max(xs))/2, (min(ys) + max(ys))/2, 0)
                                
                                elem_type = AdvancedStructuralElementType.MAT_FOUNDATION
                                if "raft" in layer.lower() or "رادیه" in layer:
                                    elem_type = AdvancedStructuralElementType.RAFT_FOUNDATION
                                
                                elements.append(AdvancedStructuralElementInfo(
                                    element_type=elem_type,
                                    layer=layer,
                                    position=center,
                                    geometry_type="polygon"
                                ))
                    except:
                        continue
        
        return elements
    
    def _detect_advanced_connections(self) -> List[AdvancedStructuralElementInfo]:
        """تشخیص اتصالات پیشرفته (گیردار، پیچی، جوشی)"""
        elements = []
        connection_layers = ["CONNECTION", "MOMENT", "SPLICE", "BOLT", "WELD", "اتصال", "گیردار"]
        
        for entity in self.msp:
            layer = entity.dxf.layer if hasattr(entity.dxf, 'layer') else ""
            
            if any(kw in layer.upper() for kw in connection_layers):
                if entity.dxftype() in ["INSERT", "CIRCLE"]:
                    try:
                        if entity.dxftype() == "INSERT":
                            position = (entity.dxf.insert.x, entity.dxf.insert.y, entity.dxf.insert.z)
                            block_name = entity.dxf.name.lower()
                            
                            elem_type = AdvancedStructuralElementType.MOMENT_CONNECTION
                            if "bolt" in block_name or "پیچ" in block_name:
                                elem_type = AdvancedStructuralElementType.BOLTED_CONNECTION
                            elif "weld" in block_name or "جوش" in block_name:
                                elem_type = AdvancedStructuralElementType.WELDED_CONNECTION
                            elif "base plate" in block_name or "صفحه پایه" in block_name:
                                elem_type = AdvancedStructuralElementType.BASE_PLATE
                            elif "column splice" in block_name or "اتصال ستون" in block_name:
                                elem_type = AdvancedStructuralElementType.COLUMN_SPLICE
                            elif "beam splice" in block_name or "اتصال تیر" in block_name:
                                elem_type = AdvancedStructuralElementType.BEAM_SPLICE
                            elif "gusset" in block_name or "گوشه" in block_name:
                                elem_type = AdvancedStructuralElementType.GUSSET_PLATE
                            
                            elements.append(AdvancedStructuralElementInfo(
                                element_type=elem_type,
                                layer=layer,
                                position=position,
                                geometry_type="block",
                                label=entity.dxf.name
                            ))
                        else:  # CIRCLE - اتصال پیچی
                            center = entity.dxf.center
                            position = (center.x, center.y, center.z)
                            
                            elements.append(AdvancedStructuralElementInfo(
                                element_type=AdvancedStructuralElementType.BOLTED_CONNECTION,
                                layer=layer,
                                position=position,
                                geometry_type="circle"
                            ))
                    except:
                        continue
        
        return elements
    
    def _detect_retrofit_elements(self) -> List[AdvancedStructuralElementInfo]:
        """تشخیص عناصر تقویت و بازسازی (FRP، ژاکت، شاتکریت)"""
        elements = []
        retrofit_layers = ["RETROFIT", "STRENGTHEN", "FRP", "JACKET", "FIBER", "تقویت", "ژاکت"]
        
        for entity in self.msp:
            layer = entity.dxf.layer if hasattr(entity.dxf, 'layer') else ""
            
            if any(kw in layer.upper() for kw in retrofit_layers):
                if entity.dxftype() in ["INSERT", "LWPOLYLINE", "RECTANGLE"]:
                    try:
                        if entity.dxftype() == "INSERT":
                            position = (entity.dxf.insert.x, entity.dxf.insert.y, entity.dxf.insert.z)
                            block_name = entity.dxf.name.lower()
                            
                            elem_type = AdvancedStructuralElementType.FRP_REINFORCEMENT
                            if "carbon" in block_name or "کربن" in block_name:
                                elem_type = AdvancedStructuralElementType.CARBON_FIBER
                            elif "steel jacket" in block_name or "ژاکت فولاد" in block_name:
                                elem_type = AdvancedStructuralElementType.STEEL_JACKET
                            elif "concrete jacket" in block_name or "ژاکت بتن" in block_name:
                                elem_type = AdvancedStructuralElementType.CONCRETE_JACKET
                            elif "shotcrete" in block_name or "شاتکریت" in block_name:
                                elem_type = AdvancedStructuralElementType.SHOTCRETE
                            elif "post" in block_name and "tension" in block_name:
                                elem_type = AdvancedStructuralElementType.POST_TENSION
                            elif "pre" in block_name and "stress" in block_name:
                                elem_type = AdvancedStructuralElementType.PRE_STRESS
                            
                            elements.append(AdvancedStructuralElementInfo(
                                element_type=elem_type,
                                layer=layer,
                                position=position,
                                geometry_type="block",
                                label=entity.dxf.name
                            ))
                        elif entity.dxftype() in ["LWPOLYLINE", "RECTANGLE"]:
                            # منطقه تقویت
                            points = list(entity.get_points(format='xy'))
                            if points:
                                xs = [p[0] for p in points]
                                ys = [p[1] for p in points]
                                center = ((min(xs) + max(xs))/2, (min(ys) + max(ys))/2, 0)
                                
                                elements.append(AdvancedStructuralElementInfo(
                                    element_type=AdvancedStructuralElementType.FRP_REINFORCEMENT,
                                    layer=layer,
                                    position=center,
                                    geometry_type="polygon"
                                ))
                    except:
                        continue
        
        return elements

    # ---------------------- Special Equipment Detection ----------------------
    def _detect_special_equipment_elements(self) -> List['SpecialEquipmentElementInfo']:
        """تشخیص تجهیزات ویژه"""
        elements: List[SpecialEquipmentElementInfo] = []
        elements.extend(self._detect_vertical_transport())
        elements.extend(self._detect_loading_handling())
        elements.extend(self._detect_kitchen_equipment())
        elements.extend(self._detect_medical_equipment())
        elements.extend(self._detect_stage_equipment())
        elements.extend(self._detect_pool_equipment())
        return elements

    def _detect_vertical_transport(self) -> List['SpecialEquipmentElementInfo']:
        elements: List[SpecialEquipmentElementInfo] = []
        layer_keys = ["ELEVATOR", "LIFT", "ESCALATOR", "TRAVELATOR", "DUMBWAITER", "آسانسور", "پله برقی"]
        for entity in self.msp:
            try:
                layer = entity.dxf.layer if hasattr(entity.dxf, 'layer') else ""
                if not any(kw in layer.upper() for kw in layer_keys):
                    continue
                if entity.dxftype() == "INSERT":
                    pos = (entity.dxf.insert.x, entity.dxf.insert.y, entity.dxf.insert.z)
                    bname = entity.dxf.name.lower()
                    et = SpecialEquipmentElementType.ELEVATOR
                    if "escalator" in bname:
                        et = SpecialEquipmentElementType.ESCALATOR
                    elif "moving" in bname or "travel" in bname:
                        et = SpecialEquipmentElementType.MOVING_WALKWAY
                    elif "dumbwait" in bname:
                        et = SpecialEquipmentElementType.DUMBWAITER
                    elements.append(SpecialEquipmentElementInfo(
                        element_type=et, layer=layer, position=pos, geometry_type="block", label=entity.dxf.name
                    ))
            except:
                continue
        return elements

    def _detect_loading_handling(self) -> List['SpecialEquipmentElementInfo']:
        elements: List[SpecialEquipmentElementInfo] = []
        layer_keys = ["CRANE", "HOIST", "MONORAIL", "DOCK", "LEVELER", "RAMP", "جرثقیل", "بالابر"]
        for entity in self.msp:
            try:
                layer = entity.dxf.layer if hasattr(entity.dxf, 'layer') else ""
                if not any(kw in layer.upper() for kw in layer_keys):
                    continue
                if entity.dxftype() == "INSERT":
                    pos = (entity.dxf.insert.x, entity.dxf.insert.y, entity.dxf.insert.z)
                    bname = entity.dxf.name.lower()
                    et = SpecialEquipmentElementType.CRANE
                    if "hoist" in bname:
                        et = SpecialEquipmentElementType.HOIST
                    elif "monorail" in bname:
                        et = SpecialEquipmentElementType.MONORAIL_HOIST
                    elif "leveler" in bname:
                        et = SpecialEquipmentElementType.DOCK_LEVELER
                    elif "ramp" in bname:
                        et = SpecialEquipmentElementType.LOADING_RAMP
                    elements.append(SpecialEquipmentElementInfo(
                        element_type=et, layer=layer, position=pos, geometry_type="block", label=entity.dxf.name
                    ))
            except:
                continue
        return elements

    def _detect_kitchen_equipment(self) -> List['SpecialEquipmentElementInfo']:
        elements: List[SpecialEquipmentElementInfo] = []
        layer_keys = ["KITCHEN-EQUIP", "KITCHEN", "آشپزخانه"]
        for entity in self.msp:
            try:
                layer = entity.dxf.layer if hasattr(entity.dxf, 'layer') else ""
                if not any(kw in layer.upper() for kw in layer_keys):
                    continue
                if entity.dxftype() == "INSERT":
                    pos = (entity.dxf.insert.x, entity.dxf.insert.y, entity.dxf.insert.z)
                    bname = entity.dxf.name.lower()
                    et = SpecialEquipmentElementType.COOKING_RANGE
                    if "hood" in bname or "exhaust" in bname:
                        et = SpecialEquipmentElementType.HOOD_EXHAUST
                    elif "dishwash" in bname:
                        et = SpecialEquipmentElementType.DISHWASHER_COMMERCIAL
                    elif "walk" in bname and ("fridge" in bname or "cool" in bname):
                        et = SpecialEquipmentElementType.WALKIN_FRIDGE
                    elif "walk" in bname and "freezer" in bname:
                        et = SpecialEquipmentElementType.WALKIN_FREEZER
                    elements.append(SpecialEquipmentElementInfo(
                        element_type=et, layer=layer, position=pos, geometry_type="block", label=entity.dxf.name
                    ))
            except:
                continue
        return elements

    def _detect_medical_equipment(self) -> List['SpecialEquipmentElementInfo']:
        elements: List[SpecialEquipmentElementInfo] = []
        layer_keys = ["MEDICAL", "MRI", "CT", "X-RAY", "RADIOLOGY", "پزشکی"]
        for entity in self.msp:
            try:
                layer = entity.dxf.layer if hasattr(entity.dxf, 'layer') else ""
                if not any(kw in layer.upper() for kw in layer_keys):
                    continue
                if entity.dxftype() == "INSERT":
                    pos = (entity.dxf.insert.x, entity.dxf.insert.y, entity.dxf.insert.z)
                    bname = entity.dxf.name.lower()
                    et = SpecialEquipmentElementType.X_RAY
                    if "mri" in bname:
                        et = SpecialEquipmentElementType.MRI
                    elif bname.startswith("ct") or " ct" in bname:
                        et = SpecialEquipmentElementType.CT
                    elif "x-ray" in bname or "xray" in bname or "radiology" in bname:
                        et = SpecialEquipmentElementType.X_RAY
                    elif "operat" in bname:
                        et = SpecialEquipmentElementType.OPERATING_TABLE
                    elif "autoclave" in bname:
                        et = SpecialEquipmentElementType.AUTOCLAVE
                    elements.append(SpecialEquipmentElementInfo(
                        element_type=et, layer=layer, position=pos, geometry_type="block", label=entity.dxf.name
                    ))
            except:
                continue
        return elements

    def _detect_stage_equipment(self) -> List['SpecialEquipmentElementInfo']:
        elements: List[SpecialEquipmentElementInfo] = []
        layer_keys = ["STAGE", "RIGGING", "ORCHESTRA", "PIT", "صحنه"]
        for entity in self.msp:
            try:
                layer = entity.dxf.layer if hasattr(entity.dxf, 'layer') else ""
                if not any(kw in layer.upper() for kw in layer_keys):
                    continue
                if entity.dxftype() == "INSERT":
                    pos = (entity.dxf.insert.x, entity.dxf.insert.y, entity.dxf.insert.z)
                    bname = entity.dxf.name.lower()
                    et = SpecialEquipmentElementType.STAGE_LIFT
                    if "rig" in bname:
                        et = SpecialEquipmentElementType.RIGGING
                    elif "orchestra" in bname or "pit" in bname:
                        et = SpecialEquipmentElementType.ORCHESTRA_PIT_LIFT
                    elements.append(SpecialEquipmentElementInfo(
                        element_type=et, layer=layer, position=pos, geometry_type="block", label=entity.dxf.name
                    ))
            except:
                continue
        return elements

    def _detect_pool_equipment(self) -> List['SpecialEquipmentElementInfo']:
        elements: List[SpecialEquipmentElementInfo] = []
        layer_keys = ["POOL", "POOL-EQUIP", "JACUZZI", "استخر", "جکوزی"]
        for entity in self.msp:
            try:
                layer = entity.dxf.layer if hasattr(entity.dxf, 'layer') else ""
                if not any(kw in layer.upper() for kw in layer_keys):
                    continue
                if entity.dxftype() == "INSERT":
                    pos = (entity.dxf.insert.x, entity.dxf.insert.y, entity.dxf.insert.z)
                    bname = entity.dxf.name.lower()
                    et = SpecialEquipmentElementType.POOL_PUMP
                    if "filter" in bname:
                        et = SpecialEquipmentElementType.FILTRATION_UNIT
                    elif "chlor" in bname:
                        et = SpecialEquipmentElementType.CHLORINATION_UNIT
                    elif "jacuzzi" in bname:
                        et = SpecialEquipmentElementType.JACUZZI_EQUIPMENT
                    elements.append(SpecialEquipmentElementInfo(
                        element_type=et, layer=layer, position=pos, geometry_type="block", label=entity.dxf.name
                    ))
            except:
                continue
        return elements

    # ---------------------- Regulatory & Compliance Detection ----------------------
    def _detect_regulatory_elements(self) -> List['RegulatoryComplianceElementInfo']:
        elements: List[RegulatoryComplianceElementInfo] = []
        elements.extend(self._detect_zoning_setbacks())
        elements.extend(self._detect_code_compliance())
        elements.extend(self._detect_occupancy())
        elements.extend(self._detect_accessibility())
        elements.extend(self._detect_parking_compliance())
        elements.extend(self._detect_environmental_constraints())
        elements.extend(self._detect_permit_notes())
        return elements

    def _detect_zoning_setbacks(self) -> List['RegulatoryComplianceElementInfo']:
        elements: List[RegulatoryComplianceElementInfo] = []
        layer_keys = ["ZONING", "SETBACK", "RIGHT-OF-WAY", "EASEMENT", "ENVELOPE", "LEGAL", "حریم", "ضوابط"]
        for e in self.msp:
            try:
                layer = e.dxf.layer if hasattr(e.dxf, 'layer') else ""
                if not any(kw in layer.upper() for kw in layer_keys):
                    continue
                if e.dxftype() in ["LWPOLYLINE", "LINE"]:
                    # Treat as property/setback/easement line
                    et = RegulatoryComplianceElementType.SETBACK_LINE
                    lname = layer.lower()
                    if "property" in lname or "LEGAL" in layer.upper() or "حد" in layer:
                        et = RegulatoryComplianceElementType.PROPERTY_LINE
                    elif "easement" in lname or "ارتفاق" in layer:
                        et = RegulatoryComplianceElementType.EASEMENT
                    elif "right-of-way" in layer.upper():
                        et = RegulatoryComplianceElementType.RIGHT_OF_WAY
                    elif "envelope" in lname:
                        et = RegulatoryComplianceElementType.BUILDING_ENVELOPE
                    # Approximate position as start or centroid
                    pos = (0.0, 0.0, 0.0)
                    if e.dxftype() == "LINE":
                        pos = (e.dxf.start.x, e.dxf.start.y, 0.0)
                    elif e.dxftype() == "LWPOLYLINE":
                        pts = list(e.get_points(format='xy'))
                        if pts:
                            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                            pos = ((min(xs)+max(xs))/2, (min(ys)+max(ys))/2, 0.0)
                    elements.append(RegulatoryComplianceElementInfo(
                        element_type=et, layer=layer, position=pos, geometry_type=e.dxftype().lower()
                    ))
                elif e.dxftype() == "INSERT":
                    pos = (e.dxf.insert.x, e.dxf.insert.y, e.dxf.insert.z)
                    elements.append(RegulatoryComplianceElementInfo(
                        element_type=RegulatoryComplianceElementType.SETBACK_LINE,
                        layer=layer, position=pos, geometry_type="block", label=getattr(e.dxf, 'name', None)
                    ))
                elif e.dxftype() in ["TEXT", "MTEXT"]:
                    # استخراج متن از TEXT/MTEXT
                    txt = e.dxf.text if e.dxftype() == "TEXT" else (e.text if hasattr(e, 'text') else getattr(e, 'plain_text', ''))
                    pos = (e.dxf.insert.x, e.dxf.insert.y, 0.0) if hasattr(e.dxf, 'insert') else (0.0, 0.0, 0.0)
                    elements.append(RegulatoryComplianceElementInfo(
                        element_type=RegulatoryComplianceElementType.SETBACK_LINE,
                        layer=layer, position=pos, geometry_type="text", text=txt
                    ))
            except:
                continue
        return elements

    def _detect_code_compliance(self) -> List['RegulatoryComplianceElementInfo']:
        elements: List[RegulatoryComplianceElementInfo] = []
        layer_keys = ["CODE", "FIRE", "FIRE-RATING", "EGRESS", "EXIT", "TRAVEL", "کد", "حریق", "خروج"]
        for e in self.msp:
            try:
                layer = e.dxf.layer if hasattr(e.dxf, 'layer') else ""
                if not any(kw in layer.upper() for kw in layer_keys):
                    continue
                if e.dxftype() == "INSERT":
                    pos = (e.dxf.insert.x, e.dxf.insert.y, e.dxf.insert.z)
                    name = getattr(e.dxf, 'name', '').lower()
                    et = RegulatoryComplianceElementType.EGRESS_PATH
                    # ابتدا تشخیص‌های خاص‌تر
                    if ("door" in name and "fire" in name):
                        et = RegulatoryComplianceElementType.FIRE_RATING_DOOR
                    elif "exit" in name:
                        et = RegulatoryComplianceElementType.EXIT_COUNT
                    elif "stair" in name or "capacity" in name:
                        et = RegulatoryComplianceElementType.STAIR_CAPACITY
                    elif "travel" in name:
                        et = RegulatoryComplianceElementType.TRAVEL_DISTANCE
                    elif "rating" in name or "1h" in name or "2h" in name:
                        et = RegulatoryComplianceElementType.FIRE_RATING_WALL
                    elements.append(RegulatoryComplianceElementInfo(
                        element_type=et, layer=layer, position=pos, geometry_type="block", label=getattr(e.dxf, 'name', None)
                    ))
                elif e.dxftype() in ["TEXT", "MTEXT"]:
                    txt = e.dxf.text if e.dxftype() == "TEXT" else (e.text if hasattr(e, 'text') else getattr(e, 'plain_text', ''))
                    txt_low = txt.lower()
                    pos = (e.dxf.insert.x, e.dxf.insert.y, 0.0) if hasattr(e.dxf, 'insert') else (0.0, 0.0, 0.0)
                    et = RegulatoryComplianceElementType.CODE_REFERENCE
                    if "egress" in txt_low or "path" in txt_low:
                        et = RegulatoryComplianceElementType.EGRESS_PATH
                    elif "exit" in txt_low:
                        et = RegulatoryComplianceElementType.EXIT_COUNT
                    elif "stair" in txt_low and ("cap" in txt_low or "width" in txt_low):
                        et = RegulatoryComplianceElementType.STAIR_CAPACITY
                    elif any(h in txt_low for h in ["1h", "2h", "3h", "1-hour", "2-hour"]):
                        et = RegulatoryComplianceElementType.FIRE_RATING_WALL
                    elif "travel" in txt_low:
                        et = RegulatoryComplianceElementType.TRAVEL_DISTANCE
                    elements.append(RegulatoryComplianceElementInfo(
                        element_type=et, layer=layer, position=pos, geometry_type="text", text=txt
                    ))
            except:
                continue
        return elements

    def _detect_occupancy(self) -> List['RegulatoryComplianceElementInfo']:
        elements: List[RegulatoryComplianceElementInfo] = []
        layer_keys = ["OCCUPANCY", "LOAD", "HAZARD", "اشغال", "خطر"]
        for e in self.msp:
            try:
                layer = e.dxf.layer if hasattr(e.dxf, 'layer') else ""
                if not any(kw in layer.upper() for kw in layer_keys):
                    continue
                if e.dxftype() in ["TEXT", "MTEXT"]:
                    txt = e.dxf.text if e.dxftype() == "TEXT" else (e.text if hasattr(e, 'text') else getattr(e, 'plain_text', ''))
                    txt_low = txt.lower()
                    pos = (e.dxf.insert.x, e.dxf.insert.y, 0.0) if hasattr(e.dxf, 'insert') else (0.0, 0.0, 0.0)
                    et = RegulatoryComplianceElementType.OCCUPANCY_TYPE
                    if "load" in txt_low or "per" in txt_low:
                        et = RegulatoryComplianceElementType.OCCUPANT_LOAD
                    elif "hazard" in txt_low or "h-" in txt_low:
                        et = RegulatoryComplianceElementType.HAZARD_CLASS
                    elements.append(RegulatoryComplianceElementInfo(
                        element_type=et, layer=layer, position=pos, geometry_type="text", text=txt
                    ))
            except:
                continue
        return elements

    def _detect_accessibility(self) -> List['RegulatoryComplianceElementInfo']:
        elements: List[RegulatoryComplianceElementInfo] = []
        layer_keys = ["ADA", "ACCESS", "ACCESSIBLE", "معلول", "دسترسی"]
        for e in self.msp:
            try:
                layer = e.dxf.layer if hasattr(e.dxf, 'layer') else ""
                if not any(kw in layer.upper() for kw in layer_keys):
                    continue
                if e.dxftype() == "INSERT":
                    pos = (e.dxf.insert.x, e.dxf.insert.y, e.dxf.insert.z)
                    name = getattr(e.dxf, 'name', '').lower()
                    et = RegulatoryComplianceElementType.ADA_RAMP
                    if "door" in name or "clear" in name:
                        et = RegulatoryComplianceElementType.ADA_DOOR_CLEARANCE
                    elif "rest" in name or "wc" in name:
                        et = RegulatoryComplianceElementType.ADA_RESTROOM
                    elif "sign" in name:
                        et = RegulatoryComplianceElementType.ADA_SIGNAGE
                    elements.append(RegulatoryComplianceElementInfo(
                        element_type=et, layer=layer, position=pos, geometry_type="block", label=getattr(e.dxf, 'name', None)
                    ))
                elif e.dxftype() in ["TEXT", "MTEXT"]:
                    txt = e.dxf.text if e.dxftype() == "TEXT" else (e.text if hasattr(e, 'text') else getattr(e, 'plain_text', ''))
                    pos = (e.dxf.insert.x, e.dxf.insert.y, 0.0) if hasattr(e.dxf, 'insert') else (0.0, 0.0, 0.0)
                    elements.append(RegulatoryComplianceElementInfo(
                        element_type=RegulatoryComplianceElementType.ADA_RAMP,
                        layer=layer, position=pos, geometry_type="text", text=txt
                    ))
            except:
                continue
        return elements

    def _detect_parking_compliance(self) -> List['RegulatoryComplianceElementInfo']:
        elements: List[RegulatoryComplianceElementInfo] = []
        layer_keys = ["PARKING-REQ", "PARKING", "EV", "BICYCLE", "پارکینگ"]
        for e in self.msp:
            try:
                layer = e.dxf.layer if hasattr(e.dxf, 'layer') else ""
                if not any(kw in layer.upper() for kw in layer_keys):
                    continue
                if e.dxftype() == "INSERT":
                    pos = (e.dxf.insert.x, e.dxf.insert.y, e.dxf.insert.z)
                    name = getattr(e.dxf, 'name', '').lower()
                    et = RegulatoryComplianceElementType.PARKING_STALL_COUNT
                    if "ev" in name:
                        et = RegulatoryComplianceElementType.EV_PARKING
                    elif "access" in name:
                        et = RegulatoryComplianceElementType.ACCESSIBLE_PARKING
                    elif "bike" in name or "bicycle" in name:
                        et = RegulatoryComplianceElementType.BICYCLE_PARKING
                    elements.append(RegulatoryComplianceElementInfo(
                        element_type=et, layer=layer, position=pos, geometry_type="block", label=getattr(e.dxf, 'name', None)
                    ))
                elif e.dxftype() in ["TEXT", "MTEXT"]:
                    txt = e.dxf.text if e.dxftype() == "TEXT" else (e.text if hasattr(e, 'text') else getattr(e, 'plain_text', ''))
                    pos = (e.dxf.insert.x, e.dxf.insert.y, 0.0) if hasattr(e.dxf, 'insert') else (0.0, 0.0, 0.0)
                    elements.append(RegulatoryComplianceElementInfo(
                        element_type=RegulatoryComplianceElementType.PARKING_STALL_COUNT,
                        layer=layer, position=pos, geometry_type="text", text=txt
                    ))
            except:
                continue
        return elements

    def _detect_environmental_constraints(self) -> List['RegulatoryComplianceElementInfo']:
        elements: List[RegulatoryComplianceElementInfo] = []
        layer_keys = ["ENV", "ENVIRONMENT", "FLOOD", "WETLAND", "NO-BUILD", "HEIGHT", "FAR", "محیط"]
        for e in self.msp:
            try:
                layer = e.dxf.layer if hasattr(e.dxf, 'layer') else ""
                if not any(kw in layer.upper() for kw in layer_keys):
                    continue
                if e.dxftype() in ["LWPOLYLINE", "LINE"]:
                    pos = (0.0, 0.0, 0.0)
                    if e.dxftype() == "LINE":
                        pos = (e.dxf.start.x, e.dxf.start.y, 0.0)
                    else:
                        pts = list(e.get_points(format='xy'))
                        if pts:
                            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                            pos = ((min(xs)+max(xs))/2, (min(ys)+max(ys))/2, 0.0)
                    et = RegulatoryComplianceElementType.NO_BUILD_ZONE
                    lname = layer.lower()
                    if "flood" in lname:
                        et = RegulatoryComplianceElementType.FLOOD_ZONE
                    elif "wet" in lname:
                        et = RegulatoryComplianceElementType.WETLANDS
                    elif "height" in lname:
                        et = RegulatoryComplianceElementType.HEIGHT_LIMIT
                    elif "far" in lname:
                        et = RegulatoryComplianceElementType.FAR
                    elements.append(RegulatoryComplianceElementInfo(
                        element_type=et, layer=layer, position=pos, geometry_type=e.dxftype().lower()
                    ))
                elif e.dxftype() in ["TEXT", "MTEXT"]:
                    txt = e.dxf.text if e.dxftype() == "TEXT" else (e.text if hasattr(e, 'text') else getattr(e, 'plain_text', ''))
                    txt_low = txt.lower()
                    pos = (e.dxf.insert.x, e.dxf.insert.y, 0.0) if hasattr(e.dxf, 'insert') else (0.0, 0.0, 0.0)
                    et = RegulatoryComplianceElementType.NO_BUILD_ZONE
                    if "height" in txt_low:
                        et = RegulatoryComplianceElementType.HEIGHT_LIMIT
                    elif "far" in txt_low:
                        et = RegulatoryComplianceElementType.FAR
                    elements.append(RegulatoryComplianceElementInfo(
                        element_type=et,
                        layer=layer, position=pos, geometry_type="text", text=txt
                    ))
            except:
                continue
        return elements

    def _detect_permit_notes(self) -> List['RegulatoryComplianceElementInfo']:
        elements: List[RegulatoryComplianceElementInfo] = []
        layer_keys = ["PERMIT", "LEGAL", "INSPECTION", "مجوز", "بازرسی"]
        for e in self.msp:
            try:
                layer = e.dxf.layer if hasattr(e.dxf, 'layer') else ""
                if not any(kw in layer.upper() for kw in layer_keys):
                    continue
                if e.dxftype() in ["TEXT", "MTEXT"]:
                    txt = e.dxf.text if e.dxftype() == "TEXT" else (e.text if hasattr(e, 'text') else getattr(e, 'plain_text', ''))
                    pos = (e.dxf.insert.x, e.dxf.insert.y, 0.0) if hasattr(e.dxf, 'insert') else (0.0, 0.0, 0.0)
                    elements.append(RegulatoryComplianceElementInfo(
                        element_type=RegulatoryComplianceElementType.PERMIT_NOTE,
                        layer=layer, position=pos, geometry_type="text", text=txt
                    ))
                elif e.dxftype() == "INSERT":
                    pos = (e.dxf.insert.x, e.dxf.insert.y, e.dxf.insert.z)
                    elements.append(RegulatoryComplianceElementInfo(
                        element_type=RegulatoryComplianceElementType.INSPECTION_NOTE,
                        layer=layer, position=pos, geometry_type="block", label=getattr(e.dxf, 'name', None)
                    ))
            except:
                continue
        return elements

    # ---------------------- Sustainability & Energy Detection ----------------------
    def _detect_sustainability_elements(self) -> List['SustainabilityElementInfo']:
        elements: List[SustainabilityElementInfo] = []
        elements.extend(self._detect_renewables())
        elements.extend(self._detect_meters_monitoring())
        elements.extend(self._detect_envelope_insulation())
        elements.extend(self._detect_daylight_ventilation())
        elements.extend(self._detect_water_sustainability())
        elements.extend(self._detect_green_features())
        elements.extend(self._detect_certification_notes())
        elements.extend(self._detect_energy_zones())
        return elements

    def _detect_renewables(self) -> List['SustainabilityElementInfo']:
        elements: List[SustainabilityElementInfo] = []
        layer_keys = ["SOLAR", "PV", "ENERGY", "BATTERY", "خورشیدی", "باتری"]
        for e in self.msp:
            try:
                layer = e.dxf.layer if hasattr(e.dxf, 'layer') else ""
                if not any(kw in layer.upper() for kw in layer_keys):
                    continue
                if e.dxftype() == "INSERT":
                    pos = (e.dxf.insert.x, e.dxf.insert.y, e.dxf.insert.z)
                    name = getattr(e.dxf, 'name', '').lower()
                    et = SustainabilityElementType.SOLAR_PV_PANEL
                    if "inverter" in name:
                        et = SustainabilityElementType.SOLAR_INVERTER
                    elif "collector" in name or "thermal" in name:
                        et = SustainabilityElementType.SOLAR_THERMAL_COLLECTOR
                    elif "battery" in name or "storage" in name:
                        et = SustainabilityElementType.BATTERY_STORAGE
                    elements.append(SustainabilityElementInfo(
                        element_type=et, layer=layer, position=pos, geometry_type="block", label=getattr(e.dxf,'name',None)
                    ))
            except:
                continue
        return elements

    # ---------------------- Transportation & Traffic Detection ----------------------
    def _detect_transportation_elements(self) -> List['TransportationElementInfo']:
        elements: List[TransportationElementInfo] = []
        elements.extend(self._detect_parking_layout())
        elements.extend(self._detect_vehicle_routes())
        elements.extend(self._detect_pedestrian_paths())
        elements.extend(self._detect_signage_markings())
        elements.extend(self._detect_loading_areas())
        elements.extend(self._detect_bike_bus_features())
        elements.extend(self._detect_traffic_calming())
        return elements

    def _in_transport_layer(self, layer: str) -> bool:
        return any(key in (layer.upper() if layer else "") for key in self.TRANSPORT_LAYERS)

    def _detect_parking_layout(self) -> List['TransportationElementInfo']:
        elements: List[TransportationElementInfo] = []
        for e in self.msp:
            try:
                layer = e.dxf.layer if hasattr(e.dxf, 'layer') else ""
                if not self._in_transport_layer(layer) and not any(kw in (layer.upper()) for kw in ["PARK", "EV", "ADA"]):
                    continue
                if e.dxftype() == 'LWPOLYLINE':
                    # Heuristic: small rectangles are stalls; treat repeated first/last point as closed
                    pts = list(e.get_points(format='xy'))
                    if len(pts) >= 4:
                        # remove duplicate last point if equals first
                        if pts[0] == pts[-1]:
                            pts = pts[:-1]
                        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                        w = max(xs)-min(xs); h = max(ys)-min(ys)
                        area = abs(self._calculate_area(pts))
                        cx = (max(xs)+min(xs))/2; cy = (max(ys)+min(ys))/2
                        et = None
                        # typical stall ~ 2.4m x 4.8m in mm
                        if 1800 <= min(w,h) <= 3500 and 3500 <= max(w,h) <= 7000:
                            et = TransportationTrafficElementType.PARKING_STALL
                        elif area > 50_000 and min(w,h) > 3000:
                            et = TransportationTrafficElementType.PARKING_AISLE
                        if et:
                            elements.append(TransportationElementInfo(
                                element_type=et, layer=layer,
                                position=(cx, cy, 0.0), geometry_type='polygon',
                                width=w, length=h
                            ))
                elif e.dxftype() in ['TEXT','MTEXT'] and self._in_transport_layer(layer):
                    txt = e.dxf.text if e.dxftype()=="TEXT" else (getattr(e,'text',None) or getattr(e,'plain_text',''))
                    low = (txt or '').lower()
                    pos = (e.dxf.insert.x, e.dxf.insert.y, 0.0) if hasattr(e.dxf,'insert') else (0.0,0.0,0.0)
                    if any(k in low for k in ["ada", "accessible", "معلول"]):
                        elements.append(TransportationElementInfo(
                            element_type=TransportationTrafficElementType.ACCESSIBLE_PARKING,
                            layer=layer, position=pos, geometry_type='text', note=txt
                        ))
                    elif "ev" in low:
                        elements.append(TransportationElementInfo(
                            element_type=TransportationTrafficElementType.EV_PARKING,
                            layer=layer, position=pos, geometry_type='text', note=txt
                        ))
            except:
                continue
        return elements

    def _detect_vehicle_routes(self) -> List['TransportationElementInfo']:
        elements: List[TransportationElementInfo] = []
        for e in self.msp:
            try:
                layer = e.dxf.layer if hasattr(e.dxf, 'layer') else ""
                if not self._in_transport_layer(layer):
                    continue
                if e.dxftype() in ['LINE','LWPOLYLINE']:
                    # lanes and driveways
                    if e.dxftype() == 'LINE':
                        sx, sy = e.dxf.start.x, e.dxf.start.y
                        ex, ey = e.dxf.end.x, e.dxf.end.y
                        cx, cy = (sx+ex)/2, (sy+ey)/2
                        geom = 'line'
                    else:
                        pts = list(e.get_points(format='xy'))
                        if not pts:
                            continue
                        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                        cx, cy = sum(xs)/len(xs), sum(ys)/len(ys)
                        geom = 'polyline'
                    et = TransportationTrafficElementType.TRAFFIC_LANE
                    lname = layer.upper()
                    if "DRIVE" in lname or "ENTRY" in lname:
                        et = TransportationTrafficElementType.DRIVEWAY_ENTRY
                    elif "RAMP" in lname:
                        et = TransportationTrafficElementType.RAMP
                    elements.append(TransportationElementInfo(
                        element_type=et, layer=layer, position=(cx, cy, 0.0), geometry_type=geom
                    ))
                elif e.dxftype() == 'CIRCLE' and self._in_transport_layer(layer):
                    # could be turning radius annotation circle
                    c = e.dxf.center; r = e.dxf.radius
                    elements.append(TransportationElementInfo(
                        element_type=TransportationTrafficElementType.TURNING_RADIUS,
                        layer=layer, position=(c.x, c.y, 0.0), geometry_type='circle', radius=r, turning_radius=r
                    ))
                elif e.dxftype() == 'INSERT':
                    name = getattr(e.dxf,'name','').lower()
                    pos = (e.dxf.insert.x, e.dxf.insert.y, e.dxf.insert.z)
                    if 'arrow' in name:
                        elements.append(TransportationElementInfo(
                            element_type=TransportationTrafficElementType.TRAFFIC_FLOW_ARROW,
                            layer=layer, position=pos, geometry_type='block', label=getattr(e.dxf,'name',None)
                        ))
            except:
                continue
        return elements

    def _detect_pedestrian_paths(self) -> List['TransportationElementInfo']:
        elements: List[TransportationElementInfo] = []
        for e in self.msp:
            try:
                layer = e.dxf.layer if hasattr(e.dxf, 'layer') else ""
                if not self._in_transport_layer(layer):
                    continue
                lname = layer.upper()
                if e.dxftype() in ['LINE','LWPOLYLINE']:
                    pts = None
                    if e.dxftype() == 'LWPOLYLINE':
                        pts = list(e.get_points(format='xy'))
                    cx, cy = (0.0, 0.0)
                    if pts:
                        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                        cx, cy = sum(xs)/len(xs), sum(ys)/len(ys)
                    elif e.dxftype()=='LINE':
                        sx, sy = e.dxf.start.x, e.dxf.start.y
                        ex, ey = e.dxf.end.x, e.dxf.end.y
                        cx, cy = (sx+ex)/2, (sy+ey)/2
                    et = None
                    # Specific before general to avoid CROSSWALK being classified as WALKWAY
                    if "CROSS" in lname:
                        et = TransportationTrafficElementType.CROSSWALK
                    elif "WALK" in lname or "SIDEWALK" in lname:
                        et = TransportationTrafficElementType.WALKWAY
                    elif "CURB" in lname:
                        et = TransportationTrafficElementType.CURB_CUT
                    if et:
                        elements.append(TransportationElementInfo(
                            element_type=et, layer=layer, position=(cx, cy, 0.0), geometry_type=e.dxftype().lower()
                        ))
            except:
                continue
        return elements

    def _detect_signage_markings(self) -> List['TransportationElementInfo']:
        elements: List[TransportationElementInfo] = []
        for e in self.msp:
            try:
                layer = e.dxf.layer if hasattr(e.dxf, 'layer') else ""
                if not self._in_transport_layer(layer):
                    continue
                if e.dxftype() in ['TEXT','MTEXT']:
                    txt = e.dxf.text if e.dxftype()=="TEXT" else (getattr(e,'text',None) or getattr(e,'plain_text',''))
                    low = (txt or '').lower()
                    pos = (e.dxf.insert.x, e.dxf.insert.y, 0.0) if hasattr(e.dxf,'insert') else (0.0,0.0,0.0)
                    if 'stop' in low:
                        et = TransportationTrafficElementType.STOP_SIGN
                    elif 'yield' in low:
                        et = TransportationTrafficElementType.YIELD_SIGN
                    elif 'speed' in low and ('limit' in low or any(ch.isdigit() for ch in low)):
                        et = TransportationTrafficElementType.SPEED_LIMIT
                    else:
                        et = TransportationTrafficElementType.LANE_MARKING if any(k in low for k in ['lane', 'mark']) else None
                    if et:
                        # Extract speed limit number if present
                        spd = None
                        import re
                        m = re.search(r'(\d+)', low)
                        if m and et == TransportationTrafficElementType.SPEED_LIMIT:
                            spd = float(m.group(1))
                        elements.append(TransportationElementInfo(
                            element_type=et, layer=layer, position=pos, geometry_type='text', sign_text=txt, speed_limit=spd
                        ))
                elif e.dxftype() == 'INSERT':
                    name = getattr(e.dxf,'name','').lower()
                    pos = (e.dxf.insert.x, e.dxf.insert.y, e.dxf.insert.z)
                    et = None
                    if 'signal' in name or 'traffic light' in name or 'semaphore' in name:
                        et = TransportationTrafficElementType.TRAFFIC_LIGHT
                    elif 'sign' in name:
                        et = TransportationTrafficElementType.LANE_MARKING
                    if et:
                        elements.append(TransportationElementInfo(
                            element_type=et, layer=layer, position=pos, geometry_type='block', label=getattr(e.dxf,'name',None)
                        ))
            except:
                continue
        return elements

    def _detect_loading_areas(self) -> List['TransportationElementInfo']:
        elements: List[TransportationElementInfo] = []
        for e in self.msp:
            try:
                layer = e.dxf.layer if hasattr(e.dxf, 'layer') else ""
                if not self._in_transport_layer(layer):
                    continue
                lname = layer.upper()
                if e.dxftype() in ['LWPOLYLINE','HATCH','SOLID'] and ('LOADING' in lname or 'SERVICE' in lname):
                    # area polygon for loading
                    cx, cy = 0.0, 0.0
                    if e.dxftype()=='LWPOLYLINE':
                        pts = list(e.get_points(format='xy'))
                        if pts:
                            xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
                            cx, cy = (min(xs)+max(xs))/2, (min(ys)+max(ys))/2
                    elements.append(TransportationElementInfo(
                        element_type=TransportationTrafficElementType.LOADING_ZONE,
                        layer=layer, position=(cx, cy, 0.0), geometry_type=e.dxftype().lower()
                    ))
            except:
                continue
        return elements

    def _detect_bike_bus_features(self) -> List['TransportationElementInfo']:
        elements: List[TransportationElementInfo] = []
        for e in self.msp:
            try:
                layer = e.dxf.layer if hasattr(e.dxf, 'layer') else ""
                if not self._in_transport_layer(layer):
                    continue
                lname = layer.upper()
                if e.dxftype() in ['LINE','LWPOLYLINE']:
                    cx, cy = 0.0, 0.0
                    if e.dxftype()=='LWPOLYLINE':
                        pts=list(e.get_points(format='xy'))
                        if pts:
                            xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
                            cx, cy = sum(xs)/len(xs), sum(ys)/len(ys)
                    if 'BIKE' in lname or 'BICYCLE' in lname:
                        elements.append(TransportationElementInfo(
                            element_type=TransportationTrafficElementType.BIKE_LANE,
                            layer=layer, position=(cx, cy, 0.0), geometry_type=e.dxftype().lower()
                        ))
                if e.dxftype()=='INSERT':
                    name = getattr(e.dxf,'name','').upper()
                    pos = (e.dxf.insert.x, e.dxf.insert.y, e.dxf.insert.z)
                    if 'BUS' in name and 'STOP' in name:
                        elements.append(TransportationElementInfo(
                            element_type=TransportationTrafficElementType.BUS_STOP,
                            layer=layer, position=pos, geometry_type='block', label=getattr(e.dxf,'name',None)
                        ))
                    elif 'DROP' in name:
                        elements.append(TransportationElementInfo(
                            element_type=TransportationTrafficElementType.DROP_OFF,
                            layer=layer, position=pos, geometry_type='block', label=getattr(e.dxf,'name',None)
                        ))
            except:
                continue
        return elements

    def _detect_traffic_calming(self) -> List['TransportationElementInfo']:
        elements: List[TransportationElementInfo] = []
        for e in self.msp:
            try:
                layer = e.dxf.layer if hasattr(e.dxf, 'layer') else ""
                if not self._in_transport_layer(layer):
                    continue
                if e.dxftype() in ['TEXT','MTEXT']:
                    txt = e.dxf.text if e.dxftype()=="TEXT" else (getattr(e,'text',None) or getattr(e,'plain_text',''))
                    low = (txt or '').lower()
                    if any(k in low for k in ['speed bump', 'bump', 'speed table']):
                        pos = (e.dxf.insert.x, e.dxf.insert.y, 0.0) if hasattr(e.dxf,'insert') else (0.0,0.0,0.0)
                        elements.append(TransportationElementInfo(
                            element_type=TransportationTrafficElementType.SPEED_BUMP,
                            layer=layer, position=pos, geometry_type='text', note=txt
                        ))
            except:
                continue
        return elements

    # ---------------------- IT & Network Detection ----------------------
    def _detect_it_network_elements(self) -> List['ITNetworkElementInfo']:
        """تشخیص عناصر زیرساخت IT و شبکه"""
        elements: List[ITNetworkElementInfo] = []
        elements.extend(self._detect_it_blocks())
        elements.extend(self._detect_it_text_ports())
        elements.extend(self._detect_it_cabling())
        elements.extend(self._detect_it_rooms())
        return elements

    def _in_it_layer(self, layer: str) -> bool:
        return any(key in (layer.upper() if layer else "") for key in self.IT_LAYERS)

    def _detect_it_blocks(self) -> List['ITNetworkElementInfo']:
        """تشخیص بلوک‌های تجهیزات شبکه (رک، پچ پنل، AP، سرور)"""
        elements: List[ITNetworkElementInfo] = []
        for e in self.msp:
            try:
                layer = e.dxf.layer if hasattr(e.dxf, 'layer') else ""
                if not self._in_it_layer(layer):
                    continue
                if e.dxftype() == 'INSERT':
                    pos = (e.dxf.insert.x, e.dxf.insert.y, e.dxf.insert.z)
                    name = getattr(e.dxf, 'name', '').lower()
                    et = None
                    if 'rack' in name or 'رک' in name:
                        et = ITNetworkElementType.RACK
                    elif 'patch' in name or 'panel' in name or 'پچ' in name:
                        et = ITNetworkElementType.PATCH_PANEL
                    elif 'ap' in name or 'wifi' in name or 'access point' in name or 'وای فای' in name:
                        et = ITNetworkElementType.WIFI_AP
                    elif 'server' in name or 'سرور' in name:
                        et = ITNetworkElementType.SERVER_EQUIPMENT
                    if et:
                        elements.append(ITNetworkElementInfo(
                            element_type=et, layer=layer, position=pos, geometry_type='block', label=getattr(e.dxf, 'name', None)
                        ))
            except:
                continue
        return elements

    def _detect_it_text_ports(self) -> List['ITNetworkElementInfo']:
        """تشخیص پریزهای شبکه از طریق متن (DATA، RJ45، CAT6، پریز، LAN)"""
        elements: List[ITNetworkElementInfo] = []
        for e in self.msp:
            try:
                layer = e.dxf.layer if hasattr(e.dxf, 'layer') else ""
                if not self._in_it_layer(layer):
                    continue
                if e.dxftype() in ['TEXT', 'MTEXT']:
                    txt = e.dxf.text if e.dxftype() == "TEXT" else (getattr(e, 'text', None) or getattr(e, 'plain_text', ''))
                    low = (txt or '').lower()
                    pos = (e.dxf.insert.x, e.dxf.insert.y, 0.0) if hasattr(e.dxf, 'insert') else (0.0, 0.0, 0.0)
                    if any(kw in low for kw in ['data', 'rj45', 'cat6', 'lan', 'پریز', 'دیتا']):
                        elements.append(ITNetworkElementInfo(
                            element_type=ITNetworkElementType.DATA_PORT, layer=layer, position=pos, geometry_type='text', note=txt
                        ))
            except:
                continue
        return elements

    def _detect_it_cabling(self) -> List['ITNetworkElementInfo']:
        """تشخیص سینی کابل، فیبر، ترانکینگ و لوله ضعیف جریان"""
        elements: List[ITNetworkElementInfo] = []
        for e in self.msp:
            try:
                layer = e.dxf.layer if hasattr(e.dxf, 'layer') else ""
                if not self._in_it_layer(layer):
                    continue
                lname = layer.upper()
                if e.dxftype() in ['LINE', 'LWPOLYLINE']:
                    pts = None
                    if e.dxftype() == 'LWPOLYLINE':
                        pts = list(e.get_points(format='xy'))
                    cx, cy = (0.0, 0.0)
                    if pts:
                        xs = [p[0] for p in pts]
                        ys = [p[1] for p in pts]
                        cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
                    elif e.dxftype() == 'LINE':
                        sx, sy = e.dxf.start.x, e.dxf.start.y
                        ex, ey = e.dxf.end.x, e.dxf.end.y
                        cx, cy = (sx + ex) / 2, (sy + ey) / 2
                    et = None
                    if 'FIBER' in lname or 'FIBRE' in lname or 'BACKBONE' in lname:
                        et = ITNetworkElementType.FIBER_CABLE
                    elif 'TRAY' in lname or 'TRUNKING' in lname or 'TRUNK' in lname:
                        et = ITNetworkElementType.CABLE_TRAY
                    elif 'CONDUIT' in lname:
                        et = ITNetworkElementType.CONDUIT
                    if et:
                        elements.append(ITNetworkElementInfo(
                            element_type=et, layer=layer, position=(cx, cy, 0.0), geometry_type=e.dxftype().lower()
                        ))
            except:
                continue
        return elements

    def _detect_it_rooms(self) -> List['ITNetworkElementInfo']:
        """تشخیص اتاق سرور و اتاق تجهیزات از پلیگون‌ها یا لیبل متن"""
        elements: List[ITNetworkElementInfo] = []
        for e in self.msp:
            try:
                layer = e.dxf.layer if hasattr(e.dxf, 'layer') else ""
                lname = layer.upper()
                # Check polygon layer
                if self._in_it_layer(layer) and e.dxftype() == 'LWPOLYLINE' and ('SERVER' in lname or 'IDF' in lname or 'MDF' in lname):
                    pts = list(e.get_points(format='xy'))
                    if pts:
                        xs = [p[0] for p in pts]
                        ys = [p[1] for p in pts]
                        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
                        elements.append(ITNetworkElementInfo(
                            element_type=ITNetworkElementType.SERVER_ROOM, layer=layer, position=(cx, cy, 0.0), geometry_type='polygon'
                        ))
                # Check text labels
                elif e.dxftype() in ['TEXT', 'MTEXT']:
                    txt = e.dxf.text if e.dxftype() == "TEXT" else (getattr(e, 'text', None) or getattr(e, 'plain_text', ''))
                    low = (txt or '').lower()
                    pos = (e.dxf.insert.x, e.dxf.insert.y, 0.0) if hasattr(e.dxf, 'insert') else (0.0, 0.0, 0.0)
                    if any(kw in low for kw in ['server room', 'idf', 'mdf', 'اتاق سرور', 'سرور']):
                        elements.append(ITNetworkElementInfo(
                            element_type=ITNetworkElementType.SERVER_ROOM, layer=layer, position=pos, geometry_type='text', note=txt
                        ))
            except:
                continue
        return elements

    # ---------------------- Construction Phasing Detection ----------------------
    def _detect_construction_phasing_elements(self) -> List['ConstructionPhasingElementInfo']:
        """تشخیص عناصر مراحل پروژه و ساخت"""
        elements: List[ConstructionPhasingElementInfo] = []
        elements.extend(self._detect_phase_markers())
        elements.extend(self._detect_demolition_zones())
        elements.extend(self._detect_temporary_structures())
        elements.extend(self._detect_construction_sequence())
        elements.extend(self._detect_staging_areas())
        return elements

    def _in_phasing_layer(self, layer: str) -> bool:
        return any(key in (layer.upper() if layer else "") for key in self.PHASING_LAYERS)

    def _detect_phase_markers(self) -> List['ConstructionPhasingElementInfo']:
        """تشخیص نشانگرهای فاز (PHASE 1، PHASE 2، ...) و مرزهای فاز"""
        elements: List[ConstructionPhasingElementInfo] = []
        for e in self.msp:
            try:
                layer = e.dxf.layer if hasattr(e.dxf, 'layer') else ""
                if not self._in_phasing_layer(layer):
                    continue
                if e.dxftype() in ['TEXT', 'MTEXT']:
                    txt = e.dxf.text if e.dxftype() == "TEXT" else (getattr(e, 'text', None) or getattr(e, 'plain_text', ''))
                    low = (txt or '').lower()
                    pos = (e.dxf.insert.x, e.dxf.insert.y, 0.0) if hasattr(e.dxf, 'insert') else (0.0, 0.0, 0.0)
                    if 'phase' in low or 'فاز' in low or 'مرحله' in low:
                        import re
                        phase_num = None
                        m = re.search(r'(\d+)', txt)
                        if m:
                            phase_num = int(m.group(1))
                        elements.append(ConstructionPhasingElementInfo(
                            element_type=ConstructionPhasingElementType.PHASE_MARKER,
                            layer=layer, position=pos, geometry_type='text', phase_number=phase_num, label=txt
                        ))
                elif e.dxftype() == 'LWPOLYLINE' and 'PHASE' in layer.upper():
                    pts = list(e.get_points(format='xy'))
                    if pts:
                        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
                        elements.append(ConstructionPhasingElementInfo(
                            element_type=ConstructionPhasingElementType.PHASE_BOUNDARY,
                            layer=layer, position=(cx, cy, 0.0), geometry_type='polygon'
                        ))
            except:
                continue
        return elements

    def _detect_demolition_zones(self) -> List['ConstructionPhasingElementInfo']:
        """تشخیص مناطق تخریب، دیوارهای قابل تخریب، و موارد خطرناک"""
        elements: List[ConstructionPhasingElementInfo] = []
        for e in self.msp:
            try:
                layer = e.dxf.layer if hasattr(e.dxf, 'layer') else ""
                lname = layer.upper()
                if not self._in_phasing_layer(layer):
                    continue
                if 'DEMO' in lname or 'DEMOLITION' in lname or 'تخریب' in lname:
                    if e.dxftype() == 'LWPOLYLINE':
                        pts = list(e.get_points(format='xy'))
                        if pts:
                            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                            cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
                            et = ConstructionPhasingElementType.DEMOLITION_WALL if len(pts) == 2 else ConstructionPhasingElementType.DEMOLITION_ZONE
                            elements.append(ConstructionPhasingElementInfo(
                                element_type=et, layer=layer, position=(cx, cy, 0.0), geometry_type='polygon', status='demolish'
                            ))
                    elif e.dxftype() in ['TEXT', 'MTEXT']:
                        txt = e.dxf.text if e.dxftype() == "TEXT" else (getattr(e, 'text', None) or getattr(e, 'plain_text', ''))
                        low = txt.lower()
                        pos = (e.dxf.insert.x, e.dxf.insert.y, 0.0) if hasattr(e.dxf, 'insert') else (0.0, 0.0, 0.0)
                        et = ConstructionPhasingElementType.DEMOLITION_ZONE
                        if 'salvage' in low:
                            et = ConstructionPhasingElementType.SALVAGE_ITEM
                        elif 'hazmat' in low or 'asbestos' in low or 'lead' in low:
                            et = ConstructionPhasingElementType.HAZMAT_AREA
                        elements.append(ConstructionPhasingElementInfo(
                            element_type=et, layer=layer, position=pos, geometry_type='text', note=txt, status='demolish'
                        ))
            except:
                continue
        return elements

    def _detect_temporary_structures(self) -> List['ConstructionPhasingElementInfo']:
        """تشخیص سازه‌های موقت: دیوار، حصار، شمع‌کوبی، داربست، مانع"""
        elements: List[ConstructionPhasingElementInfo] = []
        for e in self.msp:
            try:
                layer = e.dxf.layer if hasattr(e.dxf, 'layer') else ""
                lname = layer.upper()
                if not self._in_phasing_layer(layer):
                    continue
                if 'TEMP' in lname or 'TEMPORARY' in lname or 'موقت' in lname or 'SHORING' in lname or 'SCAFFOLD' in lname or 'PROTECTION' in lname or 'BARRIER' in lname:
                    if e.dxftype() in ['LINE', 'LWPOLYLINE']:
                        pts = None
                        if e.dxftype() == 'LWPOLYLINE':
                            pts = list(e.get_points(format='xy'))
                        cx, cy = (0.0, 0.0)
                        if pts:
                            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                            cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
                        elif e.dxftype() == 'LINE':
                            sx, sy = e.dxf.start.x, e.dxf.start.y
                            ex, ey = e.dxf.end.x, e.dxf.end.y
                            cx, cy = (sx + ex) / 2, (sy + ey) / 2
                        et = ConstructionPhasingElementType.TEMPORARY_WALL
                        if 'FENCE' in lname or 'حصار' in lname:
                            et = ConstructionPhasingElementType.TEMPORARY_FENCE
                        elif 'SHORING' in lname or 'نگهداری' in lname:
                            et = ConstructionPhasingElementType.SHORING
                        elif 'SCAFFOLD' in lname or 'داربست' in lname:
                            et = ConstructionPhasingElementType.SCAFFOLDING
                        elif 'PROTECTION' in lname or 'BARRIER' in lname:
                            et = ConstructionPhasingElementType.PROTECTION_BARRIER
                        elements.append(ConstructionPhasingElementInfo(
                            element_type=et, layer=layer, position=(cx, cy, 0.0), geometry_type=e.dxftype().lower(), status='temporary'
                        ))
            except:
                continue
        return elements

    def _detect_construction_sequence(self) -> List['ConstructionPhasingElementInfo']:
        """تشخیص توالی ساخت، بتن‌ریزی، نصب"""
        elements: List[ConstructionPhasingElementInfo] = []
        for e in self.msp:
            try:
                layer = e.dxf.layer if hasattr(e.dxf, 'layer') else ""
                if not self._in_phasing_layer(layer):
                    continue
                if e.dxftype() in ['TEXT', 'MTEXT']:
                    txt = e.dxf.text if e.dxftype() == "TEXT" else (getattr(e, 'text', None) or getattr(e, 'plain_text', ''))
                    low = txt.lower()
                    pos = (e.dxf.insert.x, e.dxf.insert.y, 0.0) if hasattr(e.dxf, 'insert') else (0.0, 0.0, 0.0)
                    et = None
                    seq = None
                    import re
                    m = re.search(r'(\d+)', txt)
                    if m:
                        seq = int(m.group(1))
                    if 'pour' in low or 'بتن' in low:
                        et = ConstructionPhasingElementType.POUR_SEQUENCE
                    elif 'erection' in low or 'install' in low or 'نصب' in low:
                        et = ConstructionPhasingElementType.ERECTION_SEQUENCE
                    elif 'sequence' in low or 'توالی' in low:
                        et = ConstructionPhasingElementType.CONSTRUCTION_SEQUENCE
                    if et:
                        elements.append(ConstructionPhasingElementInfo(
                            element_type=et, layer=layer, position=pos, geometry_type='text', sequence_order=seq, note=txt
                        ))
            except:
                continue
        return elements

    def _detect_staging_areas(self) -> List['ConstructionPhasingElementInfo']:
        """تشخیص مناطق کاری: آماده‌سازی، ذخیره، موقعیت جرثقیل، مسیر دسترسی"""
        elements: List[ConstructionPhasingElementInfo] = []
        for e in self.msp:
            try:
                layer = e.dxf.layer if hasattr(e.dxf, 'layer') else ""
                lname = layer.upper()
                if not self._in_phasing_layer(layer):
                    continue
                if e.dxftype() == 'LWPOLYLINE' and ('STAGING' in lname or 'LAYDOWN' in lname or 'ACCESS' in lname):
                    pts = list(e.get_points(format='xy'))
                    if pts:
                        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
                        et = ConstructionPhasingElementType.STAGING_AREA
                        if 'LAYDOWN' in lname:
                            et = ConstructionPhasingElementType.LAYDOWN_AREA
                        elif 'ACCESS' in lname:
                            et = ConstructionPhasingElementType.ACCESS_ROUTE
                        elements.append(ConstructionPhasingElementInfo(
                            element_type=et, layer=layer, position=(cx, cy, 0.0), geometry_type='polygon'
                        ))
                elif e.dxftype() == 'INSERT' and ('CRANE' in lname or 'جرثقیل' in lname):
                    pos = (e.dxf.insert.x, e.dxf.insert.y, e.dxf.insert.z)
                    elements.append(ConstructionPhasingElementInfo(
                        element_type=ConstructionPhasingElementType.CRANE_LOCATION,
                        layer=layer, position=pos, geometry_type='block', label=getattr(e.dxf, 'name', None)
                    ))
            except:
                continue
        return elements

    def _detect_meters_monitoring(self) -> List['SustainabilityElementInfo']:
        elements: List[SustainabilityElementInfo] = []
        layer_keys = ["METER", "SUB", "BMS", "ENERGY"]
        for e in self.msp:
            try:
                layer = e.dxf.layer if hasattr(e.dxf, 'layer') else ""
                if not any(kw in layer.upper() for kw in layer_keys):
                    continue
                if e.dxftype() == "INSERT":
                    pos = (e.dxf.insert.x, e.dxf.insert.y, e.dxf.insert.z)
                    name = getattr(e.dxf, 'name', '').lower()
                    et = SustainabilityElementType.ENERGY_METER
                    if "sub" in name:
                        et = SustainabilityElementType.SUB_METER
                    elif "bms" in name or "building" in name:
                        et = SustainabilityElementType.BMS_PANEL
                    elements.append(SustainabilityElementInfo(
                        element_type=et, layer=layer, position=pos, geometry_type="block", label=getattr(e.dxf,'name',None)
                    ))
            except:
                continue
        return elements

    def _detect_envelope_insulation(self) -> List['SustainabilityElementInfo']:
        elements: List[SustainabilityElementInfo] = []
        layer_keys = ["INSULATION", "U-VALUE", "R-VALUE", "THERMAL", "عایق"]
        for e in self.msp:
            try:
                layer = e.dxf.layer if hasattr(e.dxf, 'layer') else ""
                if not any(kw in layer.upper() for kw in layer_keys):
                    continue
                if e.dxftype() in ["LWPOLYLINE", "HATCH", "SOLID"]:
                    # zone polygon
                    pos = (0.0, 0.0, 0.0)
                    if e.dxftype() == "LWPOLYLINE":
                        pts = list(e.get_points(format='xy'))
                        if pts:
                            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                            pos = ((min(xs)+max(xs))/2, (min(ys)+max(ys))/2, 0.0)
                    elements.append(SustainabilityElementInfo(
                        element_type=SustainabilityElementType.INSULATION_ZONE,
                        layer=layer, position=pos, geometry_type=e.dxftype().lower()
                    ))
                elif e.dxftype() in ["TEXT", "MTEXT"]:
                    txt = e.dxf.text if e.dxftype()=="TEXT" else (getattr(e,'text',None) or getattr(e,'plain_text',''))
                    low = txt.lower()
                    pos = (e.dxf.insert.x, e.dxf.insert.y, 0.0) if hasattr(e.dxf,'insert') else (0.0,0.0,0.0)
                    et = None
                    if "u-" in low or "u=" in low or "u value" in low:
                        et = SustainabilityElementType.U_VALUE_NOTE
                    elif "r-" in low or "r=" in low or "r value" in low:
                        et = SustainabilityElementType.R_VALUE_NOTE
                    elif "thermal" in low and "break" in low:
                        et = SustainabilityElementType.THERMAL_BREAK
                    if et:
                        elements.append(SustainabilityElementInfo(
                            element_type=et, layer=layer, position=pos, geometry_type="text", note=txt
                        ))
            except:
                continue
        return elements

    def _detect_daylight_ventilation(self) -> List['SustainabilityElementInfo']:
        elements: List[SustainabilityElementInfo] = []
        layer_keys = ["DAYLIGHT", "SKYLIGHT", "SHADING", "LOUVER", "LIGHT", "سایه"]
        for e in self.msp:
            try:
                layer = e.dxf.layer if hasattr(e.dxf, 'layer') else ""
                if not any(kw in layer.upper() for kw in layer_keys):
                    continue
                if e.dxftype() == "INSERT":
                    pos = (e.dxf.insert.x, e.dxf.insert.y, e.dxf.insert.z)
                    name = getattr(e.dxf,'name','').lower()
                    et = SustainabilityElementType.SHADING_DEVICE
                    if "skylight" in name:
                        et = SustainabilityElementType.SKYLIGHT
                    elif "shelf" in name:
                        et = SustainabilityElementType.LIGHT_SHELF
                    elif "louver" in name:
                        et = SustainabilityElementType.LOUVER
                    elements.append(SustainabilityElementInfo(
                        element_type=et, layer=layer, position=pos, geometry_type="block", label=getattr(e.dxf,'name',None)
                    ))
            except:
                continue
        return elements

    def _detect_water_sustainability(self) -> List['SustainabilityElementInfo']:
        elements: List[SustainabilityElementInfo] = []
        layer_keys = ["RAINWATER", "GREYWATER", "WATER", "آب"]
        for e in self.msp:
            try:
                layer = e.dxf.layer if hasattr(e.dxf, 'layer') else ""
                if not any(kw in layer.upper() for kw in layer_keys):
                    continue
                if e.dxftype() == "INSERT":
                    pos = (e.dxf.insert.x, e.dxf.insert.y, e.dxf.insert.z)
                    name = getattr(e.dxf,'name','').lower()
                    et = SustainabilityElementType.RAINWATER_TANK
                    if "filter" in name:
                        if "grey" in name:
                            et = SustainabilityElementType.GREYWATER_FILTER
                        else:
                            et = SustainabilityElementType.RAINWATER_FILTER
                    elif "grey" in name:
                        et = SustainabilityElementType.GREYWATER_TANK
                    elements.append(SustainabilityElementInfo(
                        element_type=et, layer=layer, position=pos, geometry_type="block", label=getattr(e.dxf,'name',None)
                    ))
            except:
                continue
        return elements

    def _detect_green_features(self) -> List['SustainabilityElementInfo']:
        elements: List[SustainabilityElementInfo] = []
        layer_keys = ["GREEN", "ROOF", "WALL", "سبز"]
        for e in self.msp:
            try:
                layer = e.dxf.layer if hasattr(e.dxf, 'layer') else ""
                if not any(kw in layer.upper() for kw in layer_keys):
                    continue
                if e.dxftype() == "INSERT":
                    pos = (e.dxf.insert.x, e.dxf.insert.y, e.dxf.insert.z)
                    name = getattr(e.dxf,'name','').lower()
                    et = SustainabilityElementType.GREEN_ROOF
                    if "wall" in name:
                        et = SustainabilityElementType.GREEN_WALL
                    elements.append(SustainabilityElementInfo(
                        element_type=et, layer=layer, position=pos, geometry_type="block", label=getattr(e.dxf,'name',None)
                    ))
            except:
                continue
        return elements

    def _detect_certification_notes(self) -> List['SustainabilityElementInfo']:
        elements: List[SustainabilityElementInfo] = []
        layer_keys = ["LEED", "BREEAM", "ENERGY", "CODE"]
        for e in self.msp:
            try:
                layer = e.dxf.layer if hasattr(e.dxf, 'layer') else ""
                if not any(kw in layer.upper() for kw in layer_keys):
                    continue
                if e.dxftype() in ["TEXT", "MTEXT"]:
                    txt = e.dxf.text if e.dxftype()=="TEXT" else (getattr(e,'text',None) or getattr(e,'plain_text',''))
                    low = txt.lower()
                    pos = (e.dxf.insert.x, e.dxf.insert.y, 0.0) if hasattr(e.dxf,'insert') else (0.0,0.0,0.0)
                    et = SustainabilityElementType.ENERGY_CODE_NOTE
                    if "leed" in low:
                        et = SustainabilityElementType.LEED_NOTE
                    elif "breeam" in low:
                        et = SustainabilityElementType.BREEAM_NOTE
                    elements.append(SustainabilityElementInfo(
                        element_type=et, layer=layer, position=pos, geometry_type="text", note=txt
                    ))
            except:
                continue
        return elements

    def _detect_energy_zones(self) -> List['SustainabilityElementInfo']:
        elements: List[SustainabilityElementInfo] = []
        layer_keys = ["THERMAL-ZONE", "ENERGY-ZONE", "ZONE"]
        for e in self.msp:
            try:
                layer = e.dxf.layer if hasattr(e.dxf, 'layer') else ""
                if not any(kw in layer.upper() for kw in layer_keys):
                    continue
                if e.dxftype() == "LWPOLYLINE":
                    pts = list(e.get_points(format='xy'))
                    if pts:
                        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                        pos = ((min(xs)+max(xs))/2, (min(ys)+max(ys))/2, 0.0)
                        elements.append(SustainabilityElementInfo(
                            element_type=SustainabilityElementType.ENERGY_ZONE,
                            layer=layer, position=pos, geometry_type="polygon"
                        ))
                elif e.dxftype() in ["TEXT", "MTEXT"]:
                    txt = e.dxf.text if e.dxftype()=="TEXT" else (getattr(e,'text',None) or getattr(e,'plain_text',''))
                    low = txt.lower()
                    pos = (e.dxf.insert.x, e.dxf.insert.y, 0.0) if hasattr(e.dxf,'insert') else (0.0,0.0,0.0)
                    et = SustainabilityElementType.ENERGY_ZONE
                    if "thermal" in low:
                        et = SustainabilityElementType.THERMAL_ZONE
                    elements.append(SustainabilityElementInfo(
                        element_type=et, layer=layer, position=pos, geometry_type="text", note=txt
                    ))
            except:
                continue
        return elements
    
    def _extract_pipe_size(self, text: str) -> str:
        """استخراج قطر لوله از متن"""
        if not text:
            return "Ø??"
        
        import re
        # الگوهای معمول: Ø100, DN100, 100mm, 4"
        patterns = [
            r'[ØDø][-\s]?(\d+)',
            r'DN[-\s]?(\d+)',
            r'(\d+)\s*mm',
            r'(\d+\.?\d*)\s*"',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                size = match.group(1)
                return f"Ø{size}mm"
        
        return text[:20]
    
    def _extract_duct_size(self, text: str) -> str:
        """استخراج ابعاد کانال هوا از متن"""
        if not text:
            return "??"
        
        import re
        # الگوهای معمول: 600x400, 600×400
        pattern = r'(\d+)[x×](\d+)'
        match = re.search(pattern, text)
        
        if match:
            return f"{match.group(1)}×{match.group(2)}mm"
        
        return text[:20]
    
    def _extract_voltage(self, text: str) -> Optional[str]:
        """استخراج ولتاژ از متن"""
        if not text:
            return None
        
        import re
        # الگوهای معمول: 220V, 380V, 12KV
        patterns = [
            r'(\d+)\s*KV',
            r'(\d+)\s*V',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)
        
        return None
    
    def _extract_power(self, text: str) -> Optional[float]:
        """استخراج توان از متن (وات)"""
        if not text:
            return None
        
        import re
        # الگوهای معمول: 2000W, 2KW, 2kW
        patterns = [
            r'(\d+\.?\d*)\s*KW',
            r'(\d+\.?\d*)\s*W',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = float(match.group(1))
                if 'KW' in text.upper():
                    return value * 1000
                return value
        
        return None
    
    def _calculate_building_footprint(self) -> Tuple[float, float, float, float]:
        """محاسبه کادر محیطی کل ساختمان"""
        all_x = []
        all_y = []
        
        for entity in self.msp:
            try:
                if hasattr(entity.dxf, 'insert'):
                    all_x.append(entity.dxf.insert.x)
                    all_y.append(entity.dxf.insert.y)
                elif entity.dxftype() == 'LINE':
                    all_x.extend([entity.dxf.start.x, entity.dxf.end.x])
                    all_y.extend([entity.dxf.start.y, entity.dxf.end.y])
                elif entity.dxftype() == 'LWPOLYLINE':
                    points = list(entity.get_points(format='xy'))
                    all_x.extend(p[0] for p in points)
                    all_y.extend(p[1] for p in points)
            except:
                continue
        
        if not all_x:
            return (0, 0, 0, 0)
        
        return (min(all_x), min(all_y), max(all_x), max(all_y))


def generate_analysis_report(analysis: ArchitecturalAnalysis) -> str:
    """تولید گزارش متنی از تحلیل"""
    report = []
    report.append("=" * 80)
    report.append("گزارش تحلیل نقشه معماری")
    report.append("Architectural Drawing Analysis Report")
    report.append("=" * 80)
    report.append("")
    
    # نوع نقشه
    drawing_type_fa = {
        DrawingType.PLAN: "پلان",
        DrawingType.ELEVATION: "نما",
        DrawingType.SECTION: "برش",
        DrawingType.SITE_PLAN: "پلان سایت",
        DrawingType.DETAIL: "جزئیات",
        DrawingType.UNKNOWN: "نامشخص",
    }
    report.append(f"نوع نقشه: {drawing_type_fa[analysis.drawing_type]} ({analysis.drawing_type.value})")
    report.append("")
    
    # اطلاعات کلی
    report.append("اطلاعات کلی:")
    report.append(f"  تعداد فضاها: {len(analysis.rooms)}")
    report.append(f"  تعداد دیوارها: {len(analysis.walls)}")
    report.append(f"  تعداد ابعاد: {len(analysis.dimensions)}")
    report.append(f"  تعداد عناصر سازه‌ای: {len(analysis.structural_elements)}")
    report.append(f"  تعداد عناصر تأسیساتی: {len(analysis.mep_elements)}")
    report.append(f"  مساحت کل: {analysis.total_area:.2f} متر مربع")
    
    bbox = analysis.building_footprint
    building_width = (bbox[2] - bbox[0]) / 1000
    building_length = (bbox[3] - bbox[1]) / 1000
    report.append(f"  ابعاد کلی ساختمان: {building_width:.2f}m × {building_length:.2f}m")
    report.append("")
    
    # لیست اتاق‌ها
    if analysis.rooms:
        report.append("فضاها و اتاق‌ها:")
        report.append("-" * 80)
        
        space_type_fa = {
            SpaceType.BEDROOM: "اتاق خواب",
            SpaceType.LIVING_ROOM: "پذیرایی",
            SpaceType.KITCHEN: "آشپزخانه",
            SpaceType.BATHROOM: "سرویس بهداشتی",
            SpaceType.CORRIDOR: "راهرو",
            SpaceType.BALCONY: "بالکن",
            SpaceType.PARKING: "پارکینگ",
            SpaceType.STORAGE: "انباری",
            SpaceType.STAIRCASE: "راه پله",
            SpaceType.ELEVATOR: "آسانسور",
            SpaceType.LOBBY: "لابی",
            SpaceType.UNKNOWN: "نامشخص",
        }
        
        for i, room in enumerate(analysis.rooms, 1):
            report.append(f"{i}. {room.name}")
            report.append(f"   نوع: {space_type_fa[room.space_type]} ({room.space_type.value})")
            report.append(f"   مساحت: {room.area:.2f} m²")
            report.append(f"   ابعاد تقریبی: {room.width:.2f}m × {room.length:.2f}m")
            report.append(f"   لایه: {room.layer}")
            if room.text_entities:
                report.append(f"   متن‌های داخلی: {', '.join(room.text_entities[:3])}")
            report.append("")
    
    # عناصر سازه‌ای
    if analysis.structural_elements:
        report.append("عناصر سازه‌ای:")
        report.append("-" * 80)
        
        element_type_fa = {
            StructuralElementType.COLUMN: "ستون",
            StructuralElementType.BEAM: "تیر",
            StructuralElementType.SLAB: "دال",
            StructuralElementType.WALL_BEARING: "دیوار باربر",
            StructuralElementType.FOUNDATION: "فونداسیون",
            StructuralElementType.FOOTING: "پی",
            StructuralElementType.PILE: "شمع",
            StructuralElementType.SHEAR_WALL: "دیوار برشی",
            StructuralElementType.BRACE: "مهاربند",
            StructuralElementType.TRUSS: "خرپا",
            StructuralElementType.REBAR: "آرماتور",
        }
        
        # گروه‌بندی بر اساس نوع
        by_type = {}
        for elem in analysis.structural_elements:
            if elem.element_type not in by_type:
                by_type[elem.element_type] = []
            by_type[elem.element_type].append(elem)
        
        for elem_type, elements in by_type.items():
            report.append(f"\n{element_type_fa[elem_type]} ({elem_type.value}): {len(elements)} عدد")
            for i, elem in enumerate(elements[:5], 1):  # نمایش 5 عنصر اول
                report.append(f"  {i}. {elem.size_designation}")
                if elem.dimensions[0] > 0:
                    report.append(f"     ابعاد: {elem.dimensions[0]:.0f} × {elem.dimensions[1]:.0f} mm")
                report.append(f"     موقعیت: ({elem.position[0]:.0f}, {elem.position[1]:.0f})")
                report.append(f"     لایه: {elem.layer}")
            
            if len(elements) > 5:
                report.append(f"  ... و {len(elements) - 5} عنصر دیگر")
        
        report.append("")
    
    # عناصر تأسیساتی MEP
    if analysis.mep_elements:
        report.append("عناصر تأسیساتی (MEP):")
        report.append("-" * 80)
        
        mep_type_fa = {
            MEPElementType.WATER_PIPE: "لوله آب",
            MEPElementType.DRAIN_PIPE: "لوله فاضلاب",
            MEPElementType.VENT_PIPE: "لوله تهویه",
            MEPElementType.WATER_HEATER: "آبگرمکن",
            MEPElementType.PUMP: "پمپ",
            MEPElementType.VALVE: "شیر",
            MEPElementType.FIXTURE: "شیرآلات",
            MEPElementType.TANK: "مخزن",
            MEPElementType.DUCT: "کانال هوا",
            MEPElementType.DIFFUSER: "دریچه",
            MEPElementType.GRILLE: "گریل",
            MEPElementType.AIR_HANDLER: "هواساز",
            MEPElementType.FAN: "فن",
            MEPElementType.CHILLER: "چیلر",
            MEPElementType.BOILER: "بویلر",
            MEPElementType.COOLING_TOWER: "برج خنک‌کننده",
            MEPElementType.THERMOSTAT: "ترموستات",
            MEPElementType.FCU: "فن کویل",
            MEPElementType.VAV: "VAV",
            MEPElementType.PANEL: "تابلو برق",
            MEPElementType.OUTLET: "پریز",
            MEPElementType.SWITCH: "کلید",
            MEPElementType.LIGHT_FIXTURE: "چراغ",
            MEPElementType.CABLE: "کابل",
            MEPElementType.CONDUIT: "کاندوئیت",
            MEPElementType.TRANSFORMER: "ترانسفورماتور",
            MEPElementType.GENERATOR: "ژنراتور",
            MEPElementType.UPS: "UPS",
            MEPElementType.SPRINKLER: "اسپرینکلر",
            MEPElementType.FIRE_ALARM: "اعلام حریق",
            MEPElementType.SMOKE_DETECTOR: "دتکتور دود",
            MEPElementType.FIRE_EXTINGUISHER: "کپسول آتش‌نشانی",
            MEPElementType.FIRE_HOSE: "شیلنگ آتش‌نشانی",
        }
        
        # گروه‌بندی بر اساس نوع
        by_type = {}
        for elem in analysis.mep_elements:
            if elem.element_type not in by_type:
                by_type[elem.element_type] = []
            by_type[elem.element_type].append(elem)
        
        for elem_type, elements in by_type.items():
            type_name = mep_type_fa.get(elem_type, elem_type.value)
            report.append(f"\n{type_name} ({elem_type.value}): {len(elements)} عدد")
            for i, elem in enumerate(elements[:5], 1):  # نمایش 5 عنصر اول
                report.append(f"  {i}. {elem.size_designation}")
                report.append(f"     موقعیت: ({elem.position[0]:.0f}, {elem.position[1]:.0f})")
                if elem.system:
                    report.append(f"     سیستم: {elem.system}")
                if elem.power:
                    report.append(f"     توان: {elem.power:.0f}W")
                if elem.voltage:
                    report.append(f"     ولتاژ: {elem.voltage}")
                report.append(f"     لایه: {elem.layer}")
            
            if len(elements) > 5:
                report.append(f"  ... و {len(elements) - 5} عنصر دیگر")
        
        report.append("")
    
    # جزئیات اجرایی
    if analysis.construction_details:
        report.append("جزئیات اجرایی (Construction Details):")
        report.append("-" * 80)
        
        detail_type_fa = {
            DetailType.BEAM_COLUMN_CONNECTION: "اتصال تیر به ستون",
            DetailType.WALL_FOUNDATION_CONNECTION: "اتصال دیوار به فونداسیون",
            DetailType.SLAB_WALL_CONNECTION: "اتصال دال به دیوار",
            DetailType.ROOF_WALL_CONNECTION: "اتصال سقف به دیوار",
            DetailType.EXPANSION_JOINT: "درز انبساط",
            DetailType.DOOR_DETAIL: "جزئیات درب",
            DetailType.WINDOW_DETAIL: "جزئیات پنجره",
            DetailType.DOOR_FRAME: "چارچوب درب",
            DetailType.WINDOW_FRAME: "چارچوب پنجره",
            DetailType.THRESHOLD: "آستانه",
            DetailType.FACADE_PANEL: "پانل نما",
            DetailType.CLADDING: "روکش نما",
            DetailType.CURTAIN_WALL: "دیوار پرده‌ای",
            DetailType.STONE_CLADDING: "نمای سنگ",
            DetailType.METAL_PANEL: "پانل فلزی",
            DetailType.FLOOR_FINISH: "پوشش کف",
            DetailType.CEILING_DETAIL: "جزئیات سقف",
            DetailType.FALSE_CEILING: "سقف کاذب",
            DetailType.FLOORING_SECTION: "مقطع کف‌سازی",
            DetailType.WATERPROOFING: "آب‌بندی",
            DetailType.INSULATION: "عایق",
            DetailType.VAPOR_BARRIER: "عایق رطوبتی",
            DetailType.DAMP_PROOF_COURSE: "لایه ضد رطوبت",
            DetailType.WALL_SECTION: "مقطع دیوار",
            DetailType.PARTITION_WALL: "دیوار پارتیشن",
            DetailType.RETAINING_WALL: "دیوار حائل",
            DetailType.STAIR_DETAIL: "جزئیات پله",
            DetailType.RAILING_DETAIL: "جزئیات نرده",
            DetailType.HANDRAIL: "دست‌انداز",
            DetailType.ROOF_SECTION: "مقطع سقف",
            DetailType.ROOF_EDGE: "لبه سقف",
            DetailType.GUTTER: "ناودان",
            DetailType.FLASHING: "برگه‌های آب‌بندی",
            DetailType.SKYLIGHT: "نورگیر",
        }
        
        material_type_fa = {
            MaterialType.CONCRETE: "بتن",
            MaterialType.STEEL: "فولاد",
            MaterialType.BRICK: "آجر",
            MaterialType.BLOCK: "بلوک",
            MaterialType.STONE: "سنگ",
            MaterialType.WOOD: "چوب",
            MaterialType.GLASS: "شیشه",
            MaterialType.ALUMINUM: "آلومینیوم",
            MaterialType.COMPOSITE_PANEL: "پانل کامپوزیت",
            MaterialType.CERAMIC: "سرامیک",
            MaterialType.FOAM_INSULATION: "عایق فوم",
            MaterialType.MINERAL_WOOL: "پشم سنگ",
            MaterialType.POLYSTYRENE: "پلی‌استایرن",
            MaterialType.BITUMEN: "قیر",
            MaterialType.MEMBRANE: "غشا",
            MaterialType.SEALANT: "درزگیر",
            MaterialType.PLASTER: "گچ",
            MaterialType.MORTAR: "ملات",
            MaterialType.PAINT: "رنگ",
        }
        
        # گروه‌بندی بر اساس نوع
        by_type = {}
        for detail in analysis.construction_details:
            if detail.detail_type not in by_type:
                by_type[detail.detail_type] = []
            by_type[detail.detail_type].append(detail)
        
        for detail_type, details_list in by_type.items():
            type_name = detail_type_fa.get(detail_type, detail_type.value)
            report.append(f"\n{type_name} ({detail_type.value}): {len(details_list)} عدد")
            for i, detail in enumerate(details_list[:3], 1):  # نمایش 3 جزئیات اول
                report.append(f"  {i}. موقعیت: ({detail.position[0]:.0f}, {detail.position[1]:.0f})")
                if detail.scale:
                    report.append(f"     مقیاس: {detail.scale}")
                if detail.materials:
                    materials_str = ", ".join(material_type_fa.get(m, m.value) for m in detail.materials[:3])
                    report.append(f"     مصالح: {materials_str}")
                if detail.annotations:
                    report.append(f"     یادداشت‌ها: {len(detail.annotations)} مورد")
                report.append(f"     لایه: {detail.layer}")
            
            if len(details_list) > 3:
                report.append(f"  ... و {len(details_list) - 3} جزئیات دیگر")
        
        report.append("")
    
    # عناصر سایت
    if analysis.site_elements:
        report.append(f"عناصر سایت: {len(analysis.site_elements)} عنصر")
        
        # ترجمه فارسی
        site_element_fa = {
            SiteElementType.MAIN_BUILDING: "ساختمان اصلی",
            SiteElementType.ADJACENT_BUILDING: "ساختمان مجاور",
            SiteElementType.EXISTING_BUILDING: "ساختمان موجود",
            SiteElementType.PROPERTY_LINE: "خط ملک",
            SiteElementType.SETBACK_LINE: "خط عقب‌نشینی",
            SiteElementType.BUILDING_LINE: "خط ساختمان",
            SiteElementType.ROAD: "جاده",
            SiteElementType.DRIVEWAY: "ورودی اتومبیل",
            SiteElementType.PARKING_LOT: "محوطه پارکینگ",
            SiteElementType.PARKING_SPACE: "جای پارک",
            SiteElementType.WALKWAY: "پیاده‌رو",
            SiteElementType.SIDEWALK: "کف‌پوش پیاده",
            SiteElementType.ENTRY_GATE: "دروازه ورودی",
            SiteElementType.TREE: "درخت",
            SiteElementType.SHRUB: "بوته",
            SiteElementType.GRASS_AREA: "چمن",
            SiteElementType.GARDEN: "باغچه",
            SiteElementType.PLANTER: "گلدان",
            SiteElementType.HEDGE: "پرچین",
            SiteElementType.FENCE: "حصار",
            SiteElementType.WALL: "دیوار",
            SiteElementType.RETAINING_WALL: "دیوار حائل",
            SiteElementType.POOL: "استخر",
            SiteElementType.FOUNTAIN: "فواره",
            SiteElementType.PATIO: "پاسیو",
            SiteElementType.DECK: "عرشه",
            SiteElementType.UTILITY_LINE: "خط تاسیسات",
            SiteElementType.MANHOLE: "منهول",
            SiteElementType.CATCH_BASIN: "حوضچه جمع‌آوری",
            SiteElementType.LIGHT_POLE: "پایه چراغ",
            SiteElementType.SIGNAGE: "تابلو",
            SiteElementType.CONTOUR_LINE: "خط تراز",
            SiteElementType.SPOT_ELEVATION: "ارتفاع نقطه‌ای",
            SiteElementType.SLOPE: "شیب",
            SiteElementType.NORTH_ARROW: "جهت شمال",
            SiteElementType.SCALE_BAR: "خط مقیاس",
            SiteElementType.UNKNOWN: "نامشخص",
        }
        
        # گروه‌بندی بر اساس نوع
        by_type = {}
        for elem in analysis.site_elements:
            if elem.element_type not in by_type:
                by_type[elem.element_type] = []
            by_type[elem.element_type].append(elem)
        
        for elem_type, elems_list in by_type.items():
            type_name = site_element_fa.get(elem_type, elem_type.value)
            report.append(f"\n{type_name} ({elem_type.value}): {len(elems_list)} عدد")
            for i, elem in enumerate(elems_list[:3], 1):  # نمایش 3 عنصر اول
                report.append(f"  {i}. موقعیت: ({elem.position[0]:.0f}, {elem.position[1]:.0f})")
                if elem.area and elem.area > 0:
                    report.append(f"     مساحت: {elem.area:.1f} متر مربع")
                if elem.length and elem.length > 0:
                    report.append(f"     طول: {elem.length:.1f} متر")
                if elem.distance_to_main and elem.distance_to_main > 0:
                    report.append(f"     فاصله تا ساختمان اصلی: {elem.distance_to_main:.1f} متر")
                if elem.label:
                    report.append(f"     برچسب: {elem.label}")
                report.append(f"     لایه: {elem.layer}")
            
            if len(elems_list) > 3:
                report.append(f"  ... و {len(elems_list) - 3} عنصر دیگر")
        
        # آمار کلی
        total_building_area = sum(e.area for e in analysis.site_elements 
                                   if e.area and "BUILDING" in e.element_type.value)
        total_parking_area = sum(e.area for e in analysis.site_elements 
                                  if e.area and "PARKING" in e.element_type.value)
        total_landscape_area = sum(e.area for e in analysis.site_elements 
                                    if e.area and e.element_type in [SiteElementType.GRASS_AREA, SiteElementType.GARDEN])
        num_trees = len([e for e in analysis.site_elements if e.element_type == SiteElementType.TREE])
        
        if total_building_area > 0 or total_parking_area > 0 or total_landscape_area > 0 or num_trees > 0:
            report.append("\nآمار کلی سایت:")
            if total_building_area > 0:
                report.append(f"  مساحت کل ساختمان‌ها: {total_building_area:.1f} متر مربع")
            if total_parking_area > 0:
                report.append(f"  مساحت کل پارکینگ: {total_parking_area:.1f} متر مربع")
            if total_landscape_area > 0:
                report.append(f"  مساحت کل فضای سبز: {total_landscape_area:.1f} متر مربع")
            if num_trees > 0:
                report.append(f"  تعداد درختان: {num_trees} عدد")
        
        report.append("")
    
    # آمار لایه‌ها
    if analysis.layers_info:
        report.append("آمار لایه‌ها:")
        for layer, count in sorted(analysis.layers_info.items(), key=lambda x: x[1], reverse=True)[:10]:
            report.append(f"  {layer}: {count} entity")
        report.append("")
    
    report.append("=" * 80)
    
    return "\n".join(report)
