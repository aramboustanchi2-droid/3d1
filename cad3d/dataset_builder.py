"""
Dataset Builder for Architectural Learning
سازنده مجموعه داده برای آموزش درک نقشه‌های معماری
"""
from __future__ import annotations
import json
import csv
from pathlib import Path
from typing import List, Dict
from dataclasses import asdict

from .architectural_analyzer import (
    ArchitecturalAnalyzer,
    ArchitecturalAnalysis,
    SpaceType,
    DrawingType,
    StructuralElementType,
    MEPElementType,
    DetailType,
    MaterialType,
    SiteElementType,
    CivilElementType,
    InteriorElementType,
    SafetySecurityElementType,
    AdvancedStructuralElementType,
    SpecialEquipmentElementType,
    RegulatoryComplianceElementType,
    SustainabilityElementType,
    TransportationTrafficElementType,
    ITNetworkElementType,
    ConstructionPhasingElementType,
)


class ArchitecturalDatasetBuilder:
    """سازنده dataset برای یادگیری نقشه‌های معماری"""
    
    def __init__(self, output_dir: str):
        """
        Args:
            output_dir: مسیر پوشه خروجی dataset
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.analyses: List[ArchitecturalAnalysis] = []
        self.statistics = {
            "total_drawings": 0,
            "total_rooms": 0,
            "total_walls": 0,
            "total_structural_elements": 0,
            "total_mep_elements": 0,
            "total_construction_details": 0,
            "total_site_elements": 0,
            "total_civil_elements": 0,
            "total_interior_elements": 0,
            "total_safety_security_elements": 0,
            "total_advanced_structural_elements": 0,
            "total_special_equipment_elements": 0,
            "total_regulatory_elements": 0,
            "total_sustainability_elements": 0,
            "total_transportation_elements": 0,
            "total_it_network_elements": 0,
            "total_construction_phasing_elements": 0,
            "space_types_count": {},
            "drawing_types_count": {},
            "structural_element_types_count": {},
            "mep_element_types_count": {},
            "detail_types_count": {},
            "material_types_count": {},
            "site_element_types_count": {},
            "civil_element_types_count": {},
            "interior_element_types_count": {},
            "safety_security_element_types_count": {},
            "advanced_structural_element_types_count": {},
            "special_equipment_element_types_count": {},
            "regulatory_element_types_count": {},
            "sustainability_element_types_count": {},
            "transportation_element_types_count": {},
            "it_network_element_types_count": {},
            "construction_phasing_element_types_count": {},
            "total_area": 0,
        }
    
    def process_drawing(self, dxf_path: str) -> ArchitecturalAnalysis:
        """
        پردازش یک نقشه و افزودن به dataset
        
        Args:
            dxf_path: مسیر فایل DXF
            
        Returns:
            ArchitecturalAnalysis
        """
        analyzer = ArchitecturalAnalyzer(dxf_path)
        analysis = analyzer.analyze()
        
        self.analyses.append(analysis)
        self._update_statistics(analysis)
        
        return analysis
    
    def process_folder(self, folder_path: str, recursive: bool = True):
        """
        پردازش یک پوشه کامل از نقشه‌ها
        
        Args:
            folder_path: مسیر پوشه
            recursive: جستجوی بازگشتی در زیر-پوشه‌ها
        """
        folder = Path(folder_path)
        
        if recursive:
            dxf_files = list(folder.rglob("*.dxf"))
            dxf_files.extend(folder.rglob("*.DXF"))
        else:
            dxf_files = list(folder.glob("*.dxf"))
            dxf_files.extend(folder.glob("*.DXF"))
        
        print(f"یافت شد {len(dxf_files)} فایل DXF")
        
        for i, dxf_file in enumerate(dxf_files, 1):
            try:
                print(f"[{i}/{len(dxf_files)}] پردازش: {dxf_file.name}")
                self.process_drawing(str(dxf_file))
            except Exception as e:
                print(f"  ❌ خطا: {e}")
                continue
        
        print(f"\n✅ تکمیل: {len(self.analyses)} نقشه پردازش شد")
    
    def _update_statistics(self, analysis: ArchitecturalAnalysis):
        """به‌روزرسانی آمار dataset"""
        self.statistics["total_drawings"] += 1
        self.statistics["total_rooms"] += len(analysis.rooms)
        self.statistics["total_walls"] += len(analysis.walls)
        self.statistics["total_area"] += analysis.total_area
        
        # شمارش انواع نقشه
        dt = analysis.drawing_type.value
        self.statistics["drawing_types_count"][dt] = \
            self.statistics["drawing_types_count"].get(dt, 0) + 1
        
        # شمارش انواع فضاها
        for room in analysis.rooms:
            st = room.space_type.value
            self.statistics["space_types_count"][st] = \
                self.statistics["space_types_count"].get(st, 0) + 1
        
        # شمارش عناصر سازه‌ای
        self.statistics["total_structural_elements"] += len(analysis.structural_elements)
        for elem in analysis.structural_elements:
            et = elem.element_type.value
            self.statistics["structural_element_types_count"][et] = \
                self.statistics["structural_element_types_count"].get(et, 0) + 1
        
        # شمارش عناصر تأسیساتی
        self.statistics["total_mep_elements"] += len(analysis.mep_elements)
        for elem in analysis.mep_elements:
            et = elem.element_type.value
            self.statistics["mep_element_types_count"][et] = \
                self.statistics["mep_element_types_count"].get(et, 0) + 1
        
        # شمارش جزئیات اجرایی
        self.statistics["total_construction_details"] += len(analysis.construction_details)
        for detail in analysis.construction_details:
            dt = detail.detail_type.value
            self.statistics["detail_types_count"][dt] = \
                self.statistics["detail_types_count"].get(dt, 0) + 1
            
            # شمارش مصالح
            if detail.materials:
                for material in detail.materials:
                    mt = material.value
                    self.statistics["material_types_count"][mt] = \
                        self.statistics["material_types_count"].get(mt, 0) + 1
        
        # شمارش عناصر سایت
        self.statistics["total_site_elements"] += len(analysis.site_elements)
        for elem in analysis.site_elements:
            et = elem.element_type.value
            self.statistics["site_element_types_count"][et] = \
                self.statistics["site_element_types_count"].get(et, 0) + 1
        
        # شمارش عناصر مهندسی سایت
        self.statistics["total_civil_elements"] += len(analysis.civil_elements)
        for elem in analysis.civil_elements:
            et = elem.element_type.value
            self.statistics["civil_element_types_count"][et] = \
                self.statistics["civil_element_types_count"].get(et, 0) + 1
        
        # شمارش عناصر معماری داخلی
        self.statistics["total_interior_elements"] += len(analysis.interior_elements)
        for elem in analysis.interior_elements:
            et = elem.element_type.value
            self.statistics["interior_element_types_count"][et] = \
                self.statistics["interior_element_types_count"].get(et, 0) + 1
        
        # شمارش عناصر ایمنی و امنیت
        self.statistics["total_safety_security_elements"] += len(analysis.safety_security_elements)
        for elem in analysis.safety_security_elements:
            et = elem.element_type.value
            self.statistics["safety_security_element_types_count"][et] = \
                self.statistics["safety_security_element_types_count"].get(et, 0) + 1

        # شمارش عناصر تحلیل پیشرفته سازه‌ای
        if hasattr(analysis, 'advanced_structural_elements'):
            self.statistics["total_advanced_structural_elements"] += len(analysis.advanced_structural_elements)
            for elem in analysis.advanced_structural_elements:
                et = elem.element_type.value
                self.statistics["advanced_structural_element_types_count"][et] = \
                    self.statistics["advanced_structural_element_types_count"].get(et, 0) + 1

        # شمارش تجهیزات ویژه
        if hasattr(analysis, 'special_equipment_elements'):
            self.statistics["total_special_equipment_elements"] += len(analysis.special_equipment_elements)
            for elem in analysis.special_equipment_elements:
                et = elem.element_type.value
                self.statistics["special_equipment_element_types_count"][et] = \
                    self.statistics["special_equipment_element_types_count"].get(et, 0) + 1

        # شمارش ضوابط و مقررات
        if hasattr(analysis, 'regulatory_elements'):
            self.statistics["total_regulatory_elements"] += len(analysis.regulatory_elements)
            for elem in analysis.regulatory_elements:
                et = elem.element_type.value
                self.statistics["regulatory_element_types_count"][et] = \
                    self.statistics["regulatory_element_types_count"].get(et, 0) + 1

        # شمارش عناصر پایداری و انرژی
        if hasattr(analysis, 'sustainability_elements'):
            self.statistics["total_sustainability_elements"] += len(analysis.sustainability_elements)
            for elem in analysis.sustainability_elements:
                et = elem.element_type.value
                self.statistics["sustainability_element_types_count"][et] = \
                    self.statistics["sustainability_element_types_count"].get(et, 0) + 1
        
        # شمارش عناصر حمل‌ونقل و ترافیک
        if hasattr(analysis, 'transportation_elements'):
            self.statistics["total_transportation_elements"] += len(analysis.transportation_elements)
            for elem in analysis.transportation_elements:
                et = elem.element_type.value
                self.statistics["transportation_element_types_count"][et] = \
                    self.statistics["transportation_element_types_count"].get(et, 0) + 1

        # شمارش عناصر زیرساخت IT و شبکه
        if hasattr(analysis, 'it_network_elements'):
            self.statistics["total_it_network_elements"] += len(analysis.it_network_elements)
            for elem in analysis.it_network_elements:
                et = elem.element_type.value
                self.statistics["it_network_element_types_count"][et] = \
                    self.statistics["it_network_element_types_count"].get(et, 0) + 1

        # شمارش عناصر مراحل پروژه و ساخت
        if hasattr(analysis, 'construction_phasing_elements'):
            self.statistics["total_construction_phasing_elements"] += len(analysis.construction_phasing_elements)
            for elem in analysis.construction_phasing_elements:
                et = elem.element_type.value
                self.statistics["construction_phasing_element_types_count"][et] = \
                    self.statistics["construction_phasing_element_types_count"].get(et, 0) + 1
    
    def export_to_json(self, filename: str = "architectural_dataset.json"):
        """ذخیره dataset به فرمت JSON"""
        output_path = self.output_dir / filename
        
        # تبدیل به dict قابل serialize
        dataset = {
            "metadata": {
                "total_drawings": len(self.analyses),
                "statistics": self.statistics,
            },
            "drawings": []
        }
        
        for analysis in self.analyses:
            drawing_data = {
                "drawing_type": analysis.drawing_type.value,
                "total_area": analysis.total_area,
                "building_footprint": {
                    "min_x": analysis.building_footprint[0],
                    "min_y": analysis.building_footprint[1],
                    "max_x": analysis.building_footprint[2],
                    "max_y": analysis.building_footprint[3],
                },
                "layers_info": analysis.layers_info,
                "rooms": [
                    {
                        "name": room.name,
                        "space_type": room.space_type.value,
                        "area": room.area,
                        "width": room.width,
                        "length": room.length,
                        "perimeter": room.perimeter,
                        "layer": room.layer,
                        "center": {"x": room.center[0], "y": room.center[1]},
                        "text_entities": room.text_entities,
                    }
                    for room in analysis.rooms
                ],
                "walls": [
                    {
                        "length": wall.length,
                        "thickness": wall.thickness,
                        "layer": wall.layer,
                        "is_load_bearing": wall.is_load_bearing,
                        "is_exterior": wall.is_exterior,
                    }
                    for wall in analysis.walls
                ],
                "dimensions": [
                    {
                        "value": dim.value,
                        "text": dim.text,
                        "layer": dim.layer,
                    }
                    for dim in analysis.dimensions
                ],
                "structural_elements": [
                    {
                        "element_type": elem.element_type.value,
                        "position": {"x": elem.position[0], "y": elem.position[1], "z": elem.position[2]},
                        "dimensions": {"width": elem.dimensions[0], "depth": elem.dimensions[1], "height": elem.dimensions[2]},
                        "size_designation": elem.size_designation,
                        "layer": elem.layer,
                        "material": elem.material,
                        "reinforcement": elem.reinforcement,
                        "load_capacity": elem.load_capacity,
                    }
                    for elem in analysis.structural_elements
                ],
                "metadata": analysis.metadata,
            }
            dataset["drawings"].append(drawing_data)
        
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Dataset JSON ذخیره شد: {output_path}")
        return output_path
    
    def export_rooms_to_csv(self, filename: str = "rooms_dataset.csv"):
        """ذخیره اطلاعات اتاق‌ها به CSV"""
        output_path = self.output_dir / filename
        
        with output_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow([
                "drawing_file", "room_name", "space_type", "area_m2",
                "width_m", "length_m", "perimeter_m", "layer", "text_entities"
            ])
            
            for analysis in self.analyses:
                drawing_file = analysis.metadata.get("file_path", "unknown")
                for room in analysis.rooms:
                    writer.writerow([
                        drawing_file,
                        room.name,
                        room.space_type.value,
                        f"{room.area:.2f}",
                        f"{room.width:.2f}",
                        f"{room.length:.2f}",
                        f"{room.perimeter:.2f}",
                        room.layer,
                        "; ".join(room.text_entities),
                    ])
        
        print(f"✅ Rooms CSV ذخیره شد: {output_path}")
        return output_path
    
    def export_structural_elements_to_csv(self, filename: str = "structural_elements_dataset.csv"):
        """ذخیره اطلاعات عناصر سازه‌ای به CSV"""
        output_path = self.output_dir / filename
        
        with output_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow([
                "drawing_file", "element_type", "size_designation", "width", "depth", "height",
                "position_x", "position_y", "layer", "material", "reinforcement", "load_capacity"
            ])
            
            for analysis in self.analyses:
                drawing_file = analysis.metadata.get("file_path", "unknown")
                for elem in analysis.structural_elements:
                    writer.writerow([
                        drawing_file,
                        elem.element_type.value,
                        elem.size_designation,
                        f"{elem.dimensions[0]:.1f}",
                        f"{elem.dimensions[1]:.1f}",
                        f"{elem.dimensions[2]:.1f}",
                        f"{elem.position[0]:.1f}",
                        f"{elem.position[1]:.1f}",
                        elem.layer,
                        elem.material or "",
                        elem.reinforcement or "",
                        elem.load_capacity or "",
                    ])
        
        print(f"✅ Structural Elements CSV ذخیره شد: {output_path}")
        return output_path
    
    def export_statistics(self, filename: str = "dataset_statistics.json"):
        """ذخیره آمار dataset"""
        output_path = self.output_dir / filename
        
        # محاسبه آمار پیشرفته
        advanced_stats = {
            **self.statistics,
            "average_rooms_per_drawing": self.statistics["total_rooms"] / max(1, self.statistics["total_drawings"]),
            "average_walls_per_drawing": self.statistics["total_walls"] / max(1, self.statistics["total_drawings"]),
            "average_structural_elements_per_drawing": self.statistics["total_structural_elements"] / max(1, self.statistics["total_drawings"]),
            "average_area_per_drawing": self.statistics["total_area"] / max(1, self.statistics["total_drawings"]),
        }
        
        # محاسبه میانگین مساحت هر نوع فضا
        space_areas = {}
        space_counts = {}
        for analysis in self.analyses:
            for room in analysis.rooms:
                st = room.space_type.value
                space_areas[st] = space_areas.get(st, 0) + room.area
                space_counts[st] = space_counts.get(st, 0) + 1
        
        advanced_stats["average_area_by_space_type"] = {
            st: space_areas[st] / space_counts[st]
            for st in space_areas
        }
        
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(advanced_stats, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Statistics ذخیره شد: {output_path}")
        return output_path
    
    def generate_summary_report(self) -> str:
        """تولید گزارش خلاصه dataset"""
        lines = []
        lines.append("=" * 80)
        lines.append("گزارش خلاصه Dataset نقشه‌های معماری")
        lines.append("Architectural Drawings Dataset Summary")
        lines.append("=" * 80)
        lines.append("")
        
        lines.append(f"📊 تعداد کل نقشه‌ها: {self.statistics['total_drawings']}")
        lines.append(f"🏠 تعداد کل فضاها: {self.statistics['total_rooms']}")
        lines.append(f"🧱 تعداد کل دیوارها: {self.statistics['total_walls']}")
        lines.append(f"🏗️ تعداد کل عناصر سازه‌ای: {self.statistics['total_structural_elements']}")
        lines.append(f"📐 مساحت کل: {self.statistics['total_area']:.2f} متر مربع")
        lines.append("")
        
        if self.statistics['total_drawings'] > 0:
            lines.append("میانگین‌ها:")
            lines.append(f"  • فضا در هر نقشه: {self.statistics['total_rooms'] / self.statistics['total_drawings']:.1f}")
            lines.append(f"  • دیوار در هر نقشه: {self.statistics['total_walls'] / self.statistics['total_drawings']:.1f}")
            lines.append(f"  • عناصر سازه‌ای در هر نقشه: {self.statistics['total_structural_elements'] / self.statistics['total_drawings']:.1f}")
            lines.append(f"  • مساحت هر نقشه: {self.statistics['total_area'] / self.statistics['total_drawings']:.2f} m²")
            lines.append("")
        
        lines.append("انواع نقشه:")
        for dt, count in sorted(self.statistics['drawing_types_count'].items(), key=lambda x: x[1], reverse=True):
            percentage = (count / self.statistics['total_drawings'] * 100) if self.statistics['total_drawings'] > 0 else 0
            lines.append(f"  • {dt}: {count} ({percentage:.1f}%)")
        lines.append("")
        
        lines.append("انواع فضا:")
        for st, count in sorted(self.statistics['space_types_count'].items(), key=lambda x: x[1], reverse=True):
            percentage = (count / self.statistics['total_rooms'] * 100) if self.statistics['total_rooms'] > 0 else 0
            lines.append(f"  • {st}: {count} ({percentage:.1f}%)")
        lines.append("")
        
        if self.statistics['structural_element_types_count']:
            lines.append("انواع عناصر سازه‌ای:")
            for et, count in sorted(self.statistics['structural_element_types_count'].items(), key=lambda x: x[1], reverse=True):
                percentage = (count / self.statistics['total_structural_elements'] * 100) if self.statistics['total_structural_elements'] > 0 else 0
                lines.append(f"  • {et}: {count} ({percentage:.1f}%)")
            lines.append("")
        
        lines.append("=" * 80)
        
        return "\n".join(lines)
