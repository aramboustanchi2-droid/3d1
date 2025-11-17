"""
CAD Training Dataset Builder - ساخت Dataset برای آموزش مدل‌های AI
ساخت dataset برای:
- Object Detection (Bounding Boxes)
- Semantic Segmentation (Pixel Masks)
- Instance Segmentation
- OCR Training
"""

from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
from dataclasses import dataclass, asdict
import json
import numpy as np
from PIL import Image, ImageDraw
import cv2

try:
    import ezdxf
    EZDXF_AVAILABLE = True
except ImportError:
    EZDXF_AVAILABLE = False


@dataclass
class BoundingBox:
    """Bounding box برای Object Detection"""
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    category: str
    category_id: int
    confidence: float = 1.0


@dataclass
class Annotation:
    """Annotation کامل برای یک تصویر"""
    image_id: int
    image_path: str
    image_width: int
    image_height: int
    bboxes: List[BoundingBox]
    segmentation_mask: Optional[np.ndarray] = None
    metadata: Optional[Dict] = None


class CADDatasetBuilder:
    """
    ساخت Dataset برای Training مدل‌های CAD
    فرمت‌های خروجی: COCO, YOLO, Pascal VOC
    """
    
    def __init__(self, output_dir: Union[str, Path]):
        """
        Args:
            output_dir: مسیر ذخیره dataset
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # دسته‌بندی‌های CAD (15 کلاس اصلی)
        self.categories = {
            1: "wall",
            2: "door", 
            3: "window",
            4: "column",
            5: "beam",
            6: "slab",
            7: "hvac",
            8: "plumbing",
            9: "electrical",
            10: "furniture",
            11: "equipment",
            12: "dimension",
            13: "text",
            14: "symbol",
            15: "grid_line"
        }
        
        self.annotations: List[Annotation] = []
        self.image_counter = 0
        
        print(f"📦 CAD Dataset Builder initialized")
        print(f"   Output: {self.output_dir}")
        print(f"   Categories: {len(self.categories)}")
    
    def add_dxf_to_dataset(
        self,
        dxf_path: Union[str, Path],
        render_image: bool = True,
        image_size: Tuple[int, int] = (1024, 1024),
        dpi: int = 300
    ) -> Annotation:
        """
        اضافه کردن فایل DXF به dataset
        
        Args:
            dxf_path: مسیر فایل DXF
            render_image: رندر DXF به تصویر
            image_size: اندازه تصویر خروجی
            dpi: وضوح
        
        Returns:
            Annotation object
        """
        if not EZDXF_AVAILABLE:
            raise ImportError("ezdxf required for DXF processing")
        
        dxf_path = Path(dxf_path)
        self.image_counter += 1
        
        print(f"📄 Processing DXF {self.image_counter}: {dxf_path.name}")
        
        # بارگذاری DXF
        doc = ezdxf.readfile(str(dxf_path))
        msp = doc.modelspace()
        
        # استخراج Bounding Boxes
        bboxes = self._extract_bboxes_from_dxf(msp, image_size)
        
        # رندر به تصویر
        if render_image:
            image_path = self.output_dir / "images" / f"cad_{self.image_counter:05d}.png"
            image_path.parent.mkdir(exist_ok=True)
            self._render_dxf_to_image(doc, image_path, image_size, dpi)
        else:
            image_path = dxf_path
        
        # ساخت annotation
        annotation = Annotation(
            image_id=self.image_counter,
            image_path=str(image_path),
            image_width=image_size[0],
            image_height=image_size[1],
            bboxes=bboxes,
            metadata={
                "source_dxf": str(dxf_path),
                "dpi": dpi
            }
        )
        
        self.annotations.append(annotation)
        print(f"   ✅ Added {len(bboxes)} annotations")
        
        return annotation
    
    def _extract_bboxes_from_dxf(
        self,
        msp,
        image_size: Tuple[int, int]
    ) -> List[BoundingBox]:
        """استخراج Bounding Boxes از المان‌های DXF"""
        bboxes = []
        
        # محاسبه مقیاس
        extents = self._calculate_extents(msp)
        if not extents:
            return bboxes
        
        min_x, min_y, max_x, max_y = extents
        scale_x = image_size[0] / (max_x - min_x) if (max_x - min_x) > 0 else 1
        scale_y = image_size[1] / (max_y - min_y) if (max_y - min_y) > 0 else 1
        scale = min(scale_x, scale_y) * 0.9  # 90% برای حاشیه
        
        for entity in msp:
            try:
                layer = entity.dxf.layer if hasattr(entity.dxf, 'layer') else ""
                
                # تشخیص نوع المان و دسته‌بندی
                category, cat_id = self._classify_entity(entity, layer)
                if not category:
                    continue
                
                # محاسبه bounding box
                bbox_coords = self._get_entity_bbox(entity, min_x, min_y, scale, image_size)
                if not bbox_coords:
                    continue
                
                x1, y1, x2, y2 = bbox_coords
                
                # بررسی معتبر بودن bbox
                if x2 <= x1 or y2 <= y1:
                    continue
                if x1 < 0 or y1 < 0 or x2 > image_size[0] or y2 > image_size[1]:
                    continue
                
                bboxes.append(BoundingBox(
                    x_min=x1,
                    y_min=y1,
                    x_max=x2,
                    y_max=y2,
                    category=category,
                    category_id=cat_id
                ))
            except:
                continue
        
        return bboxes
    
    def _calculate_extents(self, msp) -> Optional[Tuple[float, float, float, float]]:
        """محاسبه محدوده کلی نقشه"""
        min_x = min_y = float('inf')
        max_x = max_y = float('-inf')
        
        for entity in msp:
            try:
                if entity.dxftype() == 'LINE':
                    min_x = min(min_x, entity.dxf.start.x, entity.dxf.end.x)
                    max_x = max(max_x, entity.dxf.start.x, entity.dxf.end.x)
                    min_y = min(min_y, entity.dxf.start.y, entity.dxf.end.y)
                    max_y = max(max_y, entity.dxf.start.y, entity.dxf.end.y)
                elif entity.dxftype() == 'LWPOLYLINE':
                    points = list(entity.get_points(format='xy'))
                    xs = [p[0] for p in points]
                    ys = [p[1] for p in points]
                    min_x = min(min_x, min(xs))
                    max_x = max(max_x, max(xs))
                    min_y = min(min_y, min(ys))
                    max_y = max(max_y, max(ys))
                elif entity.dxftype() == 'CIRCLE':
                    cx, cy = entity.dxf.center.x, entity.dxf.center.y
                    r = entity.dxf.radius
                    min_x = min(min_x, cx - r)
                    max_x = max(max_x, cx + r)
                    min_y = min(min_y, cy - r)
                    max_y = max(max_y, cy + r)
            except:
                continue
        
        if min_x == float('inf'):
            return None
        
        return (min_x, min_y, max_x, max_y)
    
    def _classify_entity(self, entity, layer: str) -> Tuple[Optional[str], int]:
        """دسته‌بندی المان بر اساس layer و نوع"""
        layer_upper = layer.upper()
        
        # دیوار
        if 'WALL' in layer_upper or 'دیوار' in layer:
            return "wall", 1
        
        # درب
        if 'DOOR' in layer_upper or 'درب' in layer:
            return "door", 2
        
        # پنجره
        if 'WINDOW' in layer_upper or 'پنجره' in layer:
            return "window", 3
        
        # ستون
        if 'COLUMN' in layer_upper or 'ستون' in layer:
            return "column", 4
        
        # تیر
        if 'BEAM' in layer_upper or 'تیر' in layer:
            return "beam", 5
        
        # تاسیسات
        if 'HVAC' in layer_upper or 'تهویه' in layer:
            return "hvac", 7
        
        if 'PLUMB' in layer_upper or 'لوله' in layer:
            return "plumbing", 8
        
        if 'ELEC' in layer_upper or 'برق' in layer:
            return "electrical", 9
        
        # مبلمان
        if 'FURNITURE' in layer_upper or 'مبل' in layer:
            return "furniture", 10
        
        # ابعاد و متن
        if entity.dxftype() in ['TEXT', 'MTEXT']:
            return "text", 13
        
        if entity.dxftype() == 'DIMENSION':
            return "dimension", 12
        
        return None, 0
    
    def _get_entity_bbox(
        self,
        entity,
        min_x: float,
        min_y: float,
        scale: float,
        image_size: Tuple[int, int]
    ) -> Optional[Tuple[float, float, float, float]]:
        """محاسبه bounding box یک المان"""
        try:
            if entity.dxftype() == 'LINE':
                x1 = (entity.dxf.start.x - min_x) * scale
                y1 = (entity.dxf.start.y - min_y) * scale
                x2 = (entity.dxf.end.x - min_x) * scale
                y2 = (entity.dxf.end.y - min_y) * scale
                
                # Flip Y (CAD coordinate system)
                y1 = image_size[1] - y1
                y2 = image_size[1] - y2
                
                return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
            
            elif entity.dxftype() == 'LWPOLYLINE':
                points = list(entity.get_points(format='xy'))
                xs = [(p[0] - min_x) * scale for p in points]
                ys = [image_size[1] - (p[1] - min_y) * scale for p in points]
                
                return (min(xs), min(ys), max(xs), max(ys))
            
            elif entity.dxftype() == 'CIRCLE':
                cx = (entity.dxf.center.x - min_x) * scale
                cy = image_size[1] - (entity.dxf.center.y - min_y) * scale
                r = entity.dxf.radius * scale
                
                return (cx - r, cy - r, cx + r, cy + r)
            
            elif entity.dxftype() in ['TEXT', 'MTEXT']:
                if hasattr(entity.dxf, 'insert'):
                    x = (entity.dxf.insert.x - min_x) * scale
                    y = image_size[1] - (entity.dxf.insert.y - min_y) * scale
                    
                    # تخمین اندازه متن
                    height = getattr(entity.dxf, 'height', 100) * scale
                    width = height * 5  # تخمین
                    
                    return (x, y - height, x + width, y)
        except:
            pass
        
        return None
    
    def _render_dxf_to_image(
        self,
        doc,
        output_path: Path,
        size: Tuple[int, int],
        dpi: int
    ):
        """رندر DXF به تصویر PNG"""
        try:
            from ezdxf.addons.drawing import RenderContext, Frontend
            from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
            import matplotlib.pyplot as plt
            
            # ساخت رندرر
            fig = plt.figure(figsize=(size[0]/dpi, size[1]/dpi), dpi=dpi)
            ax = fig.add_axes([0, 0, 1, 1])
            ctx = RenderContext(doc)
            out = MatplotlibBackend(ax)
            Frontend(ctx, out).draw_layout(doc.modelspace(), finalize=True)
            
            # ذخیره
            fig.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
        except ImportError:
            # اگر matplotlib نباشد، یک تصویر خالی بسازیم
            img = Image.new('RGB', size, 'white')
            img.save(output_path)
    
    def export_coco_format(self, split: str = "train") -> str:
        """
        Export به فرمت COCO JSON
        
        Args:
            split: 'train', 'val', or 'test'
        
        Returns:
            مسیر فایل JSON
        """
        output_file = self.output_dir / f"annotations_{split}.json"
        
        # ساخت ساختار COCO
        coco_data = {
            "info": {
                "description": "CAD Drawing Dataset",
                "version": "1.0",
                "year": 2025,
                "contributor": "CAD3D Neural System"
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        # Categories
        for cat_id, cat_name in self.categories.items():
            coco_data["categories"].append({
                "id": cat_id,
                "name": cat_name,
                "supercategory": "cad_element"
            })
        
        # Images and Annotations
        annotation_id = 1
        for ann in self.annotations:
            # Image info
            coco_data["images"].append({
                "id": ann.image_id,
                "file_name": Path(ann.image_path).name,
                "width": ann.image_width,
                "height": ann.image_height
            })
            
            # Annotations (bboxes)
            for bbox in ann.bboxes:
                width = bbox.x_max - bbox.x_min
                height = bbox.y_max - bbox.y_min
                area = width * height
                
                coco_data["annotations"].append({
                    "id": annotation_id,
                    "image_id": ann.image_id,
                    "category_id": bbox.category_id,
                    "bbox": [bbox.x_min, bbox.y_min, width, height],  # COCO format: [x, y, w, h]
                    "area": area,
                    "iscrowd": 0
                })
                annotation_id += 1
        
        # ذخیره JSON
        with output_file.open('w', encoding='utf-8') as f:
            json.dump(coco_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ COCO format exported: {output_file}")
        print(f"   Images: {len(coco_data['images'])}")
        print(f"   Annotations: {len(coco_data['annotations'])}")
        
        return str(output_file)
    
    def export_yolo_format(self, split: str = "train"):
        """
        Export به فرمت YOLO
        
        فرمت: <class_id> <x_center> <y_center> <width> <height> (normalized)
        """
        labels_dir = self.output_dir / "labels" / split
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        for ann in self.annotations:
            label_file = labels_dir / f"{Path(ann.image_path).stem}.txt"
            
            with label_file.open('w') as f:
                for bbox in ann.bboxes:
                    # تبدیل به فرمت YOLO (normalized)
                    x_center = ((bbox.x_min + bbox.x_max) / 2) / ann.image_width
                    y_center = ((bbox.y_min + bbox.y_max) / 2) / ann.image_height
                    width = (bbox.x_max - bbox.x_min) / ann.image_width
                    height = (bbox.y_max - bbox.y_min) / ann.image_height
                    
                    # کلاس YOLO (0-indexed)
                    class_id = bbox.category_id - 1
                    
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        # ساخت data.yaml
        yaml_file = self.output_dir / "data.yaml"
        with yaml_file.open('w') as f:
            f.write(f"path: {self.output_dir.absolute()}\n")
            f.write(f"train: images/{split}\n")
            f.write(f"val: images/val\n")
            f.write(f"test: images/test\n\n")
            f.write(f"nc: {len(self.categories)}\n")
            f.write(f"names: {list(self.categories.values())}\n")
        
        print(f"✅ YOLO format exported: {labels_dir}")
        print(f"   Config: {yaml_file}")
    
    def visualize_annotations(self, annotation: Annotation, output_path: Optional[Path] = None):
        """رسم bounding boxes روی تصویر برای بررسی"""
        img = Image.open(annotation.image_path).convert('RGB')
        draw = ImageDraw.Draw(img)
        
        # رنگ‌های مختلف برای هر دسته
        colors = {
            1: "red",      # wall
            2: "green",    # door
            3: "blue",     # window
            4: "yellow",   # column
            5: "orange",   # beam
            7: "purple",   # hvac
            8: "cyan",     # plumbing
            9: "magenta",  # electrical
            10: "brown",   # furniture
            13: "pink",    # text
        }
        
        for bbox in annotation.bboxes:
            color = colors.get(bbox.category_id, "white")
            
            # رسم bbox
            draw.rectangle(
                [bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max],
                outline=color,
                width=2
            )
            
            # نوشتن label
            draw.text(
                (bbox.x_min, bbox.y_min - 10),
                bbox.category,
                fill=color
            )
        
        # ذخیره
        if output_path is None:
            output_path = self.output_dir / "visualizations" / f"vis_{annotation.image_id}.png"
            output_path.parent.mkdir(exist_ok=True)
        
        img.save(output_path)
        print(f"   💾 Visualization saved: {output_path.name}")
        
        return output_path


# مثال استفاده
if __name__ == "__main__":
    print("📦 CAD Training Dataset Builder - Demo")
    
    builder = CADDatasetBuilder("training_data")
    print("\n✅ Dataset builder ready")
    print("   Use builder.add_dxf_to_dataset() to add DXF files")
    print("   Use builder.export_coco_format() for COCO format")
    print("   Use builder.export_yolo_format() for YOLO format")
