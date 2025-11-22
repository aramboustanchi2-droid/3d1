"""
Advanced Engineering Drawing Analyzer
تحلیلگر پیشرفته نقشه‌های مهندسی

Comprehensive analysis of 2D engineering drawings (DXF/DWG/PDF/Image) to extract:
- Geometric primitives (lines, polylines, circles, arcs, splines)
- Layers and their semantic meaning
- Dimensions and annotations with OCR
- Text content and title blocks
- Scale detection and calibration
- Structural element classification (walls, columns, beams, slabs, etc.)
- Material properties and metadata
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
from enum import Enum
import json
import re

import numpy as np


class ElementType(Enum):
    """Classified engineering element types"""
    WALL = "wall"
    COLUMN = "column"
    BEAM = "beam"
    SLAB = "slab"
    DOOR = "door"
    WINDOW = "window"
    STAIR = "stair"
    RAILING = "railing"
    FOUNDATION = "foundation"
    BRIDGE_DECK = "bridge_deck"
    BRIDGE_PIER = "bridge_pier"
    BRIDGE_GIRDER = "bridge_girder"
    BRIDGE_CABLE = "bridge_cable"
    MECHANICAL_PART = "mechanical_part"
    PIPE = "pipe"
    DUCT = "duct"
    GENERIC_LINE = "generic_line"
    GENERIC_POLY = "generic_poly"
    CIRCLE = "circle"
    ARC = "arc"
    TEXT = "text"
    DIMENSION = "dimension"


class ConversionStrategy(Enum):
    """Available 3D conversion strategies"""
    EXTRUSION = "extrusion"  # Straight extrusion along Z-axis
    LOFT = "loft"  # Lofting between multiple profiles
    SWEEP = "sweep"  # Sweeping profile along path
    REVOLVE = "revolve"  # Revolving profile around axis
    BOOLEAN = "boolean"  # Boolean union/subtraction/intersection
    PARAMETRIC = "parametric"  # Parametric modeling with constraints
    STRUCTURAL = "structural"  # Structural element generation (BIM-aware)


@dataclass
class GeometryPrimitive:
    """Single geometric primitive with full metadata"""
    type: str  # LINE, LWPOLYLINE, CIRCLE, ARC, SPLINE, TEXT, DIMENSION, etc.
    layer: str
    color: Optional[Tuple[int, int, int]] = None
    vertices: List[Tuple[float, float]] = field(default_factory=list)  # For polylines/lines
    center: Optional[Tuple[float, float]] = None  # For circles/arcs
    radius: Optional[float] = None
    start_angle: Optional[float] = None
    end_angle: Optional[float] = None
    text_content: Optional[str] = None
    text_height: Optional[float] = None
    closed: bool = False
    area: Optional[float] = None
    perimeter: Optional[float] = None
    classified_type: Optional[ElementType] = None
    confidence: float = 0.0


@dataclass
class LayerInfo:
    """Layer metadata and classification"""
    name: str
    color: Optional[Tuple[int, int, int]] = None
    element_count: int = 0
    classified_type: Optional[ElementType] = None
    description: str = ""
    typical_height_mm: Optional[float] = None  # Auto-detected typical height for extrusion


@dataclass
class DimensionAnnotation:
    """Extracted dimension with value and context"""
    value_mm: float
    text: str
    location: Tuple[float, float]
    orientation: str  # horizontal, vertical, aligned
    reference_points: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class ScaleInfo:
    """Drawing scale information"""
    mm_per_unit: float = 1.0  # Detected or specified
    scale_ratio: str = "1:1"  # e.g., "1:50", "1:100"
    confidence: float = 0.0
    detection_method: str = "manual"  # manual, dimension_analysis, reference_object


@dataclass
class EngineeringDrawingAnalysis:
    """Complete analysis result of an engineering drawing"""
    file_path: str
    file_type: str  # DXF, DWG, PDF, IMAGE
    
    # Geometric data
    primitives: List[GeometryPrimitive] = field(default_factory=list)
    layers: Dict[str, LayerInfo] = field(default_factory=dict)
    
    # Annotations and metadata
    dimensions: List[DimensionAnnotation] = field(default_factory=list)
    text_annotations: List[str] = field(default_factory=list)
    title_block: Dict[str, str] = field(default_factory=dict)
    
    # Scale and units
    scale: ScaleInfo = field(default_factory=ScaleInfo)
    drawing_units: str = "mm"
    
    # Bounding box
    bbox_min: Tuple[float, float] = (0.0, 0.0)
    bbox_max: Tuple[float, float] = (0.0, 0.0)
    
    # Classification results
    drawing_type: str = "generic"  # architectural, civil, mechanical, electrical, structural
    detected_elements: Dict[ElementType, int] = field(default_factory=dict)
    
    # Quality metrics
    complexity_score: float = 0.0  # 0-100
    completeness_score: float = 0.0  # 0-100
    
    # Conversion recommendations
    recommended_strategy: ConversionStrategy = ConversionStrategy.EXTRUSION
    recommended_heights: Dict[str, float] = field(default_factory=dict)  # layer -> height_mm
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary"""
        return {
            "file_path": self.file_path,
            "file_type": self.file_type,
            "primitives_count": len(self.primitives),
            "layers": {k: {"name": v.name, "element_count": v.element_count, 
                          "classified_type": v.classified_type.value if v.classified_type else None,
                          "typical_height_mm": v.typical_height_mm}
                      for k, v in self.layers.items()},
            "dimensions_count": len(self.dimensions),
            "scale": {
                "mm_per_unit": self.scale.mm_per_unit,
                "scale_ratio": self.scale.scale_ratio,
                "confidence": self.scale.confidence,
                "detection_method": self.scale.detection_method,
            },
            "bbox": {"min": self.bbox_min, "max": self.bbox_max},
            "drawing_type": self.drawing_type,
            "detected_elements": {k.value: v for k, v in self.detected_elements.items()},
            "complexity_score": self.complexity_score,
            "completeness_score": self.completeness_score,
            "recommended_strategy": self.recommended_strategy.value,
            "recommended_heights": self.recommended_heights,
            "warnings": self.warnings,
        }


class EngineeringDrawingAnalyzer:
    """
    Advanced analyzer for engineering drawings
    Extracts all geometric, semantic, and metadata information
    """
    
    def __init__(self):
        # Layer name patterns for classification
        self.layer_patterns = {
            ElementType.WALL: [r'wall', r'mur', r'دیوار', r'a-wall', r'walls'],
            ElementType.COLUMN: [r'column', r'col', r'pillar', r'ستون', r'a-cols?'],
            ElementType.BEAM: [r'beam', r'joist', r'تیر', r'a-beam'],
            ElementType.SLAB: [r'slab', r'floor', r'deck', r'دال', r'a-flor'],
            ElementType.DOOR: [r'door', r'درب', r'a-door'],
            ElementType.WINDOW: [r'window', r'پنجره', r'a-glaz'],
            ElementType.STAIR: [r'stair', r'پله', r'a-stair'],
            ElementType.BRIDGE_DECK: [r'deck', r'bridge.*deck', r'عرشه'],
            ElementType.BRIDGE_PIER: [r'pier', r'support', r'پایه'],
            ElementType.BRIDGE_GIRDER: [r'girder', r'beam', r'تیر'],
        }
        
        # Typical heights for different element types (mm)
        self.typical_heights = {
            ElementType.WALL: 3000.0,
            ElementType.COLUMN: 3000.0,
            ElementType.BEAM: 500.0,
            ElementType.SLAB: 200.0,
            ElementType.DOOR: 2100.0,
            ElementType.WINDOW: 1500.0,
            ElementType.BRIDGE_DECK: 300.0,
            ElementType.BRIDGE_PIER: 8000.0,
            ElementType.BRIDGE_GIRDER: 1200.0,
        }
    
    def analyze_dxf(self, dxf_path: str) -> EngineeringDrawingAnalysis:
        """Analyze DXF file with full geometric and semantic extraction"""
        import ezdxf
        
        doc = ezdxf.readfile(dxf_path)
        msp = doc.modelspace()
        
        analysis = EngineeringDrawingAnalysis(
            file_path=dxf_path,
            file_type="DXF"
        )
        
        # Extract all geometric primitives
        bbox_points = []
        
        for entity in msp:
            prim = self._extract_primitive(entity)
            if prim:
                analysis.primitives.append(prim)
                
                # Update layer info
                if prim.layer not in analysis.layers:
                    analysis.layers[prim.layer] = LayerInfo(name=prim.layer)
                analysis.layers[prim.layer].element_count += 1
                
                # Collect points for bbox
                if prim.vertices:
                    bbox_points.extend(prim.vertices)
                if prim.center:
                    bbox_points.append(prim.center)
        
        # Compute bounding box
        if bbox_points:
            xs, ys = zip(*bbox_points)
            analysis.bbox_min = (min(xs), min(ys))
            analysis.bbox_max = (max(xs), max(ys))
        
        # Classify layers
        for layer_name, layer_info in analysis.layers.items():
            layer_info.classified_type = self._classify_layer(layer_name)
            if layer_info.classified_type:
                layer_info.typical_height_mm = self.typical_heights.get(layer_info.classified_type)
                if layer_info.classified_type not in analysis.detected_elements:
                    analysis.detected_elements[layer_info.classified_type] = 0
                analysis.detected_elements[layer_info.classified_type] += layer_info.element_count
        
        # Extract dimensions (from DIMENSION entities and TEXT with numeric patterns)
        analysis.dimensions = self._extract_dimensions(msp)
        
        # Detect scale from dimensions
        if analysis.dimensions:
            analysis.scale = self._detect_scale_from_dimensions(analysis.dimensions, bbox_points)
        
        # Classify drawing type
        analysis.drawing_type = self._classify_drawing_type(analysis.detected_elements)
        
        # Compute quality metrics
        analysis.complexity_score = min(100.0, len(analysis.primitives) / 10.0)
        analysis.completeness_score = self._compute_completeness(analysis)
        
        # Recommend conversion strategy and heights
        analysis.recommended_strategy = self._recommend_strategy(analysis)
        analysis.recommended_heights = {
            layer: info.typical_height_mm or 1000.0
            for layer, info in analysis.layers.items()
            if info.element_count > 0
        }
        
        # Generate warnings
        analysis.warnings = self._generate_warnings(analysis)
        
        return analysis
    
    def analyze_image(self, image_path: str, vectorized_dxf_path: str) -> EngineeringDrawingAnalysis:
        """Analyze image-derived DXF with OCR for dimensions and scale"""
        # First analyze the vectorized DXF
        analysis = self.analyze_dxf(vectorized_dxf_path)
        analysis.file_path = image_path
        analysis.file_type = "IMAGE"
        
        # Run OCR on original image for dimension extraction
        try:
            import pytesseract
            from PIL import Image
            
            img = Image.open(image_path)
            ocr_text = pytesseract.image_to_string(img, lang='eng')
            
            # Extract numeric dimensions from OCR text
            dim_pattern = r'(\d+(?:\.\d+)?)\s*(?:mm|cm|m|\'|")'
            matches = re.findall(dim_pattern, ocr_text)
            
            for match in matches:
                val = float(match)
                # Heuristic unit conversion
                if 'cm' in ocr_text:
                    val *= 10
                elif 'm' in ocr_text and val < 100:
                    val *= 1000
                
                analysis.dimensions.append(DimensionAnnotation(
                    value_mm=val,
                    text=match,
                    location=(0.0, 0.0),
                    orientation="unknown"
                ))
            
            # Re-detect scale with OCR dimensions
            if analysis.dimensions:
                bbox_points = []
                for prim in analysis.primitives:
                    if prim.vertices:
                        bbox_points.extend(prim.vertices)
                if bbox_points:
                    analysis.scale = self._detect_scale_from_dimensions(analysis.dimensions, bbox_points)
        
        except Exception as e:
            analysis.warnings.append(f"OCR dimension extraction failed: {e}")
        
        return analysis
    
    def _extract_primitive(self, entity) -> Optional[GeometryPrimitive]:
        """Extract geometric primitive from DXF entity"""
        prim = GeometryPrimitive(
            type=entity.dxftype(),
            layer=entity.dxf.layer if hasattr(entity.dxf, 'layer') else "0"
        )
        
        try:
            if entity.dxftype() == 'LINE':
                prim.vertices = [(entity.dxf.start.x, entity.dxf.start.y),
                                (entity.dxf.end.x, entity.dxf.end.y)]
            
            elif entity.dxftype() == 'LWPOLYLINE':
                prim.vertices = [(p[0], p[1]) for p in entity.get_points('xy')]
                prim.closed = entity.closed
                if prim.closed and len(prim.vertices) >= 3:
                    # Compute area and perimeter
                    verts = np.array(prim.vertices)
                    prim.perimeter = float(np.sum(np.linalg.norm(np.diff(verts, axis=0, append=[verts[0]]), axis=1)))
                    # Shoelace formula
                    x, y = verts[:, 0], verts[:, 1]
                    prim.area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
            
            elif entity.dxftype() == 'CIRCLE':
                prim.center = (entity.dxf.center.x, entity.dxf.center.y)
                prim.radius = entity.dxf.radius
                prim.area = np.pi * prim.radius ** 2
            
            elif entity.dxftype() == 'ARC':
                prim.center = (entity.dxf.center.x, entity.dxf.center.y)
                prim.radius = entity.dxf.radius
                prim.start_angle = entity.dxf.start_angle
                prim.end_angle = entity.dxf.end_angle
            
            elif entity.dxftype() in ('TEXT', 'MTEXT'):
                prim.text_content = entity.dxf.text if hasattr(entity.dxf, 'text') else ""
                prim.text_height = entity.dxf.height if hasattr(entity.dxf, 'height') else 0.0
                prim.vertices = [(entity.dxf.insert.x, entity.dxf.insert.y)]
            
            return prim
        
        except Exception:
            return None
    
    def _classify_layer(self, layer_name: str) -> Optional[ElementType]:
        """Classify layer based on name patterns"""
        layer_lower = layer_name.lower()
        
        for elem_type, patterns in self.layer_patterns.items():
            for pattern in patterns:
                if re.search(pattern, layer_lower, re.IGNORECASE):
                    return elem_type
        
        return None
    
    def _extract_dimensions(self, msp) -> List[DimensionAnnotation]:
        """Extract dimension annotations from DIMENSION entities and numeric TEXT"""
        dimensions = []
        
        for entity in msp:
            if entity.dxftype() == 'DIMENSION':
                try:
                    dim_text = entity.dxf.text if hasattr(entity.dxf, 'text') else ""
                    # Parse numeric value
                    match = re.search(r'(\d+(?:\.\d+)?)', dim_text)
                    if match:
                        val = float(match.group(1))
                        dimensions.append(DimensionAnnotation(
                            value_mm=val,
                            text=dim_text,
                            location=(0, 0),
                            orientation="unknown"
                        ))
                except Exception:
                    pass
            
            elif entity.dxftype() in ('TEXT', 'MTEXT'):
                text = entity.dxf.text if hasattr(entity.dxf, 'text') else ""
                # Look for dimension-like text (number followed by unit)
                match = re.search(r'(\d+(?:\.\d+)?)\s*(?:mm|cm|m)', text, re.IGNORECASE)
                if match:
                    val = float(match.group(1))
                    if 'cm' in text.lower():
                        val *= 10
                    elif 'm' in text.lower() and val < 100:
                        val *= 1000
                    
                    dimensions.append(DimensionAnnotation(
                        value_mm=val,
                        text=text,
                        location=(entity.dxf.insert.x, entity.dxf.insert.y),
                        orientation="unknown"
                    ))
        
        return dimensions
    
    def _detect_scale_from_dimensions(self, dimensions: List[DimensionAnnotation], 
                                     bbox_points: List[Tuple[float, float]]) -> ScaleInfo:
        """Auto-detect drawing scale by comparing dimension annotations to geometry"""
        if not dimensions or not bbox_points:
            return ScaleInfo()
        
        # Heuristic: find largest dimension and compare to drawing extent
        largest_dim = max(dimensions, key=lambda d: d.value_mm)
        
        xs, ys = zip(*bbox_points)
        drawing_width = max(xs) - min(xs)
        drawing_height = max(ys) - min(ys)
        max_extent = max(drawing_width, drawing_height)
        
        if max_extent > 1.0:
            # Estimate mm/unit
            estimated_mm_per_unit = largest_dim.value_mm / max_extent
            
            # Round to common scale ratios
            common_scales = [1, 2, 5, 10, 20, 25, 50, 100, 200, 500, 1000]
            closest_scale = min(common_scales, key=lambda s: abs(s - estimated_mm_per_unit))
            
            return ScaleInfo(
                mm_per_unit=float(closest_scale),
                scale_ratio=f"1:{int(1000.0/closest_scale)}" if closest_scale < 1000 else "1:1",
                confidence=0.7,
                detection_method="dimension_analysis"
            )
        
        return ScaleInfo()
    
    def _classify_drawing_type(self, detected_elements: Dict[ElementType, int]) -> str:
        """Classify overall drawing type based on detected elements"""
        if not detected_elements:
            return "generic"
        
        # Count element categories
        arch_count = sum(v for k, v in detected_elements.items() 
                        if k in (ElementType.WALL, ElementType.DOOR, ElementType.WINDOW, ElementType.STAIR))
        bridge_count = sum(v for k, v in detected_elements.items()
                          if k in (ElementType.BRIDGE_DECK, ElementType.BRIDGE_PIER, ElementType.BRIDGE_GIRDER))
        struct_count = sum(v for k, v in detected_elements.items()
                          if k in (ElementType.COLUMN, ElementType.BEAM, ElementType.SLAB))
        
        if bridge_count > arch_count and bridge_count > struct_count:
            return "civil_bridge"
        elif arch_count > 0:
            return "architectural"
        elif struct_count > 0:
            return "structural"
        else:
            return "generic"
    
    def _compute_completeness(self, analysis: EngineeringDrawingAnalysis) -> float:
        """Compute completeness score (0-100) based on metadata richness"""
        score = 0.0
        
        # Has layers: +20
        if len(analysis.layers) > 1:
            score += 20.0
        
        # Has dimensions: +30
        if len(analysis.dimensions) > 0:
            score += 30.0
        
        # Has classified elements: +30
        if len(analysis.detected_elements) > 0:
            score += 30.0
        
        # Has scale info: +20
        if analysis.scale.confidence > 0.5:
            score += 20.0
        
        return min(100.0, score)
    
    def _recommend_strategy(self, analysis: EngineeringDrawingAnalysis) -> ConversionStrategy:
        """Recommend best 3D conversion strategy based on drawing analysis"""
        # Default to extrusion for most cases
        if analysis.drawing_type == "civil_bridge":
            return ConversionStrategy.STRUCTURAL
        elif analysis.drawing_type in ("architectural", "structural"):
            return ConversionStrategy.STRUCTURAL
        else:
            return ConversionStrategy.EXTRUSION
    
    def _generate_warnings(self, analysis: EngineeringDrawingAnalysis) -> List[str]:
        """Generate warnings based on analysis"""
        warnings = []
        
        if analysis.scale.confidence < 0.5:
            warnings.append("⚠️ Scale detection confidence is low. Manual scale verification recommended.")
        
        if len(analysis.layers) == 1:
            warnings.append("⚠️ Single layer detected. Layer-based height assignment not possible.")
        
        if len(analysis.dimensions) == 0:
            warnings.append("⚠️ No dimension annotations found. Scale calibration may be inaccurate.")
        
        if analysis.complexity_score > 80:
            warnings.append("⚠️ High complexity drawing. Conversion may take longer.")
        
        return warnings
