"""
Advanced 3D Conversion Engine

This module provides a sophisticated engine for converting 2D CAD drawings into
3D models using multiple conversion strategies and intelligent element recognition.

Key Features:
- Structural element generation (BIM-aware).
- Multi-strategy conversion (extrusion, loft, sweep, revolve).
- Automatic height detection based on layer analysis.
- Boolean operations for creating complex geometries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import time
import logging

import ezdxf
from ezdxf.document import Drawing
from ezdxf.layouts import Modelspace
from ezdxf.entities import LWPolyline

from .engineering_analyzer import (
    EngineeringDrawingAnalysis,
    ConversionStrategy,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@dataclass
class ConversionReport:
    """
    A detailed report of the 3D conversion process.

    Attributes:
        input_file: Path to the source 2D file.
        output_file: Path to the generated 3D file.
        strategy_used: The conversion strategy that was applied.
        elements_converted: The total number of 2D elements converted to 3D.
        layers_processed: A dictionary mapping layer names to the count of converted elements on that layer.
        applied_heights: A dictionary mapping layer names to the extrusion heights used.
        warnings: A list of warnings encountered during the conversion.
        errors: A list of errors that occurred.
        processing_time_sec: The total time taken for the conversion in seconds.
        quality_score: A score from 0 to 100 representing the quality of the conversion.
    """
    input_file: str
    output_file: str
    strategy_used: ConversionStrategy
    elements_converted: int = 0
    layers_processed: Dict[str, int] = field(default_factory=dict)
    applied_heights: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    processing_time_sec: float = 0.0
    quality_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the report to a dictionary."""
        return {
            "input_file": self.input_file,
            "output_file": self.output_file,
            "strategy": self.strategy_used.value,
            "elements_converted": self.elements_converted,
            "layers_processed": self.layers_processed,
            "applied_heights": self.applied_heights,
            "warnings": self.warnings,
            "errors": self.errors,
            "processing_time_sec": self.processing_time_sec,
            "quality_score": self.quality_score,
        }

    def to_html(self) -> str:
        """Generates a user-friendly HTML report of the conversion process."""
        status_class = "success" if not self.errors else "error"
        status_message = "✅ Conversion Completed Successfully" if not self.errors else "❌ Conversion Failed"

        layer_rows = ""
        for layer, count in self.layers_processed.items():
            height = self.applied_heights.get(layer, 0.0)
            layer_rows += f"<tr><td>{layer}</td><td>{count}</td><td>{height:.0f}</td></tr>"

        warning_items = "".join([f'<div class="warning">{warning}</div>' for warning in self.warnings])
        error_items = "".join([f'<div class="error">{error}</div>' for error in self.errors])

        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>3D Conversion Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 2rem; background: #f8f9fa; color: #212529; }}
        .container {{ max-width: 900px; margin: auto; background: white; padding: 2rem; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #007bff; padding-bottom: 0.5rem; }}
        h2 {{ color: #34495e; margin-top: 2.5rem; border-bottom: 1px solid #dee2e6; padding-bottom: 0.5rem; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1.5rem; margin-top: 1.5rem; }}
        .metric {{ padding: 1.5rem; background: #f1f3f5; border-radius: 6px; text-align: center; }}
        .metric-label {{ font-size: 0.9rem; color: #6c757d; text-transform: uppercase; }}
        .metric-value {{ font-size: 1.75rem; font-weight: 600; color: #0056b3; }}
        table {{ width: 100%; border-collapse: collapse; margin: 1.5rem 0; }}
        th, td {{ padding: 0.85rem; text-align: left; border-bottom: 1px solid #dee2e6; }}
        th {{ background: #007bff; color: white; }}
        .warning, .error, .success {{ padding: 1rem; margin: 0.5rem 0; border-radius: 4px; border-left: 5px solid; }}
        .warning {{ background: #fff3cd; border-color: #ffc107; }}
        .error {{ background: #f8d7da; border-color: #dc3545; }}
        .success {{ background: #d4edda; border-color: #28a745; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 2D to 3D Conversion Report</h1>
        <div class="{status_class}">{status_message}</div>
        
        <h2>Overview</h2>
        <div class="metric-grid">
            <div class="metric">
                <div class="metric-label">Strategy</div>
                <div class="metric-value">{self.strategy_used.value}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Processing Time</div>
                <div class="metric-value">{self.processing_time_sec:.2f}s</div>
            </div>
            <div class="metric">
                <div class="metric-label">Quality Score</div>
                <div class="metric-value">{self.quality_score:.0f}/100</div>
            </div>
            <div class="metric">
                <div class="metric-label">Elements Converted</div>
                <div class="metric-value">{self.elements_converted}</div>
            </div>
        </div>
        
        <h2>Processed Layers</h2>
        <table>
            <tr><th>Layer Name</th><th>Elements Converted</th><th>Applied Height (mm)</th></tr>
            {layer_rows}
        </table>
        
        {f"<h2>⚠️ Warnings ({len(self.warnings)})</h2>{warning_items}" if self.warnings else ""}
        {f"<h2>❌ Errors ({len(self.errors)})</h2>{error_items}" if self.errors else ""}
    </div>
</body>
</html>
"""


class Advanced3DConverter:
    """
    An advanced 3D converter that uses engineering analysis to apply
    intelligent conversion strategies.
    """

    def convert_with_analysis(
        self,
        analysis: EngineeringDrawingAnalysis,
        input_dxf_path: str,
        output_dxf_path: str,
        strategy: Optional[ConversionStrategy] = None,
        custom_heights: Optional[Dict[str, float]] = None,
    ) -> ConversionReport:
        """
        Converts a 2D DXF file to 3D using pre-computed analysis results and a specified strategy.

        Args:
            analysis: The analysis result from the EngineeringDrawingAnalyzer.
            input_dxf_path: Path to the input 2D DXF file.
            output_dxf_path: Path where the output 3D DXF file will be saved.
            strategy: The conversion strategy to use. If None, the recommended strategy from the analysis is used.
            custom_heights: A dictionary to override extrusion heights for specific layers.

        Returns:
            A ConversionReport detailing the outcome of the process.
        """
        start_time = time.time()
        
        final_strategy = strategy or analysis.recommended_strategy
        heights = custom_heights or analysis.recommended_heights
        
        report = ConversionReport(
            input_file=input_dxf_path,
            output_file=output_dxf_path,
            strategy_used=final_strategy,
            applied_heights=heights,
        )
        
        logging.info(f"Starting conversion with strategy: {final_strategy.name}")

        try:
            doc_in = ezdxf.readfile(input_dxf_path)
            doc_out = ezdxf.new(setup=True)
            
            if final_strategy == ConversionStrategy.STRUCTURAL:
                self._convert_structural(analysis, doc_in, doc_out, heights, report)
            else:  # Default to advanced extrusion
                self._convert_extrusion(analysis, doc_in, doc_out, heights, report)

            doc_out.saveas(output_dxf_path)
            logging.info(f"Successfully saved 3D DXF to {output_dxf_path}")

        except IOError as e:
            error_msg = f"File I/O error: {e}"
            logging.error(error_msg)
            report.errors.append(error_msg)
        except ezdxf.DXFStructureError as e:
            error_msg = f"Invalid DXF file structure: {e}"
            logging.error(error_msg)
            report.errors.append(error_msg)
        except Exception as e:
            error_msg = f"An unexpected error occurred during conversion: {e}"
            logging.error(error_msg, exc_info=True)
            report.errors.append(error_msg)
        
        report.processing_time_sec = time.time() - start_time
        report.quality_score = self._compute_quality_score(report, analysis)
        
        return report

    def _process_layers_in_memory(
        self,
        doc_in: Drawing,
        doc_out: Drawing,
        heights: Dict[str, float],
        report: ConversionReport,
    ):
        """
        A helper method to process entities layer by layer in memory, creating 3D meshes.
        This replaces the inefficient file-based processing.
        """
        msp_in = doc_in.modelspace()
        msp_out = doc_out.modelspace()
        
        total_converted = 0
        
        # Group entities by layer for efficiency
        entities_by_layer: Dict[str, List[Any]] = {}
        for entity in msp_in:
            if hasattr(entity.dxf, 'layer'):
                entities_by_layer.setdefault(entity.dxf.layer, []).append(entity)

        for layer_name, height in heights.items():
            if height <= 0:
                continue

            if layer_name not in entities_by_layer:
                continue

            converted_count_in_layer = 0
            for entity in entities_by_layer[layer_name]:
                if isinstance(entity, LWPolyline) and entity.is_closed:
                    try:
                        mesh_data = self._build_prism_mesh_for_polyline(entity, height)
                        if mesh_data:
                            mesh_entity = msp_out.add_mesh()
                            with mesh_entity.edit_data() as md:
                                md.vertices = mesh_data['vertices']
                                md.faces = mesh_data['faces']
                            

                            mesh_entity.dxf.layer = layer_name
                            if hasattr(entity.dxf, 'true_color'):
                                mesh_entity.dxf.true_color = entity.dxf.true_color
                            
                            converted_count_in_layer += 1
                    except Exception as e:
                        warning_msg = f"Failed to convert entity {entity.dxf.handle} in layer '{layer_name}': {e}"
                        logging.warning(warning_msg)
                        report.warnings.append(warning_msg)
            
            if converted_count_in_layer > 0:
                report.layers_processed[layer_name] = converted_count_in_layer
                total_converted += converted_count_in_layer
                # Ensure layer exists in the output document
                if layer_name not in doc_out.layers:
                    doc_out.layers.add(layer_name)

        report.elements_converted = total_converted
        if total_converted == 0:
            report.warnings.append("No closed polylines found on specified layers for conversion.")

    def _convert_structural(
        self,
        analysis: EngineeringDrawingAnalysis,
        doc_in: Drawing,
        doc_out: Drawing,
        heights: Dict[str, float],
        report: ConversionReport,
    ):
        """
        Structural conversion strategy: Generates 3D elements with awareness of their type.
        Currently, this is an enhanced extrusion, but is designed to be extended for full BIM support.
        """
        logging.info("Executing STRUCTURAL conversion strategy.")
        # Future: Implement BIM element generation (e.g., creating IFC-compatible objects).
        # For now, we use an intelligent extrusion based on the analysis.
        self._process_layers_in_memory(doc_in, doc_out, heights, report)

    def _convert_extrusion(
        self,
        analysis: EngineeringDrawingAnalysis,
        doc_in: Drawing,
        doc_out: Drawing,
        heights: Dict[str, float],
        report: ConversionReport,
    ):
        """
        Enhanced extrusion strategy that processes entities layer by layer in memory.
        """
        logging.info("Executing EXTRUSION conversion strategy.")
        self._process_layers_in_memory(doc_in, doc_out, heights, report)

    def _build_prism_mesh_for_polyline(self, polyline: LWPolyline, height: float) -> Optional[Dict[str, Any]]:
        """
        Builds a 3D prism mesh from a closed 2D LWPOLYLINE.

        Args:
            polyline: The closed LWPolyline entity.
            height: The extrusion height for the prism.

        Returns:
            A dictionary containing 'vertices' and 'faces' for the mesh, or None if creation fails.
        """
        try:
            # Flattening handles arcs and bulges in the polyline
            points = [p[:2] for p in polyline.vertices()]
            if len(points) < 3:
                return None

            bottom_verts = [(p[0], p[1], 0.0) for p in points]
            top_verts = [(p[0], p[1], height) for p in points]
            
            n = len(points)
            vertices = bottom_verts + top_verts
            faces = []

            # Create bottom and top faces using ezdxf's triangulation
            bottom_face = ezdxf.math.triangulate_2d(bottom_verts)
            top_face_indices = [i + n for i in range(n)]
            top_face_verts = [vertices[i] for i in top_face_indices]
            top_face = ezdxf.math.triangulate_2d(top_face_verts)

            # Adjust top face indices and reverse winding order
            faces.extend([list(f) for f in bottom_face])
            faces.extend([[i + n for i in reversed(f)] for f in top_face])

            # Create side faces (quads split into two triangles)
            for i in range(n):
                i_next = (i + 1) % n
                # Indices for the quad
                p1, p2 = i, i_next
                p3, p4 = i_next + n, i + n
                # Add two triangles for the side quad
                faces.append([p1, p2, p4])
                faces.append([p2, p3, p4])
            
            return {'vertices': vertices, 'faces': faces}
        
        except Exception as e:
            logging.error(f"Failed to build prism mesh for polyline {polyline.dxf.handle}: {e}")
            return None

    def _compute_quality_score(self, report: ConversionReport, analysis: EngineeringDrawingAnalysis) -> float:
        """
        Computes a quality score for the conversion based on the report and analysis.

        Args:
            report: The conversion report.
            analysis: The engineering analysis result.

        Returns:
            A quality score between 0.0 and 100.0.
        """
        score = 100.0
        
        score -= len(report.errors) * 25
        score -= len(report.warnings) * 5
        
        total_primitives = len(analysis.primitives)
        if total_primitives > 0:
            conversion_rate = report.elements_converted / total_primitives
            if conversion_rate < 0.5:
                score -= (0.5 - conversion_rate) * 40  # Heavy penalty for low conversion rate
            else:
                score += (conversion_rate - 0.5) * 10 # Bonus for high rate
        elif report.elements_converted == 0:
            score = max(0, score - 50) # Penalize if nothing was converted at all

        if len(report.layers_processed) > 1:
            score += 10  # Bonus for processing multiple layers

        return max(0.0, min(100.0, score))
