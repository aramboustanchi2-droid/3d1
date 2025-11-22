from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import logging
import os

# Import system capabilities
# Note: In a real scenario, we would import the actual modules.
# For now, we will simulate the calls or wrap them if possible.

logger = logging.getLogger(__name__)

class Agent(ABC):
    """
    Base class for a worker agent in the pipeline.
    """
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role

    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the agent's task.
        :param context: The shared context/state of the pipeline.
        :return: Updated context.
        """
        pass

class InputParserAgent(Agent):
    """
    Responsible for validating and parsing input files (DXF, DWG, Images).
    """
    def __init__(self):
        super().__init__("InputParser", "Parser")

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        input_path = context.get("input_path")
        if not input_path:
            raise ValueError("No input_path provided to InputParserAgent")
        
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
            
        ext = os.path.splitext(input_path)[1].lower()
        context["file_type"] = ext
        
        logger.info(f"[{self.name}] Parsed input: {input_path} (Type: {ext})")
        return context

class ArchitecturalAnalyzerAgent(Agent):
    """
    Analyzes the architectural intent of the drawing.
    Wraps the logic from architectural_analyzer.py
    """
    def __init__(self):
        super().__init__("ArchAnalyzer", "Architect")

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"[{self.name}] Analyzing architectural elements...")
        # Simulation of calling architectural_analyzer
        # In a full integration, we would instantiate ArchitecturalAnalyzer here
        
        # Mocking analysis results for the pipeline flow
        analysis_results = {
            "detected_spaces": ["Living Room", "Kitchen"],
            "walls_count": 12,
            "scale_factor": 1.0
        }
        context["analysis"] = analysis_results
        return context

class EngineeringValidatorAgent(Agent):
    """
    Checks for geometric validity and structural basic rules.
    """
    def __init__(self):
        super().__init__("StructEngineer", "Engineer")

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"[{self.name}] Validating geometry...")
        # Logic to check for "hard shapes" or self-intersections
        
        issues = []
        # Mock check
        if context.get("analysis", {}).get("walls_count", 0) == 0:
            issues.append("No walls detected")
            
        context["validation_issues"] = issues
        context["is_valid"] = len(issues) == 0
        return context

class ModelingConfigurationAgent(Agent):
    """
    Configures the 3D extrusion parameters.
    """
    def __init__(self):
        super().__init__("3DModeler", "Modeler")

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"[{self.name}] Configuring 3D parameters...")
        
        # Determine height based on analysis or defaults
        default_height = 3000.0
        
        config = {
            "extrusion_height": default_height,
            "layer_strategy": "split_by_color",
            "optimize_mesh": True
        }
        context["modeling_config"] = config
        return context

class OutputGeneratorAgent(Agent):
    """
    Generates the final 3D file.
    """
    def __init__(self):
        super().__init__("OutputGen", "Builder")

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"[{self.name}] Generating output...")
        
        input_path = context.get("input_path")
        if input_path:
            base, _ = os.path.splitext(input_path)
            output_path = f"{base}_3d.dxf"
            context["output_path"] = output_path
            logger.info(f"[{self.name}] Output generated at: {output_path}")
        
        return context

class FeasibilityAgent(Agent):
    """
    Inspired by Hektar.ai.
    Performs generative design and feasibility studies for site massing.
    """
    def __init__(self):
        super().__init__("HektarFeasibility", "Planner")

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"[{self.name}] Conducting feasibility study (Hektar-style)...")
        
        # Check for AI Insights (Learned Knowledge)
        ai_insights = context.get("ai_insights", {})
        
        # Simulate Hektar's workflow:
        # 1. Analyze Site & Geometry
        site_area = context.get("site_area", 1000.0)
        dims = context.get("dimensions", [])
        height = context.get("proposed_height", 30.0)
        shape = context.get("massing_shape", "rect")
        floor_height = 3.2  # avg floor height (m)
        floors = max(1, int(height / floor_height))
        if len(dims) == 2:
            footprint_area = dims[0] * dims[1]
        elif len(dims) == 1:
            footprint_area = dims[0] ** 2
        else:
            footprint_area = min(site_area, 40.0 * 40.0)  # fallback square

        # Base geometry metrics
        volume = footprint_area * height
        slenderness = height / ((dims[0] + dims[1]) / 2) if len(dims) == 2 and (dims[0] + dims[1]) else height / (dims[0] if dims else 20.0)
        gfa_estimate = footprint_area * floors

        # 2. Generate Massing Options (data-driven)
        if ai_insights:
            logger.info(f"[{self.name}] Using Deep Learning insights for generation.")
            options = [
                {
                    "type": ai_insights.get("typology", shape.title()),
                    "gfa": ai_insights.get("recommended_gfa", gfa_estimate * 1.05),
                    "floors": floors,
                    "strategy": ai_insights.get("strategy", "AI-Optimized"),
                    "source": "AI_Model"
                }
            ]
        else:
            # Heuristic alternatives based on shape
            density_factor = 0.8 if shape == "circle" else (1.1 if shape == "tower" else 1.0)
            options = [
                {"type": "Baseline", "gfa": gfa_estimate, "floors": floors, "source": "Heuristic"},
                {"type": "Optimized Core", "gfa": gfa_estimate * 1.15 * density_factor, "floors": floors + 1, "source": "Heuristic"},
                {"type": "Terraced", "gfa": gfa_estimate * 0.95, "floors": floors, "source": "Heuristic"}
            ]

        # 3. Select Best Option by GFA
        best_option = max(options, key=lambda x: x["gfa"])

        # 4. Derived performance metrics
        efficiency = min(0.92, 0.75 + (0.02 * floors))  # simplistic efficiency growth with floors
        daylight_score = "High" if slenderness > 3.5 else ("Medium" if slenderness > 2.0 else "Low")
        structural_risk = "Low" if slenderness < 6 else ("Moderate" if slenderness < 8 else "High")

        context["feasibility_report"] = {
            "site_area": round(site_area, 2),
            "footprint_area": round(footprint_area, 2),
            "floors": floors,
            "height": round(height, 2),
            "shape": shape,
            "volume_m3": round(volume, 2),
            "slenderness_ratio": round(slenderness, 2),
            "estimated_gfa": round(gfa_estimate, 2),
            "generated_options": len(options),
            "recommended_massing": best_option,
            "metrics": {
                "efficiency_ratio": f"{efficiency*100:.1f}%",
                "daylight_score": daylight_score,
                "structural_risk": structural_risk
            },
            "ai_enhanced": bool(ai_insights)
        }

        # Expose massing metrics for downstream UI consumption
        context["massing_metrics"] = {
            "footprint_area_m2": footprint_area,
            "height_m": height,
            "floors": floors,
            "volume_m3": volume,
            "slenderness_ratio": slenderness,
            "estimated_gfa_m2": gfa_estimate
        }

        logger.info(f"[{self.name}] Feasibility complete. Recommended: {best_option['type']} | GFA={best_option['gfa']:.2f}")
        return context
