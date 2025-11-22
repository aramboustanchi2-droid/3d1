from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import math
from pathlib import Path
import json

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from .cad_graph import CADGraph, CADElement, ElementType
from .industrial_gnn import IndustryType


class LoadType(Enum):
    """Defines the type of load applied to a structural element."""
    DEAD = "dead"              # Self-weight of the structure
    LIVE = "live"              # Load from usage (e.g., people, furniture)
    WIND = "wind"              # Wind load
    SNOW = "snow"              # Snow load
    SEISMIC = "seismic"        # Earthquake load
    HYDROSTATIC = "hydrostatic"  # Water pressure
    EARTH = "earth"            # Soil pressure
    THERMAL = "thermal"        # Temperature-induced load
    IMPACT = "impact"          # Sudden, short-duration load


class StressType(Enum):
    """Defines the type of stress within an element."""
    AXIAL = "axial"            # Tensile or compressive stress along the axis
    BENDING = "bending"        # Stress from bending moments
    SHEAR = "shear"            # Stress from shear forces
    TORSION = "torsion"        # Stress from twisting moments
    COMBINED = "combined"      # Combination of multiple stress types


class AnalysisType(Enum):
    """Defines the type of structural analysis to be performed."""
    STATIC = "static"          # Analysis under static (non-moving) loads
    DYNAMIC = "dynamic"        # Analysis under time-varying loads
    MODAL = "modal"            # Analysis to find natural frequencies
    BUCKLING = "buckling"      # Analysis to determine critical buckling loads
    FATIGUE = "fatigue"        # Analysis of material failure under cyclic loading
    SEISMIC = "seismic"        # Specific dynamic analysis for earthquakes


@dataclass
class Load:
    """
    Represents a load applied to a structural element.

    Attributes:
        load_type: The category of the load.
        magnitude: The force (in Newtons) or pressure (in Pascals).
        direction: The unit vector of the load's direction.
        point: The point of application for concentrated loads.
        distribution: The pattern of the load (e.g., uniform, concentrated).
    """
    load_type: LoadType
    magnitude: float  # N for concentrated, Pa (N/m^2) for distributed
    direction: Tuple[float, float, float] = (0, 0, -1)
    point: Optional[Tuple[float, float, float]] = None
    distribution: str = "uniform"

    def __post_init__(self):
        """Normalizes the direction vector after initialization."""
        if NUMPY_AVAILABLE:
            d = np.array(self.direction)
            norm = np.linalg.norm(d)
            if norm > 1e-9:
                self.direction = tuple(d / norm)


@dataclass
class Material:
    """
    Represents the physical properties of a construction material.

    Attributes:
        name: Common name of the material (e.g., "C30 Concrete", "S355 Steel").
        E: Young's Modulus (modulus of elasticity) in Pascals.
        fy: Yield strength in Pascals.
        density: Density in kg/m³.
        poisson: Poisson's ratio (dimensionless).
        G: Shear modulus in Pascals.
    """
    name: str
    E: float         # Young's Modulus (Pa)
    fy: float        # Yield Strength (Pa)
    density: float   # Density (kg/m³)
    poisson: float = 0.3
    G: Optional[float] = None

    def __post_init__(self):
        """Calculates the shear modulus if not provided."""
        if self.G is None:
            self.G = self.E / (2 * (1 + self.poisson))


@dataclass
class Section:
    """
    Represents the geometric properties of a structural cross-section.

    Attributes:
        name: Standard name of the section (e.g., "IPE300").
        A: Cross-sectional area in m².
        I: Moment of inertia in m⁴.
        W: Section modulus in m³.
        J: Polar moment of inertia in m⁴ (for torsion).
        height: The overall height of the section in meters.
        width: The overall width of the section in meters.
    """
    name: str
    A: float   # Area (m²)
    I: float   # Moment of inertia (m⁴)
    W: float   # Section modulus (m³)
    J: Optional[float] = None
    height: Optional[float] = None
    width: Optional[float] = None


@dataclass
class AnalysisResult:
    """
    Stores the outcome of a structural analysis for a single element.

    Attributes:
        is_safe: A boolean indicating if the element passes all safety checks.
        stress_ratio: The ratio of actual stress to allowable stress.
        deflection_ratio: The ratio of actual deflection to allowable deflection.
        warnings: A list of non-critical issues.
        errors: A list of critical failures.
    """
    element_id: str
    analysis_type: AnalysisType
    is_safe: bool = True
    axial_stress: Optional[float] = None
    bending_stress: Optional[float] = None
    shear_stress: Optional[float] = None
    max_stress: Optional[float] = None
    deflection: Optional[float] = None
    max_deflection: Optional[float] = None
    stress_ratio: Optional[float] = None
    deflection_ratio: Optional[float] = None
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    additional_data: Dict[str, Any] = field(default_factory=dict)


class StructuralAnalyzer:
    """
    Performs structural analysis on elements within a CADGraph.

    This class provides simplified engineering calculations for common structural
    elements like beams, columns, and slabs. It checks for safety based on
    stress, deflection, and buckling against industry-standard limits.
    """
    def __init__(self, graph: CADGraph, industry_type: IndustryType = IndustryType.GENERAL):
        self.graph = graph
        self.industry_type = industry_type
        self.results: Dict[str, AnalysisResult] = {}
        self.materials: Dict[str, Material] = {}
        self.sections: Dict[str, Section] = {}
        self.safety_factors = {
            'stress': 1.5,
            'deflection': 1.0,
            'buckling': 1.8,
        }
        self.limits = self._get_industry_limits()

    def _get_industry_limits(self) -> Dict[str, Any]:
        """Returns a dictionary of allowable limits based on the industry type."""
        limits = {
            'deflection_limit_ratio': 1/250,
            'stress_limit_factor': 0.9,
        }
        if self.industry_type == IndustryType.BUILDING:
            limits.update({'deflection_limit_ratio': 1/300, 'vibration_limit_hz': 5.0})
        elif self.industry_type == IndustryType.BRIDGE:
            limits.update({
                'deflection_limit_ratio': 1/800,
                'stress_limit_factor': 0.85,
                'vibration_limit_hz': 3.0,
                'fatigue_limit_cycles': 1e6,
            })
        elif self.industry_type == IndustryType.TUNNEL:
            limits.update({
                'deformation_limit_factor': 0.01,
                'stress_limit_factor': 0.8,
            })
        return limits

    def _check_safety(self, result: AnalysisResult, material: Material, length: float):
        """
        Performs safety checks for stress and deflection and updates the result object.
        """
        # Stress Check
        if result.max_stress is not None:
            allowable_stress = material.fy / self.safety_factors['stress']
            result.stress_ratio = result.max_stress / allowable_stress
            if result.stress_ratio > 1.0:
                result.is_safe = False
                result.errors.append(f"Stress ratio {result.stress_ratio:.2f} > 1.0 (exceeds material yield limit)")
            elif result.stress_ratio > self.limits['stress_limit_factor']:
                result.warnings.append(f"Stress ratio {result.stress_ratio:.2f} is high")

        # Deflection Check
        if result.deflection is not None:
            allowable_deflection = length * self.limits['deflection_limit_ratio']
            result.max_deflection = allowable_deflection
            result.deflection_ratio = result.deflection / allowable_deflection
            if result.deflection_ratio > 1.0:
                result.is_safe = False
                result.errors.append(f"Deflection ratio {result.deflection_ratio:.2f} > 1.0 (exceeds span/deflection limit)")
            elif result.deflection_ratio > 0.85:
                result.warnings.append(f"Deflection ratio {result.deflection_ratio:.2f} is high")

    def analyze_beam(self, element_id: str, material: Material, section: Section, loads: List[Load], length: float, support_conditions: str = "simply_supported") -> AnalysisResult:
        """Analyzes a beam element under given loads."""
        print(f"\nAnalyzing Beam: {element_id}...")
        element = self.graph.get_element(element_id)
        result = AnalysisResult(element_id, AnalysisType.STATIC)

        if not element:
            result.is_safe = False
            result.errors.append(f"Element '{element_id}' not found in graph.")
            return result

        total_load = sum(load.magnitude for load in loads)
        
        # Simplified moment and shear calculations
        if support_conditions == "simply_supported":
            M_max = total_load * length / 4
            V_max = total_load / 2
        elif support_conditions == "cantilever":
            M_max = total_load * length
            V_max = total_load
        else: # Default to a generic case
            M_max = total_load * length / 8
            V_max = total_load / 2

        result.bending_stress = M_max / section.W
        result.shear_stress = V_max / section.A
        result.max_stress = result.bending_stress # Simplified: bending usually dominates

        # Simplified deflection calculation
        w = total_load / length # Equivalent uniform load
        if support_conditions == "simply_supported":
            result.deflection = (5 * w * length**4) / (384 * material.E * section.I)
        elif support_conditions == "cantilever":
            result.deflection = (w * length**4) / (8 * material.E * section.I)
        else:
            result.deflection = (total_load * length**3) / (48 * material.E * section.I)

        self._check_safety(result, material, length)
        self.results[element_id] = result
        
        print(f"  - Analysis complete. Safe: {result.is_safe}. Stress Ratio: {result.stress_ratio:.2f}, Deflection Ratio: {result.deflection_ratio:.2f}")
        return result

    def analyze_column(self, element_id: str, material: Material, section: Section, loads: List[Load], height: float, effective_length_factor: float = 1.0) -> AnalysisResult:
        """Analyzes a column element for axial load and buckling."""
        print(f"\nAnalyzing Column: {element_id}...")
        element = self.graph.get_element(element_id)
        result = AnalysisResult(element_id, AnalysisType.BUCKLING)

        if not element:
            result.is_safe = False
            result.errors.append(f"Element '{element_id}' not found in graph.")
            return result

        axial_load = sum(load.magnitude for load in loads if load.load_type in [LoadType.DEAD, LoadType.LIVE])
        result.axial_stress = axial_load / section.A
        result.max_stress = result.axial_stress

        # Buckling check (Euler's formula)
        Le = effective_length_factor * height
        r = math.sqrt(section.I / section.A)  # Radius of gyration
        slenderness_ratio = Le / r
        result.additional_data['slenderness_ratio'] = slenderness_ratio

        if slenderness_ratio > 200:
            result.warnings.append(f"Slenderness ratio {slenderness_ratio:.1f} > 200 is high, may be unstable.")

        P_cr = (math.pi**2 * material.E * section.I) / (Le**2) # Critical buckling load
        result.additional_data['buckling_load_N'] = P_cr
        
        buckling_load_ratio = axial_load / (P_cr / self.safety_factors['buckling'])
        result.additional_data['buckling_load_ratio'] = buckling_load_ratio

        if buckling_load_ratio > 1.0:
            result.is_safe = False
            result.errors.append(f"Buckling load ratio {buckling_load_ratio:.2f} > 1.0 (high risk of buckling failure)")

        self._check_safety(result, material, height)
        self.results[element_id] = result
        
        print(f"  - Analysis complete. Safe: {result.is_safe}. Buckling Ratio: {buckling_load_ratio:.2f}, Stress Ratio: {result.stress_ratio:.2f}")
        return result

    def analyze_structure(self) -> Dict[str, Any]:
        """Performs analysis on all relevant elements and returns a summary."""
        print("\n" + "="*70)
        print("Starting Full Structure Analysis...")
        print("="*70)
        
        summary = {
            'total_elements': len(self.graph.elements),
            'analyzed_count': 0,
            'safe_count': 0,
            'unsafe_count': 0,
            'critical_elements': [],
            'max_stress_ratio': 0.0,
            'max_deflection_ratio': 0.0,
        }

        # This is a placeholder loop. A real implementation would intelligently
        # select elements and their properties.
        for elem_id, element in self.graph.elements.items():
            if element.element_type == ElementType.BEAM:
                # In a real scenario, material, section, loads, etc. would be retrieved
                # from the element's properties or a central database.
                self.analyze_beam(
                    elem_id, STEEL_S355, IPE_300, 
                    [Load(LoadType.LIVE, 50000)], 
                    element.properties.get('length', 6.0)
                )
            elif element.element_type == ElementType.COLUMN:
                self.analyze_column(
                    elem_id, STEEL_S355, HEB_300,
                    [Load(LoadType.DEAD, 500000)],
                    element.properties.get('height', 3.5)
                )

        for elem_id, result in self.results.items():
            summary['analyzed_count'] += 1
            if result.is_safe:
                summary['safe_count'] += 1
            else:
                summary['unsafe_count'] += 1
                summary['critical_elements'].append(elem_id)
            
            summary['max_stress_ratio'] = max(summary['max_stress_ratio'], result.stress_ratio or 0.0)
            summary['max_deflection_ratio'] = max(summary['max_deflection_ratio'], result.deflection_ratio or 0.0)

        print("\n" + "="*70)
        print("Analysis Summary")
        print("="*70)
        print(f"  - Analyzed Elements: {summary['analyzed_count']} / {summary['total_elements']}")
        print(f"  - Safe Elements: {summary['safe_count']} ✅")
        print(f"  - Unsafe Elements: {summary['unsafe_count']} ❌")
        print(f"  - Max Stress Ratio: {summary['max_stress_ratio']:.2f}")
        print(f"  - Max Deflection Ratio: {summary['max_deflection_ratio']:.2f}")
        if summary['critical_elements']:
            print(f"  - Critical Elements: {', '.join(summary['critical_elements'])}")
        print("="*70)
        
        return summary

    def export_results_to_json(self, path: Path):
        """Saves the analysis results to a JSON file."""
        data = {
            'industry_type': self.industry_type.value,
            'safety_factors': self.safety_factors,
            'limits': self.limits,
            'results': [result.__dict__ for result in self.results.values()]
        }
        with path.open('w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"💾 Analysis results saved to {path}")


# Predefined Materials and Sections for convenience
CONCRETE_C30 = Material(name="C30 Concrete", E=30e9, fy=30e6, density=2500)
STEEL_S355 = Material(name="S355 Steel", E=200e9, fy=355e6, density=7850)
STEEL_S235 = Material(name="S235 Steel", E=210e9, fy=235e6, density=7850)

IPE_300 = Section(name="IPE300", A=0.00538, I=8356e-8, W=557e-6, height=0.3, width=0.15)
IPE_400 = Section(name="IPE400", A=0.00845, I=23130e-8, W=1156e-6, height=0.4, width=0.18)
HEB_300 = Section(name="HEB300", A=0.01491, I=18260e-8, W=1210e-6, height=0.3, width=0.3)


def demo_structural_analysis():
    """Example demonstrating the use of the StructuralAnalyzer."""
    print("="*70)
    print("Structural Analysis Demo")
    print("="*70)
    
    graph = CADGraph("Building Project")
    graph.add_element(CADElement(id="beam_001", element_type=ElementType.BEAM, properties={'length': 6.0}))
    graph.add_element(CADElement(id="column_001", element_type=ElementType.COLUMN, properties={'height': 3.5}))
    graph.add_element(CADElement(id="column_002", element_type=ElementType.COLUMN, properties={'height': 3.5})) # Unsafe column

    analyzer = StructuralAnalyzer(graph, IndustryType.BUILDING)

    # Analyze a safe beam
    analyzer.analyze_beam(
        "beam_001", STEEL_S355, IPE_400,
        [Load(LoadType.DEAD, 20000), Load(LoadType.LIVE, 30000)],
        length=8.0
    )
    
    # Analyze a safe column
    analyzer.analyze_column(
        "column_001", STEEL_S355, HEB_300,
        [Load(LoadType.DEAD, 800000)], # 800 kN
        height=3.5
    )

    # Analyze an unsafe column (high load)
    analyzer.analyze_column(
        "column_002", STEEL_S235, IPE_300, # Weaker material and section
        [Load(LoadType.DEAD, 950000)], # 950 kN
        height=4.0,
        effective_length_factor=1.2
    )

    analyzer.analyze_structure()
    analyzer.export_results_to_json(Path("structural_analysis_demo.json"))
    
    print("\n✅ Demo complete!")


if __name__ == "__main__":
    demo_structural_analysis()
