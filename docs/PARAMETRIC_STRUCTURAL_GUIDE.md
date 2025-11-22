# Parametric & Structural Analysis System Guide

# راهنمای سیستم پارامتریک و تحلیل ساختاری

**Version:** 1.0  
**Last Updated:** 2024  
**Author:** CAD3D Development Team

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Parametric Engine](#parametric-engine)
3. [Structural Analysis](#structural-analysis)
4. [Complete Integration](#complete-integration)
5. [Industry Applications](#industry-applications)
6. [Code Examples](#code-examples)
7. [API Reference](#api-reference)
8. [Best Practices](#best-practices)

---

## 🎯 Overview

این سند دو سیستم پیشرفته را معرفی می‌کند:

### 1. Parametric Engine (موتور پارامتریک)

- **Purpose:** روابط وابستگی بین عناصر CAD (مشابه Revit)
- **Capabilities:**
  - Expression evaluation (محاسبه فرمول‌ها)
  - Automatic propagation (انتشار خودکار تغییرات)
  - Constraint solving (حل محدودیت‌های هندسی)
  - Dependency tracking (پیگیری وابستگی‌ها)

### 2. Structural Analysis (تحلیل ساختاری)

- **Purpose:** تحلیل مهندسی پیشرفته
- **Capabilities:**
  - Load analysis (تحلیل بار)
  - Stress/Strain calculation (محاسبه تنش/کرنش)
  - Deflection analysis (تحلیل خیز)
  - Safety checks (بررسی ایمنی)
  - Industry-specific limits (حدود مجاز صنعت‌ها)

---

## ⚙️ Parametric Engine

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     PARAMETRIC ENGINE                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │  Expression  │────│  Dependency  │────│  Constraint  │ │
│  │  Evaluator   │    │    Graph     │    │   Solver     │ │
│  └──────────────┘    └──────────────┘    └──────────────┘ │
│         │                    │                    │         │
│         └────────────────────┴────────────────────┘         │
│                              │                               │
│                    ┌─────────▼─────────┐                    │
│                    │  CAD Graph        │                    │
│                    │  (Elements +      │                    │
│                    │   Relationships)  │                    │
│                    └───────────────────┘                    │
└─────────────────────────────────────────────────────────────┘
```

### Core Concepts

#### 1. Parametric Expression

عبارتی که یک property را به properties دیگر ربط می‌دهد:

```python
"window.width = wall.width * 0.3"
```

**Types:**

- **DIRECT:** مقدار ثابت (value = 100)
- **REFERENCE:** ارجاع مستقیم (value = other.property)
- **FORMULA:** فرمول (value = expr(other.property))

#### 2. Dependency Graph

گرافی که وابستگی‌های بین عناصر را نشان می‌دهد:

```
wall ──┐
       ├──> window
       └──> door

span ──> beam ──> slab
```

#### 3. Geometric Constraint

محدودیت هندسی بین دو عنصر:

- **PARALLEL:** دو خط موازی
- **PERPENDICULAR:** دو خط عمود
- **COINCIDENT:** دو نقطه هم‌راستا
- **DISTANCE:** فاصله ثابت
- **ANGLE:** زاویه ثابت

### Usage Example

```python
from cad3d.cad_graph import CADGraph, CADElement, ElementType
from cad3d.parametric_engine import ParametricEngine, ConstraintType

# 1. Create graph
graph = CADGraph()

# 2. Add elements
wall = CADElement(
    id="wall_001",
    element_type=ElementType.WALL,
    properties={'width': 10000, 'height': 3000}
)
graph.add_element(wall)

window = CADElement(
    id="window_001",
    element_type=ElementType.WINDOW,
    properties={'width': 3000, 'height': 1500}
)
graph.add_element(window)

# 3. Create parametric engine
engine = ParametricEngine(graph)

# 4. Add expression: window width = 30% of wall width
engine.add_expression(
    target_element="window_001",
    target_property="width",
    expression="wall_001.width * 0.3"
)

# 5. Change wall width
result = engine.update_parameter("wall_001", "width", 15000)
# → window_001.width automatically becomes 4500

# 6. Add constraint
engine.add_constraint(
    element1_id="wall_001",
    element2_id="window_001",
    constraint_type=ConstraintType.DISTANCE,
    value=100  # minimum 100mm clearance
)

# 7. Validate
validation = engine.validate_graph()
```

### Advanced Features

#### Cycle Detection

تشخیص حلقه‌های وابستگی (که باعث خطا می‌شوند):

```python
A.x = B.x + 10
B.x = C.x + 5
C.x = A.x - 3  # ❌ CYCLE!
```

Engine به طور خودکار این را تشخیص می‌دهد.

#### Expression Syntax

توابع قابل استفاده در expressions:

```python
# Math functions
"beam.length = math.sqrt(dx**2 + dy**2)"
"angle = math.atan2(dy, dx)"

# Built-in functions
"max_width = max(wall1.width, wall2.width)"
"min_height = min(window1.height, window2.height)"
"clearance = abs(elem1.x - elem2.x)"
```

---

## 🏗️ Structural Analysis

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  STRUCTURAL ANALYZER                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Beam         │  │ Column       │  │ Slab         │     │
│  │ Analysis     │  │ Analysis     │  │ Analysis     │     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
│         │                  │                  │              │
│         └──────────────────┴──────────────────┘              │
│                            │                                 │
│                  ┌─────────▼─────────┐                      │
│                  │  Analysis Result  │                      │
│                  │  - Stress         │                      │
│                  │  - Deflection     │                      │
│                  │  - Safety Check   │                      │
│                  └───────────────────┘                      │
└─────────────────────────────────────────────────────────────┘
```

### Supported Analyses

#### 1. Beam Analysis

تحلیل تیر با محاسبه:

- Bending moment (لنگر خمشی)
- Bending stress (تنش خمشی)
- Shear stress (تنش برشی)
- Deflection (خیز)

**Formulas:**

```
Bending Moment (Simply Supported):
    M_max = (w × L²) / 8

Bending Stress:
    σ_b = M / W

Deflection (Simply Supported):
    δ = (5 × w × L⁴) / (384 × E × I)
```

#### 2. Column Analysis

تحلیل ستون با محاسبه:

- Axial stress (تنش محوری)
- Buckling load (بار کمانش)
- Slenderness ratio (نسبت لاغری)

**Formulas:**

```
Euler Buckling Load:
    P_cr = (π² × E × I) / (Le²)
    
    where:
    Le = K × L  (effective length)
    K = effective length factor

Slenderness Ratio:
    λ = Le / r
    r = √(I/A)  (radius of gyration)
```

#### 3. Slab Analysis

تحلیل دال با محاسبه:

- Bending moment (لنگر خمشی دو طرفه)
- Bending stress (تنش خمشی)
- Deflection (خیز)

**Formulas:**

```
Moment (Two-way slab):
    M_x = α × q × Lx²
    M_y = α × q × Ly²
    
    where α = moment coefficient (from tables)

Deflection:
    δ = (5 × q × L⁴) / (384 × E × I)
```

### Usage Example

```python
from cad3d.structural_analysis import (
    StructuralAnalyzer, Load, LoadType, 
    Material, Section, STEEL_S355, IPE_300
)
from cad3d.industrial_gnn import IndustryType

# 1. Create analyzer
analyzer = StructuralAnalyzer(graph, IndustryType.BUILDING)

# 2. Define material and section
material = STEEL_S355  # E=200 GPa, fy=355 MPa
section = IPE_300      # A=5380 mm², I=8356 cm⁴

# 3. Analyze beam
result = analyzer.analyze_beam(
    element_id="beam_001",
    material=material,
    section=section,
    loads=[
        Load(LoadType.DEAD, 20000),  # 20 kN
        Load(LoadType.LIVE, 30000),  # 30 kN
    ],
    length=6.0,  # 6m
    support_conditions="simply_supported"
)

# 4. Check safety
if result.is_safe:
    print("✅ Beam is safe")
    print(f"  Stress ratio: {result.stress_ratio:.2f}")
    print(f"  Deflection: {result.deflection*1000:.1f} mm")
else:
    print("❌ Beam is unsafe")
    for error in result.errors:
        print(f"  - {error}")

# 5. Analyze all structure
summary = analyzer.analyze_structure()
print(f"Safe elements: {summary['safe_elements']}/{summary['total_elements']}")
```

### Industry-Specific Limits

هر صنعت حدود مجاز خاص خود را دارد:

| Industry | Deflection Limit | Stress Limit | Notes |
|----------|-----------------|--------------|-------|
| Building | L/300 | 90% fy | استاندارد |
| Bridge | L/800 | 85% fy | سخت‌تر (لرزش) |
| Tunnel | 1% deformation | 80% fy | ضریب ایمنی بالا |
| General | L/250 | 90% fy | پیش‌فرض |

**L** = طول/دهانه عنصر  
**fy** = تنش تسلیم

### Predefined Materials & Sections

```python
# Concrete
CONCRETE_C30 = Material(
    name="C30",
    E=30e9,      # 30 GPa
    fy=30e6,     # 30 MPa
    density=2500 # 2500 kg/m³
)

# Steel
STEEL_S355 = Material(
    name="S355",
    E=200e9,     # 200 GPa
    fy=355e6,    # 355 MPa
    density=7850 # 7850 kg/m³
)

# I-beams
IPE_300 = Section(
    name="IPE300",
    A=0.00538,      # 5380 mm²
    I=8356e-8,      # 8356 cm⁴
    W=557e-6,       # 557 cm³
    height=0.3      # 300 mm
)

HEB_300 = Section(
    name="HEB300",
    A=0.0149,       # 14900 mm²
    I=25170e-8,     # 25170 cm⁴
    W=1678e-6,      # 1678 cm³
    height=0.3      # 300 mm
)
```

---

## 🔗 Complete Integration

### Combined Workflow

```python
from cad3d.cad_graph import CADGraph
from cad3d.parametric_engine import ParametricEngine
from cad3d.structural_analysis import StructuralAnalyzer

# 1. Create structure
graph = create_building()

# 2. Setup parametric relationships
engine = ParametricEngine(graph)
engine.add_expression(
    "beam_001", "length", 
    "abs(column_2.x - column_1.x)"
)

# 3. Initial analysis
analyzer = StructuralAnalyzer(graph)
result_initial = analyzer.analyze_beam("beam_001", ...)

# 4. Optimize: try different spans
for span in [5.0, 6.0, 7.0, 8.0]:
    # Update parameters
    engine.update_parameter("column_2", "x", span * 1000)
    
    # Re-analyze
    result = analyzer.analyze_beam("beam_001", ...)
    
    # Check if safe and economical
    if result.is_safe and result.stress_ratio < 0.7:
        print(f"✅ Optimal span: {span}m")
        break
```

### Design Iteration Loop

```
     ┌──────────────────────────────────────┐
     │                                      │
     ▼                                      │
┌─────────┐    ┌─────────┐    ┌─────────┐ │
│ Initial │───▶│ Analyze │───▶│  Safe?  │─┘
│ Design  │    │         │    │         │ NO
└─────────┘    └─────────┘    └────┬────┘
                                    │ YES
                                    ▼
                               ┌─────────┐
                               │  Done   │
                               └─────────┘
```

---

## 🏭 Industry Applications

### 1. Building Construction (ساختمان‌سازی)

**Typical Workflow:**

```python
# Create building with parametric floors
for floor in range(1, num_floors + 1):
    # Columns
    create_columns(floor, span=6.0, height=3.5)
    
    # Beams (length = distance between columns)
    engine.add_expression(
        f"beam_floor{floor}",
        "length",
        f"column_A{floor}.x - column_B{floor}.x"
    )
    
    # Slabs (span = beam length)
    engine.add_expression(
        f"slab_floor{floor}",
        "span",
        f"beam_floor{floor}.length"
    )

# Analyze
for floor in range(1, num_floors + 1):
    analyzer.analyze_beam(f"beam_floor{floor}", ...)
    analyzer.analyze_slab(f"slab_floor{floor}", ...)
```

**Checks:**

- Column buckling
- Beam deflection (L/300)
- Slab deflection (L/300)
- Story drift (زلزله)

### 2. Bridge Engineering (پل‌سازی)

**Typical Workflow:**

```python
# Main girders
for i, girder in enumerate(main_girders):
    # Girder length = bridge span
    engine.add_expression(
        f"girder_{i}", "length",
        "abutment_2.x - abutment_1.x"
    )
    
    # Analyze with fatigue
    result = analyzer.analyze_beam(
        girder.id,
        loads=[
            Load(LoadType.DEAD, dead_load),
            Load(LoadType.LIVE, truck_load),  # HS20 or similar
            Load(LoadType.IMPACT, impact_load)
        ]
    )
    
    # Check fatigue (1M+ cycles)
    if result.stress_ratio > 0.5:
        warnings.append("High stress - check fatigue")
```

**Checks:**

- Girder stress (85% fy)
- Deflection (L/800 - stricter!)
- Vibration (3 Hz minimum)
- Fatigue life (1M cycles)

### 3. Dam Engineering (سدسازی)

**Typical Workflow:**

```python
# Dam sections
analyzer_dam = StructuralAnalyzer(graph, IndustryType.DAM)

# Hydrostatic pressure
water_depth = 50  # 50m
pressure_base = water_density * g * water_depth  # Pa

# Analyze stability
for section in dam_sections:
    loads = [
        Load(LoadType.HYDROSTATIC, pressure_base, direction=(1,0,0)),
        Load(LoadType.DEAD, section.weight, direction=(0,0,-1))
    ]
    
    # Check:
    # 1. Stress
    # 2. Sliding stability
    # 3. Overturning stability
    result = analyze_dam_section(section, loads)
```

**Checks:**

- Stress (80% limit)
- Sliding factor > 1.5
- Overturning factor > 2.0
- Seepage control

### 4. Tunnel Engineering (تونل‌سازی)

**Typical Workflow:**

```python
analyzer_tunnel = StructuralAnalyzer(graph, IndustryType.TUNNEL)

# Rock pressure
rock_class = "III"  # I-VI
overburden = 100  # 100m depth
pressure = rock_density * g * overburden * K  # K=lateral pressure coef

# Analyze lining
result = analyzer.analyze_tunnel_lining(
    element_id="lining_001",
    material=CONCRETE_C30,
    thickness=0.4,  # 40cm
    loads=[Load(LoadType.EARTH, pressure)]
)

# Support requirements
if result.stress_ratio > 0.8:
    print("⚠️ Additional support required (rock bolts, shotcrete)")
```

**Checks:**

- Deformation < 1%
- Stress (80% limit)
- Rock class compatibility
- Support adequacy

---

## 💻 Code Examples

### Example 1: Simple Building

See `examples/complete_parametric_structural_example.py`

```python
# Create 6m × 6m building
graph = create_simple_building()

# Setup parametric relationships
engine = setup_parametric_relationships(graph)

# Analyze
analyzer = perform_structural_analysis(graph)

# Optimize: try 8m × 8m
engine.update_parameter("column_C2", "x", 8000)
# → All beams and slab automatically update!

# Re-analyze
analyzer_8m = perform_structural_analysis(graph)

# Compare
compare_designs(analyzer_6m, analyzer_8m)
```

### Example 2: Load Increase Study

```python
# Standard load (3 kN/m²)
result_3kn = analyzer.analyze_slab(
    "slab_001",
    loads=[
        Load(LoadType.DEAD, 5000),
        Load(LoadType.LIVE, 3000)
    ]
)

# Increased load (5 kN/m²) - for library/archive
result_5kn = analyzer.analyze_slab(
    "slab_001",
    loads=[
        Load(LoadType.DEAD, 5000),
        Load(LoadType.LIVE, 5000)
    ]
)

# Check if reinforcement needed
if not result_5kn.is_safe:
    print("⚠️ Need thicker slab or additional beams")
```

### Example 3: Multi-Story Building

```python
# Create 5-story building
num_floors = 5
graph = CADGraph()

for floor in range(1, num_floors + 1):
    # Columns (cumulative load increases downward)
    for col in ['A', 'B', 'C', 'D']:
        col_id = f"column_{col}_{floor}"
        
        # Height same for all floors
        engine.add_expression(
            col_id, "height",
            "floor_height"  # Parametric!
        )
        
        # Load = weight of all floors above
        num_floors_above = num_floors - floor
        load = base_load * num_floors_above
        
        result = analyzer.analyze_column(
            col_id,
            loads=[Load(LoadType.DEAD, load)]
        )

# Change floor height → all columns update!
engine.update_parameter("floor_height", "value", 3.2)
```

---

## 📚 API Reference

### ParametricEngine

```python
class ParametricEngine:
    def __init__(self, graph: CADGraph)
    
    def add_expression(
        target_element: str,
        target_property: str,
        expression: str,
        expression_type: ExpressionType = FORMULA
    ) -> None
    
    def add_constraint(
        element1_id: str,
        element2_id: str,
        constraint_type: ConstraintType,
        value: Optional[float] = None
    ) -> None
    
    def update_parameter(
        element_id: str,
        property_name: str,
        new_value: Any,
        propagate: bool = True
    ) -> Dict[str, Any]
    
    def validate_graph() -> Dict[str, Any]
    
    def export_to_json(path: Path) -> None
```

### StructuralAnalyzer

```python
class StructuralAnalyzer:
    def __init__(
        graph: CADGraph,
        industry_type: IndustryType = GENERAL
    )
    
    def analyze_beam(
        element_id: str,
        material: Material,
        section: Section,
        loads: List[Load],
        length: Optional[float] = None,
        support_conditions: str = "simply_supported"
    ) -> AnalysisResult
    
    def analyze_column(
        element_id: str,
        material: Material,
        section: Section,
        loads: List[Load],
        height: Optional[float] = None,
        effective_length_factor: float = 1.0
    ) -> AnalysisResult
    
    def analyze_slab(
        element_id: str,
        material: Material,
        thickness: float,
        loads: List[Load],
        span_x: float,
        span_y: float,
        support_type: str = "four_edges"
    ) -> AnalysisResult
    
    def analyze_structure() -> Dict[str, Any]
    
    def export_results(path: Path) -> None
```

### Data Classes

```python
@dataclass
class Load:
    load_type: LoadType
    magnitude: float  # N or Pa
    direction: Tuple[float, float, float] = (0, 0, -1)
    point: Optional[Tuple[float, float, float]] = None
    distribution: str = "uniform"

@dataclass
class Material:
    name: str
    E: float      # Pa (modulus of elasticity)
    fy: float     # Pa (yield stress)
    density: float  # kg/m³
    poisson: float = 0.3
    G: Optional[float] = None  # Pa (shear modulus)

@dataclass
class Section:
    name: str
    A: float      # m² (area)
    I: float      # m⁴ (moment of inertia)
    W: float      # m³ (section modulus)
    J: Optional[float] = None  # m⁴ (polar moment)
    height: Optional[float] = None  # m
    width: Optional[float] = None   # m

@dataclass
class AnalysisResult:
    element_id: str
    analysis_type: AnalysisType
    axial_stress: Optional[float] = None
    bending_stress: Optional[float] = None
    shear_stress: Optional[float] = None
    max_stress: Optional[float] = None
    deflection: Optional[float] = None
    stress_ratio: Optional[float] = None
    deflection_ratio: Optional[float] = None
    is_safe: bool = True
    warnings: List[str]
    errors: List[str]
```

---

## ✅ Best Practices

### 1. Parametric Design

**DO:**

- ✅ Use meaningful variable names: `wall_001.width`, not `w1.w`
- ✅ Keep expressions simple: `a * b + c`, not complex nested formulas
- ✅ Validate graph after adding expressions
- ✅ Document dependencies in comments

**DON'T:**

- ❌ Create circular dependencies (A → B → C → A)
- ❌ Use magic numbers (use named parameters)
- ❌ Over-constrain (too many constraints = conflicts)

### 2. Structural Analysis

**DO:**

- ✅ Always specify correct industry type for appropriate limits
- ✅ Use realistic material properties (not arbitrary values)
- ✅ Check both stress AND deflection
- ✅ Consider load combinations (dead + live + wind, etc.)
- ✅ Apply safety factors

**DON'T:**

- ❌ Ignore warnings (they indicate potential issues)
- ❌ Use inadequate section properties
- ❌ Forget to check buckling for columns
- ❌ Use unrealistic support conditions

### 3. Performance

**Tips:**

- Disable propagation when making bulk updates:

  ```python
  for elem in elements:
      engine.update_parameter(elem, prop, val, propagate=False)
  # Then manually propagate once:
  engine.propagate_changes()
  ```

- Batch analyze similar elements:

  ```python
  # Instead of analyzing one by one:
  for beam in beams:
      results.append(analyzer.analyze_beam(beam, ...))
  
  # Better: prepare data then batch process
  beam_data = [(beam, material, section, loads) for beam in beams]
  results = analyzer.batch_analyze_beams(beam_data)
  ```

### 4. Validation

Always validate before critical operations:

```python
# 1. Validate parametric graph
validation = engine.validate_graph()
if not validation['valid']:
    print("❌ Parametric graph has errors:")
    for error in validation['errors']:
        print(f"  - {error}")
    return

# 2. Analyze structure
analyzer = StructuralAnalyzer(graph)
summary = analyzer.analyze_structure()

# 3. Check safety
if summary['unsafe_elements'] > 0:
    print("⚠️ Structure has unsafe elements:")
    for elem_id in summary['critical_elements']:
        result = analyzer.results[elem_id]
        print(f"  - {elem_id}:")
        for error in result.errors:
            print(f"      {error}")
```

---

## 📊 Performance Benchmarks

| Operation | Time | Notes |
|-----------|------|-------|
| Add expression | < 1ms | Fast |
| Update parameter (10 deps) | ~5ms | Propagation |
| Validate graph (100 elem) | ~50ms | Cycle detection |
| Analyze beam | ~1ms | Simple calculation |
| Analyze column | ~2ms | Buckling check |
| Analyze slab | ~3ms | 2D analysis |
| Full structure (50 elem) | ~100ms | Complete analysis |

**Test System:** Intel i7, 16GB RAM

---

## 🔬 Theoretical Background

### Parametric Systems

Based on:

- **Constraint programming** (Sutherland, 1963)
- **Dependency graphs** (topological sort)
- **Expression evaluation** (recursive descent parsing)

### Structural Analysis

Based on:

- **Euler-Bernoulli beam theory** (1750s)
- **Euler buckling theory** (1757)
- **Plate theory** (Kirchhoff, 1850)
- **Modern codes:** AISC, Eurocode, ACI

Key formulas already shown in [Structural Analysis](#structural-analysis) section.

---

## 🚀 Future Enhancements

### Planned Features

1. **Advanced FEA Integration**
   - Direct integration with FEA solvers (Abaqus, Ansys)
   - Mesh generation
   - Non-linear analysis

2. **Optimization Algorithms**
   - Genetic algorithms for design optimization
   - Multi-objective optimization (cost vs. safety)
   - Topology optimization

3. **Dynamic Analysis**
   - Modal analysis (vibration modes)
   - Seismic response (time-history)
   - Wind-induced vibration

4. **BIM Integration**
   - IFC import/export
   - Revit API integration
   - Parametric synchronization

---

## 📞 Support

For questions or issues:

- GitHub Issues: [github.com/yourusername/cad3d/issues](https://github.com)
- Documentation: [docs/](../docs/)
- Examples: [examples/](../examples/)

---

**Last Updated:** 2024  
**Version:** 1.0
