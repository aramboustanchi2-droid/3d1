"""
Complete Example: Parametric + Structural Analysis
مثال کامل: سیستم پارامتریک + تحلیل ساختاری

این مثال نشان می‌دهد چگونه:
1. یک ساختمان ساده با روابط پارامتریک بسازیم
2. تحلیل ساختاری انجام دهیم
3. با تغییر پارامترها، به‌روزرسانی خودکار شود
4. تحلیل مجدد انجام شود

سناریو:
    ساختمان یک طبقه با:
    - 4 ستون
    - 4 تیر
    - 1 دال
    
    روابط پارامتریک:
    - beam.length = distance(column1, column2)
    - slab.span = beam.length
"""

from pathlib import Path
import math

from cad3d.cad_graph import CADGraph, CADElement, ElementType, RelationType, CADRelationship
from cad3d.parametric_engine import ParametricEngine, ExpressionType, ConstraintType
from cad3d.structural_analysis import (
    StructuralAnalyzer, Load, LoadType, Material, Section,
    CONCRETE_C30, STEEL_S355, IPE_300, HEB_300
)
from cad3d.industrial_gnn import IndustryType


def create_simple_building() -> CADGraph:
    """
    ساخت یک ساختمان ساده
    
    طرح:
        C1 -------- C2
        |           |
        |    (6m×6m)|
        |           |
        C3 -------- C4
    
    - 4 ستون در گوشه‌ها
    - 4 تیر برای اتصال ستون‌ها
    - 1 دال روی تیرها
    """
    graph = CADGraph()
    
    # ابعاد
    span = 6000  # 6 متر (mm)
    height = 3500  # 3.5 متر
    
    print("\n" + "="*70)
    print("Creating Simple Building Structure")
    print("="*70)
    print(f"  Span: {span/1000}m × {span/1000}m")
    print(f"  Height: {height/1000}m")
    
    # 1. ستون‌ها
    columns = {
        'C1': (0, 0, 0),
        'C2': (span, 0, 0),
        'C3': (0, span, 0),
        'C4': (span, span, 0)
    }
    
    print(f"\n  Creating {len(columns)} columns...")
    for col_id, (x, y, z) in columns.items():
        column = CADElement(
            id=f"column_{col_id}",
            element_type=ElementType.COLUMN,
            centroid=(x, y, height/2),
            properties={
                'height': height,
                'section': 'HEB300',
                'material': 'S355',
                'x': x,
                'y': y,
                'z': z
            }
        )
        graph.add_element(column)
    
    # 2. تیرها
    beams = [
        ('B1', 'C1', 'C2'),  # جنوبی
        ('B2', 'C3', 'C4'),  # شمالی
        ('B3', 'C1', 'C3'),  # غربی
        ('B4', 'C2', 'C4'),  # شرقی
    ]
    
    print(f"  Creating {len(beams)} beams...")
    for beam_id, col1, col2 in beams:
        c1 = graph.get_element(f"column_{col1}")
        c2 = graph.get_element(f"column_{col2}")
        
        # محاسبه طول
        dx = c2.centroid[0] - c1.centroid[0]
        dy = c2.centroid[1] - c1.centroid[1]
        length = math.sqrt(dx**2 + dy**2)
        
        # مرکز تیر
        cx = (c1.centroid[0] + c2.centroid[0]) / 2
        cy = (c1.centroid[1] + c2.centroid[1]) / 2
        
        beam = CADElement(
            id=f"beam_{beam_id}",
            element_type=ElementType.BEAM,
            centroid=(cx, cy, height),
            properties={
                'length': length,
                'section': 'IPE300',
                'material': 'S355',
                'start_column': f"column_{col1}",
                'end_column': f"column_{col2}"
            }
        )
        graph.add_element(beam)
        
        # روابط
        graph.add_relationship(CADRelationship(f"column_{col1}", beam.id, RelationType.SUPPORTED_BY))
        graph.add_relationship(CADRelationship(f"column_{col2}", beam.id, RelationType.SUPPORTED_BY))
    
    # 3. دال
    print(f"  Creating slab...")
    slab = CADElement(
        id="slab_001",
        element_type=ElementType.SLAB,
        centroid=(span/2, span/2, height),
        properties={
            'span_x': span,
            'span_y': span,
            'thickness': 200,  # 20 cm
            'material': 'C30'
        }
    )
    graph.add_element(slab)
    
    # روابط دال با تیرها
    for beam_id, _, _ in beams:
        graph.add_relationship(CADRelationship(f"beam_{beam_id}", slab.id, RelationType.SUPPORTED_BY))
    
    print(f"\n✅ Building created!")
    print(f"  Total elements: {len(graph.elements)}")
    print(f"  Total relationships: {len(graph.relationships)}")
    
    return graph


def setup_parametric_relationships(graph: CADGraph) -> ParametricEngine:
    """
    تعریف روابط پارامتریک
    
    روابط:
    - طول تیرها = فاصله بین ستون‌ها
    - دهانه دال = طول تیرها
    """
    print("\n" + "="*70)
    print("Setting up Parametric Relationships")
    print("="*70)
    
    engine = ParametricEngine(graph)
    
    # 1. رابطه تیرها با ستون‌ها
    print("\n  Beam-Column relationships:")
    
    # تیر B1 (بین C1 و C2)
    engine.add_expression(
        target_element="beam_B1",
        target_property="length",
        expression="abs(column_C2.x - column_C1.x)"
    )
    
    # تیر B2 (بین C3 و C4)
    engine.add_expression(
        target_element="beam_B2",
        target_property="length",
        expression="abs(column_C4.x - column_C3.x)"
    )
    
    # تیر B3 (بین C1 و C3)
    engine.add_expression(
        target_element="beam_B3",
        target_property="length",
        expression="abs(column_C3.y - column_C1.y)"
    )
    
    # تیر B4 (بین C2 و C4)
    engine.add_expression(
        target_element="beam_B4",
        target_property="length",
        expression="abs(column_C4.y - column_C2.y)"
    )
    
    # 2. رابطه دال با تیرها
    print("\n  Slab-Beam relationships:")
    
    engine.add_expression(
        target_element="slab_001",
        target_property="span_x",
        expression="beam_B1.length"
    )
    
    engine.add_expression(
        target_element="slab_001",
        target_property="span_y",
        expression="beam_B3.length"
    )
    
    # 3. محدودیت‌ها
    print("\n  Adding constraints:")
    
    # تیرهای مخالف باید هم اندازه باشند
    engine.add_constraint(
        element1_id="beam_B1",
        element2_id="beam_B2",
        constraint_type=ConstraintType.DISTANCE,
        value=0  # باید برابر باشند
    )
    
    # اعتبارسنجی
    engine.validate_graph()
    
    return engine


def perform_structural_analysis(
    graph: CADGraph,
    verbose: bool = True
) -> StructuralAnalyzer:
    """
    انجام تحلیل ساختاری
    
    Args:
        graph: CAD Graph
        verbose: نمایش جزئیات
    
    Returns:
        StructuralAnalyzer با نتایج
    """
    if verbose:
        print("\n" + "="*70)
        print("Performing Structural Analysis")
        print("="*70)
    
    analyzer = StructuralAnalyzer(graph, IndustryType.BUILDING)
    
    # تحلیل ستون‌ها
    if verbose:
        print("\n🔷 Analyzing Columns...")
    
    for elem_id in ['column_C1', 'column_C2', 'column_C3', 'column_C4']:
        element = graph.get_element(elem_id)
        height = element.properties['height'] / 1000  # mm → m
        
        # بار محوری (وزن دال + بار زنده)
        # فرض: هر ستون 1/4 از بار کل را تحمل می‌کند
        slab = graph.get_element('slab_001')
        area = (slab.properties['span_x'] * slab.properties['span_y']) / 1e6  # mm² → m²
        
        dead_load = 5000 * area / 4  # 5 kN/m² بار مرده
        live_load = 3000 * area / 4  # 3 kN/m² بار زنده
        
        analyzer.analyze_column(
            element_id=elem_id,
            material=STEEL_S355,
            section=HEB_300,
            loads=[
                Load(LoadType.DEAD, dead_load),
                Load(LoadType.LIVE, live_load)
            ],
            height=height,
            effective_length_factor=1.0
        )
    
    # تحلیل تیرها
    if verbose:
        print("\n🔷 Analyzing Beams...")
    
    for elem_id in ['beam_B1', 'beam_B2', 'beam_B3', 'beam_B4']:
        element = graph.get_element(elem_id)
        length = element.properties['length'] / 1000  # mm → m
        
        # بار توزیع شده از دال
        load_per_meter = 8000  # 8 kN/m (بار مرده + زنده)
        total_load = load_per_meter * length * 1000  # kN → N
        
        analyzer.analyze_beam(
            element_id=elem_id,
            material=STEEL_S355,
            section=IPE_300,
            loads=[Load(LoadType.DEAD, total_load)],
            length=length,
            support_conditions="simply_supported"
        )
    
    # تحلیل دال
    if verbose:
        print("\n🔷 Analyzing Slab...")
    
    slab = graph.get_element('slab_001')
    
    analyzer.analyze_slab(
        element_id='slab_001',
        material=CONCRETE_C30,
        thickness=slab.properties['thickness'] / 1000,  # mm → m
        loads=[
            Load(LoadType.DEAD, 5000),  # 5 kN/m²
            Load(LoadType.LIVE, 3000),  # 3 kN/m²
        ],
        span_x=slab.properties['span_x'] / 1000,  # mm → m
        span_y=slab.properties['span_y'] / 1000,
        support_type="four_edges"
    )
    
    # خلاصه
    analyzer.analyze_structure()
    
    return analyzer


def scenario_parametric_design_optimization():
    """
    سناریو: بهینه‌سازی طراحی پارامتریک
    
    ما دهانه ساختمان را تغییر می‌دهیم و تحلیل می‌کنیم
    تا بهترین دهانه را پیدا کنیم که:
    - ایمن باشد
    - اقتصادی باشد
    """
    print("\n" + "="*70)
    print("SCENARIO: Parametric Design Optimization")
    print("="*70)
    print("\nGoal: Find optimal span that is both safe and economical")
    
    # ساخت ساختمان
    graph = create_simple_building()
    engine = setup_parametric_relationships(graph)
    
    # تحلیل اولیه (6m)
    print("\n" + "="*70)
    print("📐 Initial Design (6m × 6m)")
    print("="*70)
    
    analyzer_6m = perform_structural_analysis(graph, verbose=False)
    summary_6m = analyzer_6m.analyze_structure()
    
    # تغییر به 8 متر
    print("\n" + "="*70)
    print("🔄 Changing span to 8m × 8m")
    print("="*70)
    
    # تغییر موقعیت ستون‌ها
    engine.update_parameter("column_C2", "x", 8000, propagate=True)
    engine.update_parameter("column_C4", "x", 8000, propagate=True)
    engine.update_parameter("column_C3", "y", 8000, propagate=True)
    engine.update_parameter("column_C4", "y", 8000, propagate=True)
    
    # تحلیل مجدد
    analyzer_8m = perform_structural_analysis(graph, verbose=False)
    summary_8m = analyzer_8m.analyze_structure()
    
    # مقایسه
    print("\n" + "="*70)
    print("📊 COMPARISON")
    print("="*70)
    
    print("\n  6m × 6m:")
    print(f"    Safe: {summary_6m['safe_elements']}/{summary_6m['analyzed_elements']}")
    print(f"    Max stress ratio: {summary_6m['max_stress_ratio']:.2f}")
    print(f"    Max deflection ratio: {summary_6m['max_deflection_ratio']:.2f}")
    
    print("\n  8m × 8m:")
    print(f"    Safe: {summary_8m['safe_elements']}/{summary_8m['analyzed_elements']}")
    print(f"    Max stress ratio: {summary_8m['max_stress_ratio']:.2f}")
    print(f"    Max deflection ratio: {summary_8m['max_deflection_ratio']:.2f}")
    
    # نتیجه‌گیری
    print("\n" + "="*70)
    print("💡 CONCLUSION")
    print("="*70)
    
    if summary_6m['unsafe_elements'] == 0 and summary_8m['unsafe_elements'] > 0:
        print("\n✅ 6m span is optimal - safe and economical")
        print("❌ 8m span requires larger sections")
    elif summary_8m['unsafe_elements'] == 0:
        print("\n✅ 8m span is feasible but may be more expensive")
        print("💰 Consider cost vs. usable space trade-off")
    else:
        print("\n⚠️  Both designs may need refinement")
    
    # ذخیره نتایج
    analyzer_6m.export_results(Path("analysis_6m.json"))
    analyzer_8m.export_results(Path("analysis_8m.json"))
    engine.export_to_json(Path("parametric_relationships.json"))
    
    print("\n✅ Scenario complete!")


def scenario_load_increase_analysis():
    """
    سناریو: تحلیل افزایش بار
    
    فرض کنید می‌خواهیم بار زنده را افزایش دهیم
    (مثلاً برای استفاده کتابخانه یا انبار)
    
    آیا سازه کنونی کافی است؟
    """
    print("\n" + "="*70)
    print("SCENARIO: Load Increase Analysis")
    print("="*70)
    print("\nQuestion: Can we increase live load from 3 kN/m² to 5 kN/m²?")
    
    # ساخت ساختمان
    graph = create_simple_building()
    setup_parametric_relationships(graph)
    
    # تحلیل با بار استانداد (3 kN/m²)
    print("\n📐 Standard Load (3 kN/m² live load)")
    analyzer_standard = StructuralAnalyzer(graph, IndustryType.BUILDING)
    
    # فقط تحلیل دال (که مستقیماً تحت تأثیر است)
    slab = graph.get_element('slab_001')
    
    result_3kn = analyzer_standard.analyze_slab(
        element_id='slab_001',
        material=CONCRETE_C30,
        thickness=slab.properties['thickness'] / 1000,
        loads=[
            Load(LoadType.DEAD, 5000),  # 5 kN/m²
            Load(LoadType.LIVE, 3000),  # 3 kN/m²
        ],
        span_x=slab.properties['span_x'] / 1000,
        span_y=slab.properties['span_y'] / 1000
    )
    
    # تحلیل با بار افزایش یافته (5 kN/m²)
    print("\n📐 Increased Load (5 kN/m² live load)")
    analyzer_increased = StructuralAnalyzer(graph, IndustryType.BUILDING)
    
    result_5kn = analyzer_increased.analyze_slab(
        element_id='slab_001',
        material=CONCRETE_C30,
        thickness=slab.properties['thickness'] / 1000,
        loads=[
            Load(LoadType.DEAD, 5000),  # 5 kN/m²
            Load(LoadType.LIVE, 5000),  # 5 kN/m² (افزایش یافته)
        ],
        span_x=slab.properties['span_x'] / 1000,
        span_y=slab.properties['span_y'] / 1000
    )
    
    # مقایسه
    print("\n" + "="*70)
    print("📊 COMPARISON")
    print("="*70)
    
    print(f"\n  3 kN/m² live load:")
    print(f"    Stress ratio: {result_3kn.stress_ratio:.2f}")
    print(f"    Deflection ratio: {result_3kn.deflection_ratio:.2f}")
    print(f"    Safe: {'✅ YES' if result_3kn.is_safe else '❌ NO'}")
    
    print(f"\n  5 kN/m² live load:")
    print(f"    Stress ratio: {result_5kn.stress_ratio:.2f}")
    print(f"    Deflection ratio: {result_5kn.deflection_ratio:.2f}")
    print(f"    Safe: {'✅ YES' if result_5kn.is_safe else '❌ NO'}")
    
    # نتیجه‌گیری
    print("\n" + "="*70)
    print("💡 RECOMMENDATION")
    print("="*70)
    
    if result_5kn.is_safe:
        print("\n✅ Load increase is acceptable with current design")
    elif result_5kn.stress_ratio < 1.0 and result_5kn.deflection_ratio > 1.0:
        print("\n⚠️  Deflection exceeds limit - need thicker slab")
        print(f"   Suggested: Increase from {slab.properties['thickness']}mm to {slab.properties['thickness']*1.2:.0f}mm")
    elif result_5kn.stress_ratio > 1.0:
        print("\n❌ Stress exceeds limit - need structural reinforcement")
        print("   Options:")
        print("   1. Increase slab thickness")
        print("   2. Add additional beams")
        print("   3. Reduce span")
    
    print("\n✅ Scenario complete!")


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║        COMPLETE EXAMPLE: PARAMETRIC + STRUCTURAL ANALYSIS        ║
║                                                                  ║
║  این مثال نشان می‌دهد چگونه سیستم‌های پارامتریک و تحلیل        ║
║  ساختاری با هم کار می‌کنند تا طراحی بهینه ایجاد کنند           ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")
    
    print("\nAvailable Scenarios:")
    print("  1. Parametric Design Optimization")
    print("  2. Load Increase Analysis")
    print("  3. Both")
    
    choice = input("\nSelect scenario (1/2/3) [default: 3]: ").strip() or "3"
    
    if choice in ["1", "3"]:
        scenario_parametric_design_optimization()
    
    if choice in ["2", "3"]:
        scenario_load_increase_analysis()
    
    print("\n" + "="*70)
    print("✅ ALL SCENARIOS COMPLETE!")
    print("="*70)
