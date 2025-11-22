"""
مثال: پل بتنی با دهانه 50 متر
Example: Concrete Bridge with 50m Span

این اسکریپت یک پل ساده با:
- دو تکیه‌گاه (abutments)
- عرشه (deck)
- تیرهای نگهدارنده
- گاردریل (railing)

را می‌سازد.
"""

from pathlib import Path
import sys
import math

sys.path.insert(0, str(Path(__file__).parent.parent))

from cad3d.cad_graph import (
    CADGraph, CADElement, ElementType, RelationType
)


def create_bridge(
    span: float = 50000,  # mm (50m)
    width: float = 8000,  # mm (8m)
    deck_thickness: float = 300  # mm
) -> CADGraph:
    """
    ساخت پل بتنی
    
    Args:
        span: دهانه پل (mm)
        width: عرض پل (mm)
        deck_thickness: ضخامت عرشه (mm)
    
    Returns:
        CADGraph containing bridge elements
    """
    
    print("="*70)
    print(f"Creating Bridge: {span/1000}m span, {width/1000}m width")
    print("="*70)
    
    graph = CADGraph()
    
    # ================================================================
    # Foundations
    # ================================================================
    print("\n🏗️  Creating Foundations...")
    
    for i, x_pos in enumerate([0, span]):
        foundation_id = f"foundation_{i:02d}"
        
        foundation = CADElement(
            id=foundation_id,
            element_type=ElementType.FOUNDATION,
            centroid=(x_pos, width/2, -3000),  # 3m below ground
            properties={
                'width': width + 2000,
                'length': 5000,
                'depth': 3000,
                'type': 'spread_footing',
                'material': 'concrete_C30',
                'soil_bearing_capacity': '300 kPa'
            },
            bounding_box=(
                (x_pos - 2500, -1000, -6000),
                (x_pos + 2500, width + 1000, 0)
            )
        )
        graph.add_element(foundation)
    
    print(f"   ✓ 2 foundations created")
    
    # ================================================================
    # Abutments (تکیه‌گاه‌ها)
    # ================================================================
    print("\n🏛️  Creating Abutments...")
    
    abutment_height = 5000  # 5m
    abutment_ids = []
    
    for i, x_pos in enumerate([0, span]):
        abutment_id = f"abutment_{i:02d}"
        
        abutment = CADElement(
            id=abutment_id,
            element_type=ElementType.BRIDGE,
            centroid=(x_pos, width/2, abutment_height/2),
            properties={
                'type': 'abutment',
                'height': abutment_height,
                'width': width + 1000,
                'thickness': 1000,
                'material': 'concrete_C35',
                'design': 'gravity_wall'
            },
            bounding_box=(
                (x_pos - 500, -500, 0),
                (x_pos + 500, width + 500, abutment_height)
            )
        )
        graph.add_element(abutment)
        abutment_ids.append(abutment_id)
        
        # Connect to foundation
        graph.add_relationship(
            source_id=abutment_id,
            target_id=f"foundation_{i:02d}",
            relation_type=RelationType.SUPPORTED_BY,
            weight=1.0
        )
    
    print(f"   ✓ 2 abutments created (height: {abutment_height/1000}m)")
    
    # ================================================================
    # Girders (تیرهای اصلی)
    # ================================================================
    print("\n🔩 Creating Girders...")
    
    num_girders = 4
    girder_spacing = width / (num_girders + 1)
    girder_ids = []
    
    for i in range(num_girders):
        girder_id = f"girder_{i:02d}"
        y_pos = (i + 1) * girder_spacing
        
        girder = CADElement(
            id=girder_id,
            element_type=ElementType.BEAM,
            centroid=(span/2, y_pos, abutment_height),
            properties={
                'type': 'I_beam',
                'length': span,
                'height': 1500,  # 1.5m deep I-beam
                'width': 300,
                'material': 'steel_S355',
                'section': 'IPE 750'
            },
            bounding_box=(
                (0, y_pos - 150, abutment_height - 750),
                (span, y_pos + 150, abutment_height + 750)
            )
        )
        graph.add_element(girder)
        girder_ids.append(girder_id)
        
        # Connect to abutments
        for abutment_id in abutment_ids:
            graph.add_relationship(
                source_id=girder_id,
                target_id=abutment_id,
                relation_type=RelationType.SUPPORTED_BY,
                weight=1.0
            )
    
    print(f"   ✓ {num_girders} steel girders created")
    
    # ================================================================
    # Cross Beams (تیرهای عرضی)
    # ================================================================
    print("\n🔗 Creating Cross Beams...")
    
    num_cross_beams = 10
    cross_beam_spacing = span / (num_cross_beams + 1)
    cross_beam_ids = []
    
    for i in range(num_cross_beams):
        cross_beam_id = f"cross_beam_{i:02d}"
        x_pos = (i + 1) * cross_beam_spacing
        
        cross_beam = CADElement(
            id=cross_beam_id,
            element_type=ElementType.BEAM,
            centroid=(x_pos, width/2, abutment_height),
            properties={
                'type': 'rectangular',
                'length': width,
                'height': 400,
                'width': 300,
                'material': 'concrete_C35'
            }
        )
        graph.add_element(cross_beam)
        cross_beam_ids.append(cross_beam_id)
        
        # Connect to all girders
        for girder_id in girder_ids:
            graph.add_relationship(
                source_id=cross_beam_id,
                target_id=girder_id,
                relation_type=RelationType.CONNECTED_TO,
                weight=1.0
            )
    
    print(f"   ✓ {num_cross_beams} cross beams created")
    
    # ================================================================
    # Deck (عرشه)
    # ================================================================
    print("\n🛣️  Creating Deck...")
    
    deck_id = "deck_main"
    
    deck = CADElement(
        id=deck_id,
        element_type=ElementType.SLAB,
        centroid=(span/2, width/2, abutment_height + 750),  # On top of girders
        properties={
            'length': span,
            'width': width,
            'thickness': deck_thickness,
            'material': 'concrete_C40',
            'reinforcement': 'mesh_Φ16@150',
            'design_load': 'HL-93 (AASHTO)',
            'surface': 'asphalt_50mm'
        },
        bounding_box=(
            (0, 0, abutment_height + 750),
            (span, width, abutment_height + 750 + deck_thickness)
        )
    )
    graph.add_element(deck)
    
    # Connect deck to all girders
    for girder_id in girder_ids:
        graph.add_relationship(
            source_id=deck_id,
            target_id=girder_id,
            relation_type=RelationType.SUPPORTED_BY,
            weight=1.0
        )
    
    print(f"   ✓ Deck created ({span/1000}m × {width/1000}m)")
    
    # ================================================================
    # Railings (گاردریل)
    # ================================================================
    print("\n🛡️  Creating Railings...")
    
    railing_height = 1200  # 1.2m
    deck_top = abutment_height + 750 + deck_thickness
    
    # Left and right railings
    for side, y_pos in [("left", 0), ("right", width)]:
        railing_id = f"railing_{side}"
        
        railing = CADElement(
            id=railing_id,
            element_type=ElementType.RAILING,
            centroid=(span/2, y_pos, deck_top + railing_height/2),
            properties={
                'length': span,
                'height': railing_height,
                'type': 'steel_barrier',
                'material': 'steel_S235',
                'crash_tested': True
            },
            bounding_box=(
                (0, y_pos - 50, deck_top),
                (span, y_pos + 50, deck_top + railing_height)
            )
        )
        graph.add_element(railing)
        
        # Connect to deck
        graph.add_relationship(
            source_id=railing_id,
            target_id=deck_id,
            relation_type=RelationType.HOSTED_BY,
            weight=1.0
        )
    
    print(f"   ✓ 2 railings created (height: {railing_height/1000}m)")
    
    # ================================================================
    # Bearings (تکیه‌گاه‌های لغزشی)
    # ================================================================
    print("\n⚙️  Creating Bearings...")
    
    # One bearing under each girder at each abutment
    bearing_count = 0
    for i, girder_id in enumerate(girder_ids):
        for j, x_pos in enumerate([0, span]):
            bearing_id = f"bearing_girder{i:02d}_abutment{j:02d}"
            y_pos = (i + 1) * girder_spacing
            
            bearing = CADElement(
                id=bearing_id,
                element_type=ElementType.BRIDGE,
                centroid=(x_pos, y_pos, abutment_height - 100),
                properties={
                    'type': 'elastomeric_bearing',
                    'width': 400,
                    'length': 600,
                    'thickness': 100,
                    'material': 'neoprene',
                    'capacity': '500 kN'
                }
            )
            graph.add_element(bearing)
            bearing_count += 1
            
            # Connect girder to bearing
            graph.add_relationship(
                source_id=girder_id,
                target_id=bearing_id,
                relation_type=RelationType.SUPPORTED_BY,
                weight=1.0
            )
            
            # Connect bearing to abutment
            graph.add_relationship(
                source_id=bearing_id,
                target_id=abutment_ids[j],
                relation_type=RelationType.SUPPORTED_BY,
                weight=1.0
            )
    
    print(f"   ✓ {bearing_count} elastomeric bearings created")
    
    # ================================================================
    # Summary
    # ================================================================
    stats = graph.get_statistics()
    
    print("\n" + "="*70)
    print("✅ Bridge Complete!")
    print("="*70)
    print(f"Span: {span/1000}m")
    print(f"Width: {width/1000}m")
    print(f"Total Elements: {stats['total_elements']}")
    print(f"Total Relationships: {stats['total_relationships']}")
    print(f"\nElement Breakdown:")
    for elem_type, count in stats['elements_by_type'].items():
        print(f"  {elem_type}: {count}")
    
    return graph


def main():
    """Main execution"""
    
    # Create bridge
    graph = create_bridge(
        span=50000,  # 50m
        width=8000,  # 8m
        deck_thickness=300  # 300mm
    )
    
    # Save to JSON
    output_path = Path("bridge_50m_graph.json")
    graph.save_json(output_path)
    print(f"\n💾 Graph saved to: {output_path}")
    
    # Export to DXF
    try:
        import ezdxf
        
        dxf_output = Path("bridge_50m_2d.dxf")
        doc = ezdxf.new('R2010', setup=True)
        msp = doc.modelspace()
        
        # Create layers
        doc.layers.add('FOUNDATION', color=1)
        doc.layers.add('ABUTMENT', color=2)
        doc.layers.add('GIRDER', color=3)
        doc.layers.add('DECK', color=4)
        doc.layers.add('RAILING', color=5)
        
        # Add elements
        for elem in graph.elements.values():
            if elem.centroid and elem.bounding_box:
                (x1, y1, z1), (x2, y2, z2) = elem.bounding_box
                
                # Choose layer based on type
                layer = {
                    ElementType.FOUNDATION: 'FOUNDATION',
                    ElementType.BRIDGE: 'ABUTMENT',
                    ElementType.BEAM: 'GIRDER',
                    ElementType.SLAB: 'DECK',
                    ElementType.RAILING: 'RAILING'
                }.get(elem.element_type, '0')
                
                # Draw bounding box (plan view - XY)
                points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]
                msp.add_lwpolyline(points, dxfattribs={'layer': layer})
                
                # Add center point
                x, y, z = elem.centroid
                msp.add_point((x, y), dxfattribs={'layer': layer})
        
        doc.saveas(dxf_output)
        print(f"💾 2D DXF saved to: {dxf_output}")
    
    except Exception as e:
        print(f"⚠️  Could not save DXF: {e}")
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
