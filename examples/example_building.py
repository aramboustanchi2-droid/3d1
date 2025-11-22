"""
مثال کامل: ساختمان سه طبقه
Complete Example: Three-Story Building

این اسکریپت یک ساختمان سه طبقه کامل با:
- پی و فونداسیون
- ستون‌ها و تیرها
- دیوارها
- پله‌ها
- درها و پنجره‌ها

را می‌سازد و به 3D تبدیل می‌کند.
"""

from pathlib import Path
from typing import List, Tuple
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cad3d.cad_graph import (
    CADGraph, CADElement, ElementType, RelationType
)


def create_floor(
    graph: CADGraph,
    floor_number: int,
    z_base: float,
    floor_height: float = 3000.0,
    building_width: float = 10000.0,
    building_length: float = 15000.0
) -> List[str]:
    """
    ساخت یک طبقه کامل
    
    Args:
        graph: CAD graph
        floor_number: شماره طبقه (0=همکف, 1=اول, ...)
        z_base: ارتفاع پایه طبقه (mm)
        floor_height: ارتفاع طبقه (mm)
        building_width: عرض ساختمان (mm)
        building_length: طول ساختمان (mm)
    
    Returns:
        List of element IDs created for this floor
    """
    element_ids = []
    
    # Column positions (4 corners + 2 middle columns)
    column_positions = [
        (0, 0),  # Corner 1
        (building_width, 0),  # Corner 2
        (building_width, building_length),  # Corner 3
        (0, building_length),  # Corner 4
        (building_width/2, building_length/3),  # Middle 1
        (building_width/2, 2*building_length/3)  # Middle 2
    ]
    
    column_ids = []
    
    # Create columns
    for i, (x, y) in enumerate(column_positions):
        col_id = f"column_floor{floor_number}_col{i:02d}"
        
        column = CADElement(
            id=col_id,
            element_type=ElementType.COLUMN,
            centroid=(x, y, z_base + floor_height/2),
            properties={
                'width': 400,  # 400mm square column
                'depth': 400,
                'height': floor_height,
                'material': 'concrete_C30',
                'reinforcement': '8Φ20',
                'floor': floor_number
            },
            bounding_box=(
                (x - 200, y - 200, z_base),
                (x + 200, y + 200, z_base + floor_height)
            )
        )
        
        graph.add_element(column)
        column_ids.append(col_id)
        element_ids.append(col_id)
    
    # Create beams connecting columns
    beam_connections = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Perimeter
        (0, 4), (1, 4), (4, 5), (5, 2), (5, 3)  # Interior
    ]
    
    beam_ids = []
    
    for i, (start_col, end_col) in enumerate(beam_connections):
        beam_id = f"beam_floor{floor_number}_beam{i:02d}"
        
        start_pos = column_positions[start_col]
        end_pos = column_positions[end_col]
        
        mid_x = (start_pos[0] + end_pos[0]) / 2
        mid_y = (start_pos[1] + end_pos[1]) / 2
        
        length = ((end_pos[0] - start_pos[0])**2 + 
                  (end_pos[1] - start_pos[1])**2)**0.5
        
        beam = CADElement(
            id=beam_id,
            element_type=ElementType.BEAM,
            centroid=(mid_x, mid_y, z_base + floor_height),
            properties={
                'width': 300,
                'height': 500,
                'length': length,
                'material': 'concrete_C30',
                'reinforcement': '6Φ18',
                'floor': floor_number
            }
        )
        
        graph.add_element(beam)
        beam_ids.append(beam_id)
        element_ids.append(beam_id)
        
        # Connect beam to columns
        graph.add_relationship(
            source_id=beam_id,
            target_id=column_ids[start_col],
            relation_type=RelationType.CONNECTED_TO,
            weight=1.0
        )
        graph.add_relationship(
            source_id=beam_id,
            target_id=column_ids[end_col],
            relation_type=RelationType.CONNECTED_TO,
            weight=1.0
        )
    
    # Create floor slab
    slab_id = f"slab_floor{floor_number}"
    
    slab = CADElement(
        id=slab_id,
        element_type=ElementType.SLAB,
        centroid=(building_width/2, building_length/2, z_base + floor_height),
        properties={
            'width': building_width,
            'length': building_length,
            'thickness': 200,  # 200mm slab
            'material': 'concrete_C25',
            'reinforcement': 'mesh_Φ12@200',
            'floor': floor_number
        },
        bounding_box=(
            (0, 0, z_base + floor_height),
            (building_width, building_length, z_base + floor_height + 200)
        )
    )
    
    graph.add_element(slab)
    element_ids.append(slab_id)
    
    # Connect slab to all beams
    for beam_id in beam_ids:
        graph.add_relationship(
            source_id=slab_id,
            target_id=beam_id,
            relation_type=RelationType.SUPPORTED_BY,
            weight=1.0
        )
    
    # Create walls (perimeter walls)
    wall_sections = [
        # Wall along x-axis (bottom)
        ((0, 0), (building_width, 0), "south"),
        # Wall along y-axis (right)
        ((building_width, 0), (building_width, building_length), "east"),
        # Wall along x-axis (top)
        ((building_width, building_length), (0, building_length), "north"),
        # Wall along y-axis (left)
        ((0, building_length), (0, 0), "west")
    ]
    
    wall_ids = []
    
    for i, ((x1, y1), (x2, y2), direction) in enumerate(wall_sections):
        wall_id = f"wall_floor{floor_number}_{direction}"
        
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        
        length = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
        
        wall = CADElement(
            id=wall_id,
            element_type=ElementType.WALL,
            centroid=(mid_x, mid_y, z_base + floor_height/2),
            properties={
                'length': length,
                'height': floor_height,
                'thickness': 200,  # 200mm wall
                'material': 'brick',
                'floor': floor_number,
                'direction': direction
            }
        )
        
        graph.add_element(wall)
        wall_ids.append(wall_id)
        element_ids.append(wall_id)
        
        # Connect walls to slab
        graph.add_relationship(
            source_id=wall_id,
            target_id=slab_id,
            relation_type=RelationType.SUPPORTED_BY,
            weight=1.0
        )
    
    # Add doors and windows
    if floor_number > 0:  # Not in basement
        # Door on south wall
        door_id = f"door_floor{floor_number}_main"
        door = CADElement(
            id=door_id,
            element_type=ElementType.DOOR,
            centroid=(building_width/2, 0, z_base + 1000),  # 1m from floor
            properties={
                'width': 1000,  # 1m wide
                'height': 2200,  # 2.2m high
                'type': 'swing',
                'material': 'wood',
                'floor': floor_number
            }
        )
        graph.add_element(door)
        element_ids.append(door_id)
        
        # Door hosted by wall
        graph.add_relationship(
            source_id=door_id,
            target_id=f"wall_floor{floor_number}_south",
            relation_type=RelationType.HOSTED_BY,
            weight=1.0
        )
        
        # Windows on east and west walls
        for direction, x in [("east", building_width), ("west", 0)]:
            for j in range(3):  # 3 windows per wall
                window_id = f"window_floor{floor_number}_{direction}_{j:02d}"
                y_pos = building_length * (j + 1) / 4
                
                window = CADElement(
                    id=window_id,
                    element_type=ElementType.WINDOW,
                    centroid=(x, y_pos, z_base + 1500),
                    properties={
                        'width': 1200,
                        'height': 1500,
                        'type': 'casement',
                        'material': 'glass',
                        'floor': floor_number
                    }
                )
                graph.add_element(window)
                element_ids.append(window_id)
                
                # Window hosted by wall
                graph.add_relationship(
                    source_id=window_id,
                    target_id=f"wall_floor{floor_number}_{direction}",
                    relation_type=RelationType.HOSTED_BY,
                    weight=1.0
                )
    
    return element_ids, column_ids, slab_id


def create_building() -> CADGraph:
    """ساخت یک ساختمان سه طبقه کامل"""
    
    print("="*70)
    print("Creating 3-Story Building")
    print("="*70)
    
    graph = CADGraph()
    
    building_width = 10000  # 10m
    building_length = 15000  # 15m
    floor_height = 3000  # 3m
    
    # ================================================================
    # Foundation
    # ================================================================
    print("\n🏗️  Creating Foundation...")
    
    foundation_id = "foundation_main"
    foundation = CADElement(
        id=foundation_id,
        element_type=ElementType.FOUNDATION,
        centroid=(building_width/2, building_length/2, -500),
        properties={
            'width': building_width + 2000,  # Extra 1m on each side
            'length': building_length + 2000,
            'depth': 1000,  # 1m deep
            'material': 'concrete_C25',
            'soil_bearing_capacity': '200 kPa'
        },
        bounding_box=(
            (-1000, -1000, -1000),
            (building_width + 1000, building_length + 1000, 0)
        )
    )
    graph.add_element(foundation)
    print(f"   ✓ Foundation created")
    
    # ================================================================
    # Ground Floor (همکف)
    # ================================================================
    print("\n🏢 Creating Ground Floor...")
    
    ground_elements, ground_columns, ground_slab = create_floor(
        graph=graph,
        floor_number=0,
        z_base=0,
        floor_height=floor_height,
        building_width=building_width,
        building_length=building_length
    )
    
    # Connect ground floor columns to foundation
    for col_id in ground_columns:
        graph.add_relationship(
            source_id=col_id,
            target_id=foundation_id,
            relation_type=RelationType.SUPPORTED_BY,
            weight=1.0
        )
    
    print(f"   ✓ Ground floor: {len(ground_elements)} elements")
    
    # ================================================================
    # First Floor (طبقه اول)
    # ================================================================
    print("\n🏢 Creating First Floor...")
    
    first_elements, first_columns, first_slab = create_floor(
        graph=graph,
        floor_number=1,
        z_base=floor_height,
        floor_height=floor_height,
        building_width=building_width,
        building_length=building_length
    )
    
    # Connect first floor columns to ground floor slab
    for col_id in first_columns:
        graph.add_relationship(
            source_id=col_id,
            target_id=ground_slab,
            relation_type=RelationType.SUPPORTED_BY,
            weight=1.0
        )
    
    print(f"   ✓ First floor: {len(first_elements)} elements")
    
    # ================================================================
    # Second Floor (طبقه دوم)
    # ================================================================
    print("\n🏢 Creating Second Floor...")
    
    second_elements, second_columns, second_slab = create_floor(
        graph=graph,
        floor_number=2,
        z_base=2 * floor_height,
        floor_height=floor_height,
        building_width=building_width,
        building_length=building_length
    )
    
    # Connect second floor columns to first floor slab
    for col_id in second_columns:
        graph.add_relationship(
            source_id=col_id,
            target_id=first_slab,
            relation_type=RelationType.SUPPORTED_BY,
            weight=1.0
        )
    
    print(f"   ✓ Second floor: {len(second_elements)} elements")
    
    # ================================================================
    # Roof (سقف)
    # ================================================================
    print("\n🏠 Creating Roof...")
    
    roof_id = "roof_main"
    roof = CADElement(
        id=roof_id,
        element_type=ElementType.ROOF,
        centroid=(building_width/2, building_length/2, 3*floor_height + 100),
        properties={
            'width': building_width + 1000,  # Overhang
            'length': building_length + 1000,
            'thickness': 200,
            'type': 'flat',
            'material': 'concrete_C25',
            'waterproofing': 'membrane'
        },
        bounding_box=(
            (-500, -500, 3*floor_height),
            (building_width + 500, building_length + 500, 3*floor_height + 200)
        )
    )
    graph.add_element(roof)
    
    # Connect roof to second floor slab
    graph.add_relationship(
        source_id=roof_id,
        target_id=second_slab,
        relation_type=RelationType.SUPPORTED_BY,
        weight=1.0
    )
    
    print(f"   ✓ Roof created")
    
    # ================================================================
    # Stairs (پله‌ها)
    # ================================================================
    print("\n🪜 Creating Stairs...")
    
    for floor in range(3):
        stair_id = f"stair_floor{floor}_to_floor{floor+1}"
        
        stair = CADElement(
            id=stair_id,
            element_type=ElementType.STAIR,
            centroid=(building_width - 1500, building_length - 2000, 
                     floor * floor_height + floor_height/2),
            properties={
                'width': 1200,
                'length': 3000,
                'height': floor_height,
                'num_steps': 15,
                'step_height': floor_height / 15,
                'step_depth': 3000 / 15,
                'material': 'concrete',
                'handrail': True
            }
        )
        graph.add_element(stair)
    
    print(f"   ✓ 3 stairs created")
    
    # ================================================================
    # Summary
    # ================================================================
    stats = graph.get_statistics()
    
    print("\n" + "="*70)
    print("✅ Building Complete!")
    print("="*70)
    print(f"Total Elements: {stats['total_elements']}")
    print(f"Total Relationships: {stats['total_relationships']}")
    print(f"\nElement Breakdown:")
    for elem_type, count in stats['elements_by_type'].items():
        print(f"  {elem_type}: {count}")
    
    return graph


def main():
    """Main execution"""
    
    # Create building graph
    graph = create_building()
    
    # Save to JSON
    output_path = Path("building_3story_graph.json")
    graph.save_json(output_path)
    print(f"\n💾 Graph saved to: {output_path}")
    
    # Export to DXF (2D representation)
    try:
        import ezdxf
        
        dxf_output = Path("building_3story_2d.dxf")
        doc = ezdxf.new('R2010', setup=True)
        msp = doc.modelspace()
        
        # Add elements as simple geometry
        for elem in graph.elements.values():
            if elem.centroid:
                x, y, z = elem.centroid
                
                # Different colors for different types
                color = {
                    ElementType.COLUMN: 1,  # Red
                    ElementType.BEAM: 2,    # Yellow
                    ElementType.WALL: 3,    # Green
                    ElementType.SLAB: 4,    # Cyan
                    ElementType.DOOR: 5,    # Blue
                    ElementType.WINDOW: 6,  # Magenta
                    ElementType.ROOF: 7,    # White
                    ElementType.STAIR: 30   # Orange
                }.get(elem.element_type, 7)
                
                # Add point
                msp.add_point((x, y), dxfattribs={'color': color})
                
                # Add bounding box if available
                if elem.bounding_box:
                    (x1, y1, z1), (x2, y2, z2) = elem.bounding_box
                    # Draw rectangle
                    points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]
                    msp.add_lwpolyline(points, dxfattribs={'color': color})
        
        doc.saveas(dxf_output)
        print(f"💾 2D DXF saved to: {dxf_output}")
    
    except Exception as e:
        print(f"⚠️  Could not save DXF: {e}")
    
    print("\n✅ Done! Use graph_enhanced_converter.py to convert to 3D.")


if __name__ == "__main__":
    main()
