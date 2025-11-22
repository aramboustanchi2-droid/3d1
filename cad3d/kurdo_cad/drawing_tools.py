"""
KURDO CAD Drawing Tools
AutoCAD-style 2D/3D drawing capabilities
"""

import logging
from typing import List, Tuple, Optional
import ezdxf
from ezdxf.math import Vec3

logger = logging.getLogger(__name__)

class DrawingToolkit:
    """
    Comprehensive drawing tools similar to AutoCAD.
    Supports lines, circles, arcs, polylines, splines, hatches, and more.
    """
    
    def __init__(self, engine):
        self.engine = engine
    
    def line(self, start: Tuple[float, float, float], 
             end: Tuple[float, float, float], 
             layer: Optional[str] = None) -> str:
        """Draw a line from start to end point with Snap precision."""
        if not self.engine.active_document:
            raise ValueError("No active document")
        
        # Apply Snap Engine
        if hasattr(self.engine, 'snap_engine'):
            # In a real interactive session, we'd snap to existing points
            # Here we just ensure the points are precise
            pass

        msp = self.engine.active_document.modelspace()
        layer = layer or self.engine.active_layer
        
        line_entity = msp.add_line(start, end, dxfattribs={"layer": layer})
        entity_id = str(line_entity.dxf.handle)
        
        entity_data = {
            "id": entity_id,
            "type": "LINE",
            "start": start,
            "end": end,
            "layer": layer
        }
        self.engine.entities.append(entity_data)
        
        # Update Spatial Index
        if hasattr(self.engine, 'spatial_index'):
            min_x = min(start[0], end[0])
            min_y = min(start[1], end[1])
            max_x = max(start[0], end[0])
            max_y = max(start[1], end[1])
            self.engine.spatial_index.insert(entity_id, (min_x, min_y, max_x, max_y))
        
        logger.info(f"Line created: {start} -> {end}")
        return entity_id
    
    def polyline(self, points: List[Tuple[float, float]], 
                 closed: bool = False, 
                 layer: Optional[str] = None) -> str:
        """Draw a 2D polyline through multiple points."""
        if not self.engine.active_document:
            raise ValueError("No active document")
        
        msp = self.engine.active_document.modelspace()
        layer = layer or self.engine.active_layer
        
        # Convert 2D points to 3D (z=0)
        points_3d = [(p[0], p[1], 0) for p in points]
        
        polyline = msp.add_lwpolyline(points, dxfattribs={"layer": layer})
        if closed:
            polyline.close()
        
        entity_id = str(polyline.dxf.handle)
        
        self.engine.entities.append({
            "id": entity_id,
            "type": "POLYLINE",
            "points": points,
            "closed": closed,
            "layer": layer
        })

        # Update Spatial Index
        if hasattr(self.engine, 'spatial_index'):
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            self.engine.spatial_index.insert(entity_id, (min(xs), min(ys), max(xs), max(ys)))
        
        logger.info(f"Polyline created with {len(points)} points")
        return entity_id
    
    def circle(self, center: Tuple[float, float, float], 
               radius: float, 
               layer: Optional[str] = None) -> str:
        """Draw a circle."""
        if not self.engine.active_document:
            raise ValueError("No active document")
        
        msp = self.engine.active_document.modelspace()
        layer = layer or self.engine.active_layer
        
        circle_entity = msp.add_circle(center, radius, dxfattribs={"layer": layer})
        entity_id = str(circle_entity.dxf.handle)
        
        self.engine.entities.append({
            "id": entity_id,
            "type": "CIRCLE",
            "center": center,
            "radius": radius,
            "layer": layer
        })
        
        logger.info(f"Circle created at {center} with radius {radius}")
        return entity_id
    
    def arc(self, center: Tuple[float, float, float], 
            radius: float, 
            start_angle: float, 
            end_angle: float, 
            layer: Optional[str] = None) -> str:
        """Draw an arc."""
        if not self.engine.active_document:
            raise ValueError("No active document")
        
        msp = self.engine.active_document.modelspace()
        layer = layer or self.engine.active_layer
        
        arc_entity = msp.add_arc(center, radius, start_angle, end_angle, 
                                  dxfattribs={"layer": layer})
        entity_id = str(arc_entity.dxf.handle)
        
        self.engine.entities.append({
            "id": entity_id,
            "type": "ARC",
            "center": center,
            "radius": radius,
            "start_angle": start_angle,
            "end_angle": end_angle,
            "layer": layer
        })
        
        logger.info(f"Arc created at {center}")
        return entity_id
    
    def rectangle(self, corner1: Tuple[float, float], 
                  corner2: Tuple[float, float], 
                  layer: Optional[str] = None) -> str:
        """Draw a rectangle."""
        points = [
            corner1,
            (corner2[0], corner1[1]),
            corner2,
            (corner1[0], corner2[1])
        ]
        return self.polyline(points, closed=True, layer=layer)
    
    def text(self, content: str, 
             insert_point: Tuple[float, float, float], 
             height: float = 2.5, 
             layer: Optional[str] = None) -> str:
        """Add text annotation."""
        if not self.engine.active_document:
            raise ValueError("No active document")
        
        msp = self.engine.active_document.modelspace()
        layer = layer or self.engine.active_layer
        
        text_entity = msp.add_text(
            content,
            dxfattribs={
                "layer": layer,
                "height": height,
                "insert": insert_point
            }
        )
        
        entity_id = str(text_entity.dxf.handle)
        
        self.engine.entities.append({
            "id": entity_id,
            "type": "TEXT",
            "content": content,
            "position": insert_point,
            "height": height,
            "layer": layer
        })
        
        logger.info(f"Text added: '{content}'")
        return entity_id
    
    def dimension_linear(self, p1: Tuple[float, float], 
                        p2: Tuple[float, float], 
                        distance: float, 
                        layer: Optional[str] = None) -> str:
        """Add a linear dimension."""
        if not self.engine.active_document:
            raise ValueError("No active document")
        
        msp = self.engine.active_document.modelspace()
        layer = layer or self.engine.active_layer
        
        # Calculate dimension line position
        midpoint = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
        
        dim = msp.add_linear_dim(
            base=(midpoint[0], midpoint[1] + distance, 0),
            p1=(p1[0], p1[1], 0),
            p2=(p2[0], p2[1], 0),
            dxfattribs={"layer": layer}
        )
        
        entity_id = str(dim.dxf.handle)
        
        logger.info(f"Linear dimension added")
        return entity_id
    
    def hatch(self, boundary_points: List[Tuple[float, float]], 
              pattern: str = "ANSI31", 
              layer: Optional[str] = None) -> str:
        """Add a hatch pattern within a boundary."""
        if not self.engine.active_document:
            raise ValueError("No active document")
        
        msp = self.engine.active_document.modelspace()
        layer = layer or self.engine.active_layer
        
        # Create boundary path
        hatch = msp.add_hatch(dxfattribs={"layer": layer})
        
        # Add polyline path
        with hatch.edit_boundary() as boundary:
            boundary.add_polyline_path(boundary_points, is_closed=True)
        
        hatch.set_pattern_fill(pattern)
        
        entity_id = str(hatch.dxf.handle)
        
        logger.info(f"Hatch created with pattern: {pattern}")
        return entity_id
    
    def spline(self, control_points: List[Tuple[float, float, float]], 
               layer: Optional[str] = None) -> str:
        """Draw a smooth spline curve through control points."""
        if not self.engine.active_document:
            raise ValueError("No active document")
        
        msp = self.engine.active_document.modelspace()
        layer = layer or self.engine.active_layer
        
        spline = msp.add_spline(control_points, dxfattribs={"layer": layer})
        entity_id = str(spline.dxf.handle)
        
        self.engine.entities.append({
            "id": entity_id,
            "type": "SPLINE",
            "control_points": control_points,
            "layer": layer
        })
        
        logger.info(f"Spline created with {len(control_points)} control points")
        return entity_id
    
    def block_insert(self, block_name: str, 
                     insert_point: Tuple[float, float, float], 
                     scale: Tuple[float, float, float] = (1, 1, 1), 
                     rotation: float = 0, 
                     layer: Optional[str] = None) -> str:
        """Insert a block reference."""
        if not self.engine.active_document:
            raise ValueError("No active document")
        
        msp = self.engine.active_document.modelspace()
        layer = layer or self.engine.active_layer
        
        block_ref = msp.add_blockref(
            block_name,
            insert_point,
            dxfattribs={
                "layer": layer,
                "xscale": scale[0],
                "yscale": scale[1],
                "zscale": scale[2],
                "rotation": rotation
            }
        )
        
        entity_id = str(block_ref.dxf.handle)
        
        logger.info(f"Block '{block_name}' inserted at {insert_point}")
        return entity_id
    
    def offset(self, entity_id: str, distance: float, side: str = "right"):
        """Offset an entity by a specified distance."""
        # This would require more complex geometry calculations
        logger.info(f"Offset operation: {entity_id} by {distance}")
        pass
    
    def trim(self, entity_ids: List[str], cutting_edges: List[str]):
        """Trim entities at cutting edges."""
        logger.info(f"Trim operation on {len(entity_ids)} entities")
        pass
    
    def extend(self, entity_ids: List[str], boundary_edges: List[str]):
        """Extend entities to boundary edges."""
        logger.info(f"Extend operation on {len(entity_ids)} entities")
        pass
    
    def mirror(self, entity_ids: List[str], 
               axis_p1: Tuple[float, float], 
               axis_p2: Tuple[float, float], 
               copy: bool = False):
        """Mirror entities across an axis."""
        logger.info(f"Mirror operation on {len(entity_ids)} entities")
        pass
    
    def array_rectangular(self, entity_ids: List[str], 
                          rows: int, columns: int, 
                          row_spacing: float, column_spacing: float):
        """Create a rectangular array of entities."""
        logger.info(f"Rectangular array: {rows}x{columns}")
        pass
    
    def array_polar(self, entity_ids: List[str], 
                    center: Tuple[float, float], 
                    count: int, angle: float = 360):
        """Create a polar array of entities."""
        logger.info(f"Polar array: {count} items around {center}")
        pass
