"""
KURDO CAD BIM Tools
Revit-style Building Information Modeling capabilities
"""

import logging
from typing import List, Tuple, Optional, Dict
from .drawing_tools import DrawingToolkit

logger = logging.getLogger(__name__)

class BIMToolkit:
    """
    BIM capabilities for creating intelligent building elements.
    Manages Walls, Doors, Windows, Slabs, and Roofs.
    """
    
    def __init__(self, engine):
        self.engine = engine
        self.draw = DrawingToolkit(engine)
    
    def create_wall(self, start: Tuple[float, float], 
                    end: Tuple[float, float], 
                    thickness: float = 200, 
                    height: float = 3000, 
                    type_: str = "Generic 200mm",
                    layer: str = "A-WALL") -> str:
        """Create a wall element with 2D representation and 3D data."""
        
        # Snap to existing points if possible
        if hasattr(self.engine, 'snap_engine'):
            # In a real scenario, we would query the spatial index for nearby points
            # candidates = self.engine.spatial_index.query(start[0], start[1], 100)
            # start = self.engine.snap_engine.snap_point(start, candidates)
            pass

        # Calculate wall geometry (simplified 2D representation)
        # In a real engine, this would handle vector math for thickness
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = (dx**2 + dy**2)**0.5
        
        # Normalize direction
        if length > 0:
            ux, uy = dx/length, dy/length
            # Perpendicular vector
            px, py = -uy, ux
            
            offset_x = px * (thickness / 2)
            offset_y = py * (thickness / 2)
            
            p1 = (start[0] + offset_x, start[1] + offset_y)
            p2 = (end[0] + offset_x, end[1] + offset_y)
            p3 = (end[0] - offset_x, end[1] - offset_y)
            p4 = (start[0] - offset_x, start[1] - offset_y)
            
            # Draw 2D footprint
            entity_id = self.draw.polyline([p1, p2, p3, p4], closed=True, layer=layer)
            
            # Store BIM data
            element_id = f"WALL_{entity_id}"
            self.engine.elements[element_id] = {
                "type": "WALL",
                "family": type_,
                "geometry": {"start": start, "end": end, "thickness": thickness, "height": height},
                "properties": {"fire_rating": "1hr", "structural": True},
                "entity_id": entity_id
            }
            
            # Update Spatial Index for the Wall
            if hasattr(self.engine, 'spatial_index'):
                min_x = min(p1[0], p2[0], p3[0], p4[0])
                min_y = min(p1[1], p2[1], p3[1], p4[1])
                max_x = max(p1[0], p2[0], p3[0], p4[0])
                max_y = max(p1[1], p2[1], p3[1], p4[1])
                self.engine.spatial_index.insert(element_id, (min_x, min_y, max_x, max_y))

            logger.info(f"Wall created: {type_} (L={length:.1f}, H={height})")
            return element_id
        return None

    def place_door(self, wall_id: str, 
                   position: Tuple[float, float], 
                   width: float = 900, 
                   type_: str = "Single Flush",
                   layer: str = "A-DOOR") -> str:
        """Place a door hosted in a wall."""
        if wall_id not in self.engine.elements:
            raise ValueError("Host wall not found")
        
        # Draw door symbol (simplified)
        # In reality, this would insert a dynamic block
        p1 = position
        p2 = (position[0] + width, position[1]) # Simplified
        
        # Door swing arc
        entity_id = self.draw.arc(p1, width, 0, 90, layer=layer)
        
        element_id = f"DOOR_{entity_id}"
        self.engine.elements[element_id] = {
            "type": "DOOR",
            "family": type_,
            "host": wall_id,
            "geometry": {"position": position, "width": width},
            "entity_id": entity_id
        }
        
        logger.info(f"Door placed: {type_} (W={width})")
        return element_id

    def place_window(self, wall_id: str, 
                     position: Tuple[float, float], 
                     width: float = 1200, 
                     height: float = 1500,
                     sill_height: float = 900,
                     type_: str = "Fixed",
                     layer: str = "A-WIND") -> str:
        """Place a window hosted in a wall."""
        if wall_id not in self.engine.elements:
            raise ValueError("Host wall not found")
        
        # Draw window symbol
        p1 = (position[0] - width/2, position[1])
        p2 = (position[0] + width/2, position[1])
        
        entity_id = self.draw.line((p1[0], p1[1], 0), (p2[0], p2[1], 0), layer=layer)
        
        element_id = f"WIND_{entity_id}"
        self.engine.elements[element_id] = {
            "type": "WINDOW",
            "family": type_,
            "host": wall_id,
            "geometry": {"position": position, "width": width, "height": height, "sill": sill_height},
            "entity_id": entity_id
        }
        
        logger.info(f"Window placed: {type_} (W={width} H={height})")
        return element_id

    def create_slab(self, boundary_points: List[Tuple[float, float]], 
                    thickness: float = 150, 
                    level: float = 0,
                    type_: str = "Concrete 150mm",
                    layer: str = "A-FLOR") -> str:
        """Create a floor slab."""
        entity_id = self.draw.polyline(boundary_points, closed=True, layer=layer)
        
        element_id = f"SLAB_{entity_id}"
        self.engine.elements[element_id] = {
            "type": "SLAB",
            "family": type_,
            "geometry": {"boundary": boundary_points, "thickness": thickness, "level": level},
            "entity_id": entity_id
        }
        
        logger.info(f"Slab created: {type_}")
        return element_id

    def create_room(self, boundary_points: List[Tuple[float, float]], 
                    name: str = "Room", 
                    number: str = "101",
                    layer: str = "A-ANNO") -> str:
        """Define a room/space with area calculation."""
        # Calculate area (Shoelace formula)
        area = 0.0
        for i in range(len(boundary_points)):
            j = (i + 1) % len(boundary_points)
            area += boundary_points[i][0] * boundary_points[j][1]
            area -= boundary_points[j][0] * boundary_points[i][1]
        area = abs(area) / 2.0
        
        # Draw room tag
        center_x = sum(p[0] for p in boundary_points) / len(boundary_points)
        center_y = sum(p[1] for p in boundary_points) / len(boundary_points)
        
        tag_text = f"{name}\n{number}\n{area/1000000:.1f} m²" # Assuming mm units
        entity_id = self.draw.text(tag_text, (center_x, center_y, 0), height=250, layer=layer)
        
        element_id = f"ROOM_{entity_id}"
        self.engine.spaces[element_id] = {
            "name": name,
            "number": number,
            "area": area,
            "boundary": boundary_points,
            "entity_id": entity_id
        }
        
        logger.info(f"Room created: {name} {number} ({area/1000000:.2f} m²)")
        return element_id
