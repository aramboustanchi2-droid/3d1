"""
KURDO CAD Civil Tools
Civil 3D-style infrastructure design capabilities
"""

import logging
from typing import List, Tuple, Optional, Dict
from .drawing_tools import DrawingToolkit

logger = logging.getLogger(__name__)

class CivilToolkit:
    """
    Civil engineering tools for infrastructure design.
    Manages Alignments, Profiles, Surfaces, and Pipe Networks.
    """
    
    def __init__(self, engine):
        self.engine = engine
        self.draw = DrawingToolkit(engine)
    
    def create_alignment(self, name: str, 
                         points: List[Tuple[float, float]], 
                         type_: str = "Centerline",
                         layer: str = "C-ROAD") -> str:
        """Create a road alignment."""
        # Draw alignment geometry (Polyline for now, complex curves in future)
        entity_id = self.draw.polyline(points, layer=layer)
        
        # Calculate stationing (simplified)
        length = 0.0
        stations = []
        for i in range(len(points)-1):
            p1 = points[i]
            p2 = points[i+1]
            dist = ((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)**0.5
            stations.append({"station": length, "point": p1})
            length += dist
        stations.append({"station": length, "point": points[-1]})
        
        element_id = f"ALIGN_{entity_id}"
        self.engine.alignments[element_id] = {
            "name": name,
            "type": type_,
            "geometry": points,
            "length": length,
            "stations": stations,
            "entity_id": entity_id
        }
        
        logger.info(f"Alignment created: {name} (L={length:.2f})")
        return element_id
    
    def create_surface(self, name: str, 
                       points: List[Tuple[float, float, float]], 
                       layer: str = "C-TOPO") -> str:
        """Create a TIN surface from points."""
        # In a real implementation, this would perform Delaunay triangulation
        # For visualization, we'll just plot the points
        
        entity_ids = []
        for p in points:
            # Draw point marker
            eid = self.draw.circle(p, radius=50, layer=layer) # 50mm marker
            entity_ids.append(eid)
            
        element_id = f"SURF_{name}"
        self.engine.surfaces[element_id] = {
            "name": name,
            "points": points,
            "min_elev": min(p[2] for p in points),
            "max_elev": max(p[2] for p in points),
            "entity_ids": entity_ids
        }
        
        logger.info(f"Surface created: {name} ({len(points)} points)")
        return element_id
    
    def create_pipe_network(self, name: str, layer: str = "C-UTYL") -> str:
        """Initialize a new pipe network."""
        network_id = f"NET_{name}"
        self.engine.pipe_networks[network_id] = {
            "name": name,
            "structures": [],
            "pipes": []
        }
        logger.info(f"Pipe network created: {name}")
        return network_id
    
    def add_structure(self, network_id: str, 
                      position: Tuple[float, float, float], 
                      type_: str = "Manhole 1200mm",
                      layer: str = "C-UTYL") -> str:
        """Add a structure (manhole/catch basin) to a network."""
        if network_id not in self.engine.pipe_networks:
            raise ValueError("Network not found")
        
        # Draw structure symbol
        entity_id = self.draw.circle(position, radius=600, layer=layer)
        
        struct_data = {
            "id": f"STR_{entity_id}",
            "type": type_,
            "position": position,
            "entity_id": entity_id
        }
        
        self.engine.pipe_networks[network_id]["structures"].append(struct_data)
        logger.info(f"Structure added to {network_id}")
        return struct_data["id"]
    
    def add_pipe(self, network_id: str, 
                 start_struct_id: str, 
                 end_struct_id: str, 
                 diameter: float = 300,
                 layer: str = "C-UTYL") -> str:
        """Connect two structures with a pipe."""
        if network_id not in self.engine.pipe_networks:
            raise ValueError("Network not found")
        
        network = self.engine.pipe_networks[network_id]
        
        # Find start and end positions
        p1 = next((s["position"] for s in network["structures"] if s["id"] == start_struct_id), None)
        p2 = next((s["position"] for s in network["structures"] if s["id"] == end_struct_id), None)
        
        if not p1 or not p2:
            raise ValueError("Structure not found")
        
        # Draw pipe (double line for width)
        # Simplified as single line for now
        entity_id = self.draw.line(p1, p2, layer=layer)
        
        pipe_data = {
            "id": f"PIPE_{entity_id}",
            "diameter": diameter,
            "start": start_struct_id,
            "end": end_struct_id,
            "entity_id": entity_id
        }
        
        network["pipes"].append(pipe_data)
        logger.info(f"Pipe added to {network_id} (D={diameter})")
        return pipe_data["id"]
