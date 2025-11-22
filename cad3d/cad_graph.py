"""
CAD Graph Representation System

This module provides a comprehensive, graph-based system for representing and analyzing
CAD/BIM drawings. Inspired by systems like Revit, Civil3D, and Rhino/Grasshopper, it is
designed for deep AI-driven analysis, parametric updates, and integration with GNNs.

Architecture:
- Node: Each CAD element (wall, column, beam, window, door, pipe, road, bridge, tunnel, dam, etc.)
- Edge: Relationship between elements (connection, dependency, constraint)
- Graph: The entire drawing with all parametric relationships

Key Features:
- Structural and spatial analysis
- Relationship and dependency detection
- Parametric updates (like Revit)
- GNN-ready data structures for deep learning
- Extensible for all engineering domains
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from enum import Enum, auto
import json
import numpy as np

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    print("⚠️ NetworkX not available. Install: pip install networkx")


# ============================================================================
# Universal Component & Dependency Definitions
# ============================================================================

class ComponentType(Enum):
    """
    A universal enumeration for component types across various engineering domains.
    This replaces the specific `ElementType`.
    """
    # Architectural / Civil
    COLUMN = auto()
    BEAM = auto()
    SLAB = auto()
    WALL = auto()
    FOUNDATION = auto()
    STAIR = auto()
    DOOR = auto()
    WINDOW = auto()
    PIPE = auto()
    DUCT = auto()

    # Mechanical / Robotic
    FRAME = auto()         # Structural frame or chassis
    LINKAGE = auto()       # A rigid link in a mechanism
    JOINT = auto()         # A revolute, prismatic, or spherical joint
    GEAR = auto()          # A gear for power transmission
    MOTOR = auto()         # An actuator that produces motion
    SENSOR = auto()        # A device that measures a physical quantity
    PROCESSOR = auto()     # A computing unit (e.g., microcontroller)
    BATTERY = auto()       # An energy source
    WIRE = auto()          # Electrical wiring or data cable
    FASTENER = auto()      # Bolts, screws, rivets

    # Generic
    GENERIC_SOLID = auto() # A generic solid part
    GENERIC_VOID = auto()  # A generic void or opening
    UNKNOWN = auto()

class DependencyType(Enum):
    """
    A universal enumeration for relationships and dependencies between components.
    This replaces the specific `RelationType`.
    """
    # Structural Dependencies
    SUPPORTED_BY = auto()   # e.g., A beam is supported by a column
    HOSTED_BY = auto()      # e.g., A window is hosted by a wall
    CONNECTED_TO = auto()   # General physical connection

    # Mechanical & Functional Dependencies
    MOUNTS_TO = auto()      # e.g., A motor mounts to the frame
    ACTUATES = auto()       # e.g., A motor actuates a linkage
    ROTATES_WITH = auto()   # e.g., A gear rotates with a shaft
    TRANSMITS_TO = auto()   # e.g., A gear transmits power to another gear
    CONTROLS = auto()       # e.g., A processor controls a motor
    FEEDS_DATA_TO = auto()  # e.g., A sensor feeds data to a processor
    POWERED_BY = auto()     # e.g., A motor is powered by a battery

    # Flow & Containment
    FLOWS_INTO = auto()     # e.g., Water in a pipe flows into another
    CONTAINS = auto()       # e.g., A room contains furniture (not a strong dependency)

    # Generic
    ADJACENT_TO = auto()    # Spatial adjacency without direct connection
    INTERSECTS = auto()     # Two components intersect

# ============================================================================
# Core Data Structures
# ============================================================================

class CADComponent:
    """
    Represents a single component in a design (replaces `CADElement`).
    It's a universal object for any part in an architectural or mechanical system.
    """
    def __init__(self,
                 id: str,
                 component_type: ComponentType,
                 name: Optional[str] = None,
                 parameters: Optional[Dict[str, Any]] = None,
                 centroid: Optional[Tuple[float, float, float]] = None,
                 structural_usage: Optional[str] = None): # For architectural compatibility
        self.id = id
        self.component_type = component_type
        self.name = name or f"{component_type.name}_{id}"
        self.parameters = parameters or {}
        self.centroid = centroid
        self.structural_usage = structural_usage # e.g., 'load_bearing', 'non_structural'

    def __repr__(self) -> str:
        return f"CADComponent(id={self.id}, type={self.component_type.name})"

    def get_feature_vector(self) -> np.ndarray:
        """
        Generates a feature vector for this component for machine learning.
        """
        # Type (one-hot encoded)
        type_vec = np.zeros(len(ComponentType))
        type_vec[self.component_type.value - 1] = 1

        # Parameters (e.g., length, width, height, radius)
        # We'll define a fixed set of parameters for consistency.
        param_keys = ['length', 'width', 'height', 'radius', 'thickness']
        param_vec = np.array([self.parameters.get(k, 0.0) for k in param_keys])

        # Centroid (x, y, z)
        centroid_vec = np.array(self.centroid) if self.centroid else np.zeros(3)

        # Combine all features
        return np.concatenate([type_vec, param_vec, centroid_vec]).astype(np.float32)

class CADDependency:
    """
    Represents a dependency between two components (replaces `CADRelationship`).
    """
    def __init__(self,
                 source_id: str,
                 target_id: str,
                 dependency_type: DependencyType,
                 weight: float = 1.0,
                 parameters: Optional[Dict[str, Any]] = None):
        self.source_id = source_id
        self.target_id = target_id
        self.dependency_type = dependency_type
        self.weight = weight
        self.parameters = parameters or {}

    def __repr__(self) -> str:
        return f"CADDependency({self.source_id} -> {self.target_id}, type={self.dependency_type.name})"

    def get_feature_vector(self) -> np.ndarray:
        """
        Generates a feature vector for this dependency for machine learning.
        """
        # Type (one-hot encoded)
        type_vec = np.zeros(len(DependencyType))
        type_vec[self.dependency_type.value - 1] = 1

        # Weight
        weight_vec = np.array([self.weight])

        # Combine all features
        return np.concatenate([type_vec, weight_vec]).astype(np.float32)

class CADGraph:
    """
    Represents a complete design as a graph of components and their dependencies.
    """
    def __init__(self, name: str):
        self.name = name
        self.components: Dict[str, CADComponent] = {}
        self.dependencies: List[CADDependency] = []

    def add_component(self, component: CADComponent):
        """Adds a component to the graph."""
        if component.id in self.components:
            # Handle update logic if necessary
            pass
        self.components[component.id] = component

    def add_dependency(self, dependency: CADDependency):
        """Adds a dependency to the graph."""
        self.dependencies.append(dependency)

    # ------------------------------------------------------------------
    # Persistence helpers (non‑breaking, only add new capabilities)
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the graph to a plain Python dict (JSON‑friendly).

        This is designed to be stable and backwards‑compatible so that other
        parts of the system can store and reload graphs without losing any
        information. It does not remove or change any existing behaviour; it
        only adds export capabilities on top of the in‑memory model.
        """

        def _component_to_dict(c: CADComponent) -> Dict[str, Any]:
            return {
                "id": c.id,
                "component_type": c.component_type.name,
                "name": c.name,
                "parameters": c.parameters,
                "centroid": list(c.centroid) if c.centroid is not None else None,
                "structural_usage": c.structural_usage,
            }

        def _dependency_to_dict(d: CADDependency) -> Dict[str, Any]:
            return {
                "source_id": d.source_id,
                "target_id": d.target_id,
                "dependency_type": d.dependency_type.name,
                "weight": d.weight,
                "parameters": d.parameters,
            }

        # Extra attributes (levels, metadata, etc.) are preserved in a
        # dedicated section if they exist, without enforcing any schema.
        extra: Dict[str, Any] = {}
        for key, value in self.__dict__.items():
            if key in {"name", "components", "dependencies"}:
                continue
            extra[key] = value

        return {
            "name": self.name,
            "components": [_component_to_dict(c) for c in self.components.values()],
            "dependencies": [_dependency_to_dict(d) for d in self.dependencies],
            "extra": extra,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "CADGraph":
        """Create a ``CADGraph`` instance from a dictionary produced by
        :meth:`to_dict`.

        This is intentionally forgiving: if a stored ``component_type`` or
        ``dependency_type`` name is not found in the current enums, it falls
        back to ``ComponentType.UNKNOWN`` or a generic dependency type
        (``DependencyType.CONNECTED_TO``). This preserves all original nodes
        and edges without failing, even if enums evolve in the future.
        """

        name = data.get("name", "Unnamed Graph")
        graph = CADGraph(name)

        # Rehydrate components
        for c_data in data.get("components", []):
            c_type_name = c_data.get("component_type", "UNKNOWN")
            try:
                c_type = ComponentType[c_type_name]
            except KeyError:
                c_type = ComponentType.UNKNOWN

            centroid_val = c_data.get("centroid")
            if centroid_val is not None:
                centroid_tuple: Optional[Tuple[float, float, float]] = tuple(centroid_val)  # type: ignore[arg-type]
            else:
                centroid_tuple = None

            component = CADComponent(
                id=c_data.get("id"),
                component_type=c_type,
                name=c_data.get("name"),
                parameters=c_data.get("parameters") or {},
                centroid=centroid_tuple,
                structural_usage=c_data.get("structural_usage"),
            )
            graph.add_component(component)

        # Rehydrate dependencies
        for d_data in data.get("dependencies", []):
            d_type_name = d_data.get("dependency_type", "CONNECTED_TO")
            try:
                d_type = DependencyType[d_type_name]
            except KeyError:
                d_type = DependencyType.CONNECTED_TO

            dependency = CADDependency(
                source_id=d_data.get("source_id"),
                target_id=d_data.get("target_id"),
                dependency_type=d_type,
                weight=float(d_data.get("weight", 1.0)),
                parameters=d_data.get("parameters") or {},
            )
            graph.add_dependency(dependency)

        # Restore any extra attributes without touching core ones.
        extra = data.get("extra", {}) or {}
        for key, value in extra.items():
            # Avoid overriding existing attributes like components/dependencies
            if hasattr(graph, key):
                continue
            setattr(graph, key, value)

        return graph

    def save_to_json(self, path: str, *, indent: int = 2) -> None:
        """Save the graph to a JSON file.

        This is a thin wrapper around :meth:`to_dict` and does not change any
        runtime behaviour; it only adds the option to persist graphs.
        """

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=indent)

    @staticmethod
    def load_from_json(path: str) -> "CADGraph":
        """Load a graph from a JSON file created by :meth:`save_to_json`."""

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return CADGraph.from_dict(data)

    def to_pytorch_geometric(self):
        """
        Converts the CADGraph to a PyTorch Geometric Data object.
        This is the primary input format for our GNN models.
        """
        try:
            import torch
            from torch_geometric.data import Data
        except ImportError as e:
            print(f"❌ Missing required library for GNN conversion: {e}")
            print("Please install PyTorch and PyTorch Geometric.")
            return None

        if not self.components:
            return None

        # Map component IDs to integer indices
        component_ids = list(self.components.keys())
        id_to_idx = {id: i for i, id in enumerate(component_ids)}

        # Node features
        node_features = [self.components[id].get_feature_vector() for id in component_ids]
        x = torch.tensor(np.array(node_features), dtype=torch.float)

        # Edge indices and features
        edge_indices = []
        edge_features = []
        for dep in self.dependencies:
            source_idx = id_to_idx.get(dep.source_id)
            target_idx = id_to_idx.get(dep.target_id)
            if source_idx is not None and target_idx is not None:
                edge_indices.append([source_idx, target_idx])
                edge_features.append(dep.get_feature_vector())

        if not edge_indices:
            # If no edges, create empty tensors with correct dimensions
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, len(DependencyType) + 1), dtype=torch.float) # +1 for weight
        else:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(np.array(edge_features), dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def __repr__(self) -> str:
        return f"CADGraph(name='{self.name}', components={len(self.components)}, dependencies={len(self.dependencies)})"


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("CAD Graph System - Demo")
    print("="*70)
    
    # Create a simple building
    graph = CADGraph("Simple Building")
    
    # Add levels
    graph.levels = {
        "Ground": 0.0,
        "Floor1": 3.5,
        "Floor2": 7.0
    }
    
    # Add walls
    wall1 = CADComponent(
        id="W001",
        component_type=ComponentType.WALL,
        name="Exterior Wall 1",
        parameters={"length": 10.0, "height": 3.5, "thickness": 0.3},
        centroid=(5.0, 0.0, 1.75),
        structural_usage="load_bearing"
    )
    graph.add_component(wall1)
    
    # Add column
    column1 = CADComponent(
        id="C001",
        component_type=ComponentType.COLUMN,
        name="Column 1",
        parameters={"height": 3.5, "width": 0.4, "depth": 0.4},
        centroid=(0.2, 0.2, 1.75),
        structural_usage="load_bearing"
    )
    graph.add_component(column1)
    
    # Add window
    window1 = CADComponent(
        id="WIN001",
        component_type=ComponentType.WINDOW,
        name="Window 1",
        parameters={"width": 1.2, "height": 1.5},
        centroid=(5.0, 0.0, 1.5)
    )
    graph.add_component(window1)
    
    # Add relationships
    rel1 = CADDependency(
        source_id="C001",
        target_id="W001",
        dependency_type=DependencyType.SUPPORTED_BY,
        parameters={"connection_type": "fixed"}
    )
    graph.add_dependency(rel1)
    
    rel2 = CADDependency(
        source_id="WIN001",
        target_id="W001",
        dependency_type=DependencyType.HOSTED_BY,
        weight=0.8
    )
    graph.add_dependency(rel2)
    
    # Print summary
    print(graph)
    
    # Save
    graph.save_to_json("test_building.json")
    
    # Test PyTorch Geometric conversion
    pyg_data = graph.to_pytorch_geometric()
    if pyg_data:
        print(f"\n✅ PyTorch Geometric Data:")
        print(f"   Nodes: {pyg_data.x.shape}")
        print(f"   Edges: {pyg_data.edge_index.shape}")
        print(f"   Edge Attr: {pyg_data.edge_attr.shape}")
