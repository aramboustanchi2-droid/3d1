"""
CAD Graph Builder: Automated Graph Construction from CAD Drawings

This module is responsible for the automated conversion of CAD drawings (DXF/DWG)
into a structured, intelligent graph representation. It serves as the foundational
data processing step for the entire AI analysis pipeline.

Core Functionality:
- Extracts geometric and semantic information from CAD entities.
- Intelligently detects and infers relationships (connectivity, spatial, parametric).
- Constructs a comprehensive CADGraph object, ready for GNNs and other AI systems.
- Features an `IntelligentGraphBuilder` subclass that leverages other AI models
  (e.g., Vision Transformers, GNNs) to enhance graph accuracy and richness,
  embodying a collaborative, multi-expert analysis approach.

Example Usage:
```python
from pathlib import Path
from cad3d.cad_graph_builder import IntelligentGraphBuilder

# Use the intelligent builder which can leverage other AI models
builder = IntelligentGraphBuilder(proximity_threshold=50.0)

# Build the graph from a DXF file
cad_graph = builder.build_from_dxf(Path("my_floor_plan.dxf"))

# The graph is now ready for analysis, GNN processing, or export
cad_graph.print_summary()
```
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional, Any
import numpy as np
import logging
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

try:
    import ezdxf
    from ezdxf.document import Drawing
    from ezdxf.entitydb import EntityDB
    EZDXF_AVAILABLE = True
except ImportError:
    EZDXF_AVAILABLE = False
    Drawing = None
    EntityDB = None
    logging.warning("ezdxf library not found. DXF processing will be unavailable. Please run 'pip install ezdxf'.")

from .cad_graph import (
    CADGraph, CADElement, CADRelationship,
    ElementType, RelationType
)


class CADGraphBuilder:
    """
    Automated graph constructor for CAD drawings.

    This class provides the core logic for parsing CAD files, extracting elements,
    and inferring their relationships to build a structured `CADGraph`.
    """
    
    def __init__(
        self,
        proximity_threshold: float = 0.1,
        parallel_angle_threshold: float = 5.0,  # degrees
        perpendicular_angle_threshold: float = 5.0
    ):
        """
        Initializes the graph builder.

        Args:
            proximity_threshold: Maximum distance (in drawing units, e.g., mm) to
                                 consider two elements adjacent.
            parallel_angle_threshold: Angle tolerance (in degrees) for detecting
                                      parallel relationships.
            perpendicular_angle_threshold: Angle tolerance from 90 degrees for
                                           detecting perpendicular relationships.
        """
        self.proximity_threshold = proximity_threshold
        self.parallel_angle_threshold = parallel_angle_threshold
        self.perpendicular_angle_threshold = perpendicular_angle_threshold
        
        # Maps DXF entity types to our internal `ElementType` enum
        self.entity_type_mapping = {
            'LINE': ElementType.LINE,
            'POLYLINE': ElementType.POLYLINE,
            'LWPOLYLINE': ElementType.POLYLINE,
            'CIRCLE': ElementType.CIRCLE,
            'ARC': ElementType.ARC,
            'TEXT': ElementType.TEXT,
            'MTEXT': ElementType.TEXT,
            'DIMENSION': ElementType.DIMENSION,
            'LEADER': ElementType.LEADER,
            'POINT': ElementType.POINT,
            'SPLINE': ElementType.SPLINE,
            '3DFACE': ElementType.MESH,
            'MESH': ElementType.MESH,
            'SOLID': ElementType.SOLID
        }
    
    def build_from_dxf(self, dxf_path: Path) -> CADGraph:
        """
        Constructs a CADGraph from a DXF file.

        This is the main entry point for the building process. It orchestrates
        element extraction, relationship detection, and final graph assembly.

        Args:
            dxf_path: The path to the input DXF file.
        
        Returns:
            A fully constructed `CADGraph` object representing the drawing.
        
        Raises:
            ImportError: If the `ezdxf` library is not installed.
            FileNotFoundError: If the specified DXF file does not exist.
        """
        if not EZDXF_AVAILABLE:
            raise ImportError("ezdxf is required to build graphs from DXF files. Please run 'pip install ezdxf'.")
        
        if not dxf_path.exists():
            raise FileNotFoundError(f"The specified DXF file does not exist: {dxf_path}")

        logging.info(f"Loading DXF file: {dxf_path}")
        try:
            doc = ezdxf.readfile(dxf_path)
            msp = doc.modelspace()
        except IOError as e:
            logging.error(f"Could not read DXF file: {e}")
            raise
        
        graph = CADGraph(name=dxf_path.stem)
        
        self._extract_levels_from_layers(doc, graph)
        
        logging.info("Extracting CAD elements from modelspace...")
        elements = [elem for entity in msp if (elem := self._entity_to_cad_element(entity)) is not None]
        for element in elements:
            graph.add_element(element)
        logging.info(f"Successfully extracted {len(elements)} elements.")
        
        logging.info("Detecting relationships between elements...")
        self._detect_spatial_relationships(graph)
        self._detect_connectivity_relationships(graph)
        self._detect_parametric_relationships(graph)
        logging.info(f"Detected a total of {len(graph.relationships)} relationships.")
        
        logging.info("Computing final geometry properties (e.g., bounding boxes)...")
        for element in graph.elements.values():
            element.compute_bounding_box()
        
        logging.info("Graph construction complete.")
        return graph
    
    def _extract_levels_from_layers(self, doc: Drawing, graph: CADGraph):
        """
        Extracts level/floor information from layer names using common heuristics.
        For example, a layer named "A-WALL-FLR01" would be mapped to a level.
        """
        # Heuristics for common level naming conventions
        level_patterns = {
            'Ground': ['GROUND', 'LEVEL_0', 'FLOOR_0', 'FLR00'],
            'Floor1': ['LEVEL_1', 'FLOOR_1', 'FLR01'],
            'Floor2': ['LEVEL_2', 'FLOOR_2', 'FLR02'],
        }
        
        for layer in doc.layers:
            layer_name = layer.dxf.name.upper()
            for level_name, patterns in level_patterns.items():
                if any(p in layer_name for p in patterns):
                    # Assign a default elevation if not already present
                    if level_name not in graph.levels:
                        graph.levels[level_name] = 0.0 if 'Ground' in level_name else 3.5 * int(level_name.replace('Floor', ''))
        
        if not graph.levels:
            graph.levels['Default'] = 0.0
            logging.info("No level information found in layers. Using a single 'Default' level.")
    
    def _entity_to_cad_element(self, entity: Any) -> Optional[CADElement]:
        """
        Converts a single `ezdxf` entity into a `CADElement` node.
        """
        entity_type_str = entity.dxftype()
        element_type = self.entity_type_mapping.get(entity_type_str)
        if not element_type:
            return None
        
        element_id = f"{entity_type_str}_{entity.dxf.handle}"
        
        try:
            geometry = self._extract_geometry(entity)
            if not geometry:
                return None
        except Exception as e:
            logging.warning(f"Could not extract geometry for entity {element_id}: {e}")
            return None
        
        properties = {
            'dxf_type': entity_type_str,
            'handle': entity.dxf.handle
        }
        
        layer = getattr(entity.dxf, 'layer', '0')
        color_val = getattr(entity.dxf, 'color', 256) # 256 means bylayer
        color = ezdxf.colors.aci2rgb(color_val) if color_val != 256 else None

        return CADElement(
            id=element_id,
            element_type=element_type,
            name=element_id,
            geometry=geometry,
            properties=properties,
            layer=layer,
            color=color
        )
    
    def _extract_geometry(self, entity: Any) -> Optional[Dict[str, Any]]:
        """Extracts geometric properties from an ezdxf entity."""
        entity_type = entity.dxftype()
        geometry = {}
        
        if entity_type == 'LINE':
            start, end = list(entity.dxf.start), list(entity.dxf.end)
            geometry = {
                'start': start,
                'end': end,
                'length': np.linalg.norm(np.array(end) - np.array(start))
            }
        elif entity_type == 'CIRCLE':
            geometry = {
                'center': list(entity.dxf.center),
                'radius': entity.dxf.radius,
                'area': np.pi * entity.dxf.radius ** 2
            }
        elif entity_type == 'ARC':
            geometry = {
                'center': list(entity.dxf.center),
                'radius': entity.dxf.radius,
                'start_angle': entity.dxf.start_angle,
                'end_angle': entity.dxf.end_angle
            }
        elif entity_type in ['POLYLINE', 'LWPOLYLINE']:
            points = [list(p)[:2] for p in entity.get_points('xy')] # Ensure 2D points for simplicity
            if points:
                geometry = {
                    'points': points,
                    'is_closed': entity.is_closed,
                    'num_points': len(points)
                }
        elif entity_type == 'TEXT':
            geometry = {
                'insert': list(entity.dxf.insert),
                'text': entity.dxf.text,
                'height': entity.dxf.height
            }
        elif entity_type == 'MTEXT':
            geometry = {
                'insert': list(entity.dxf.insert),
                'text': entity.text,
                'char_height': getattr(entity.dxf, 'char_height', 1.0)
            }
        
        return geometry or None
    
    def _detect_spatial_relationships(self, graph: CADGraph):
        """Detects spatial relationships like adjacency, above/below."""
        elements = [e for e in graph.elements.values() if e.centroid]
        
        for i, elem1 in enumerate(elements):
            for elem2 in elements[i+1:]:
                dist = self._compute_distance(elem1.centroid, elem2.centroid)
                
                if dist <= self.proximity_threshold:
                    graph.add_relationship(CADRelationship(
                        from_element=elem1.id, to_element=elem2.id,
                        relation_type=RelationType.ADJACENT_TO,
                        weight=1.0 / (dist + 1e-6)
                    ))
                
                # Check for Above/Below relationship based on Z-axis alignment
                if abs(elem1.centroid[0] - elem2.centroid[0]) < self.proximity_threshold and \
                   abs(elem1.centroid[1] - elem2.centroid[1]) < self.proximity_threshold:
                    if elem1.centroid[2] > elem2.centroid[2]:
                        graph.add_relationship(CADRelationship(
                            from_element=elem1.id, to_element=elem2.id,
                            relation_type=RelationType.ABOVE
                        ))
                    elif elem2.centroid[2] > elem1.centroid[2]:
                        graph.add_relationship(CADRelationship(
                            from_element=elem2.id, to_element=elem1.id,
                            relation_type=RelationType.ABOVE
                        ))

    def _detect_connectivity_relationships(self, graph: CADGraph):
        """Detects physical connectivity, such as lines meeting at endpoints."""
        lines = graph.get_elements_by_type(ElementType.LINE)
        
        for i, line1 in enumerate(lines):
            for line2 in lines[i+1:]:
                if self._lines_connected(line1, line2):
                    graph.add_relationship(CADRelationship(
                        from_element=line1.id, to_element=line2.id,
                        relation_type=RelationType.CONNECTED_TO
                    ))
    
    def _detect_parametric_relationships(self, graph: CADGraph):
        """
        Detects parametric dependencies, such as a window being hosted by a wall.
        This is a simplified heuristic based on bounding box containment.
        """
        elements = [e for e in graph.elements.values() if e.bounding_box]
        
        for elem1 in elements:
            for elem2 in elements:
                if elem1.id == elem2.id:
                    continue
                
                # If elem2's bounding box is inside elem1's, it might be hosted by it.
                if self._is_inside_bounding_box(elem2.bounding_box, elem1.bounding_box):
                    # Avoid relating two very large objects
                    vol1 = self._bbox_volume(elem1.bounding_box)
                    vol2 = self._bbox_volume(elem2.bounding_box)
                    if vol1 > vol2 * 1.5: # Ensure host is significantly larger
                        graph.add_relationship(CADRelationship(
                            from_element=elem2.id, to_element=elem1.id,
                            relation_type=RelationType.HOSTED_BY,
                            is_parametric=True
                        ))
    
    @staticmethod
    def _compute_distance(pt1: Tuple[float, ...], pt2: Tuple[float, ...]) -> float:
        """Computes the Euclidean distance between two points."""
        return np.linalg.norm(np.array(pt1) - np.array(pt2))
    
    def _lines_connected(self, line1: CADElement, line2: CADElement, tolerance: float = 1e-3) -> bool:
        """Checks if two line elements are connected at their endpoints."""
        p1_start = np.array(line1.geometry.get('start', [0,0,0]))
        p1_end = np.array(line1.geometry.get('end', [0,0,0]))
        p2_start = np.array(line2.geometry.get('start', [0,0,0]))
        p2_end = np.array(line2.geometry.get('end', [0,0,0]))
        
        return any(
            np.linalg.norm(p1 - p2) < tolerance
            for p1 in [p1_start, p1_end]
            for p2 in [p2_start, p2_end]
        )
    
    @staticmethod
    def _is_inside_bounding_box(inner_bbox: Tuple, outer_bbox: Tuple) -> bool:
        """Checks if the inner bounding box is fully contained within the outer one."""
        inner_min, inner_max = np.array(inner_bbox[0]), np.array(inner_bbox[1])
        outer_min, outer_max = np.array(outer_bbox[0]), np.array(outer_bbox[1])
        return np.all(inner_min >= outer_min) and np.all(inner_max <= outer_max)

    @staticmethod
    def _bbox_volume(bbox: Tuple) -> float:
        """Calculates the volume of a bounding box."""
        min_pt, max_pt = np.array(bbox[0]), np.array(bbox[1])
        dims = max_pt - min_pt
        return float(np.prod(dims))


class IntelligentGraphBuilder(CADGraphBuilder):
    """
    An advanced graph builder that leverages AI models to enhance graph quality.

    This class represents the collaborative AI approach, where different expert
    models (e.g., for classification, relationship prediction) work together to
    produce a more accurate and semantically rich graph.
    """
    
    def __init__(
        self,
        element_classifier_path: Optional[Path] = None,
        relationship_predictor_path: Optional[Path] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.element_classifier = None
        self.relationship_predictor = None
        
        if element_classifier_path:
            self._load_element_classifier(element_classifier_path)
        if relationship_predictor_path:
            self._load_relationship_predictor(relationship_predictor_path)
    
    def _load_element_classifier(self, model_path: Path):
        """Loads a pre-trained element classification model (e.g., a ViT or CNN)."""
        logging.info(f"Attempting to load element classifier from: {model_path}")
        try:
            import torch
            from .vision_transformer_cad import VisionTransformerCAD, VisionTransformerConfig
            
            config = VisionTransformerConfig() # Use default config
            self.element_classifier = VisionTransformerCAD(config)
            
            checkpoint = torch.load(model_path, map_location='cpu')
            self.element_classifier.load_state_dict(checkpoint['model_state_dict'])
            self.element_classifier.eval()
            logging.info("Successfully loaded AI element classifier.")
        except Exception as e:
            logging.error(f"Failed to load element classifier model: {e}", exc_info=True)
            self.element_classifier = None
    
    def _load_relationship_predictor(self, model_path: Path):
        """Loads a pre-trained GNN model for relationship prediction."""
        logging.info(f"Attempting to load relationship predictor from: {model_path}")
        try:
            import torch
            from .cad_gnn import CADRelationshipPredictor
            
            self.relationship_predictor = CADRelationshipPredictor()
            checkpoint = torch.load(model_path, map_location='cpu')
            self.relationship_predictor.load_state_dict(checkpoint['model_state_dict'])
            self.relationship_predictor.eval()
            logging.info("Successfully loaded AI relationship predictor.")
        except Exception as e:
            logging.error(f"Failed to load relationship predictor model: {e}", exc_info=True)
            self.relationship_predictor = None
    
    def classify_element_with_ai(self, element: CADElement, image_region: np.ndarray) -> ElementType:
        """
        Uses a vision model to refine the classification of a CAD element.

        Args:
            element: The CAD element to classify.
            image_region: An image patch corresponding to the element.
        
        Returns:
            The refined `ElementType` predicted by the AI model.
        """
        if not self.element_classifier:
            return element.element_type
        
        logging.debug(f"Running AI classification for element {element.id}...")
        try:
            import torch
            # Placeholder for image preprocessing and inference
            # image_tensor = preprocess(image_region).unsqueeze(0)
            # with torch.no_grad():
            #     outputs = self.element_classifier(image_tensor)
            #     predicted_class_idx = outputs['semantic'].argmax().item()
            # return self.map_class_idx_to_element_type(predicted_class_idx)
            return element.element_type # Return original as placeholder
        except Exception as e:
            logging.warning(f"AI element classification failed for {element.id}: {e}")
            return element.element_type
    
    def predict_relationships_with_gnn(self, graph: CADGraph):
        """
        Uses a GNN to infer additional or more accurate relationships.
        This represents a powerful collaborative step where the graph structure
        itself is used to refine its own connections.
        """
        if not self.relationship_predictor:
            return
        
        logging.info("Enhancing graph with GNN-based relationship prediction...")
        try:
            import torch
            
            data = graph.to_pytorch_geometric()
            if data is None:
                logging.error("Could not convert graph to PyTorch Geometric format for GNN prediction.")
                return

            # This requires a node encoder model, which is not defined here.
            # This is a conceptual placeholder for a full pipeline.
            # node_embeddings = self.node_encoder(data.x) 
            # relation_logits = self.relationship_predictor(node_embeddings, candidate_edges)
            # ... process logits to add new relationships ...
            logging.warning("GNN relationship prediction logic is a placeholder and was not executed.")
            
        except Exception as e:
            logging.error(f"GNN relationship prediction failed: {e}", exc_info=True)


if __name__ == "__main__":
    logging.info("Running CAD Graph Builder Demo")
    
    if not EZDXF_AVAILABLE:
        logging.error("ezdxf is not available. Cannot run demo. Please install with 'pip install ezdxf'.")
    else:
        builder = IntelligentGraphBuilder(proximity_threshold=100.0)
        
        logging.info("Creating a sample DXF file for demonstration...")
        doc = ezdxf.new('R2010')
        msp = doc.modelspace()
        
        # A simple square of lines
        msp.add_line((0, 0), (1000, 0))
        msp.add_line((1000, 0), (1000, 1000))
        msp.add_line((1000, 1000), (0, 1000))
        msp.add_line((0, 1000), (0, 0))
        
        # A circle inside
        msp.add_circle((500, 500), 100)
        
        dxf_path = Path("demo_graph_builder.dxf")
        try:
            doc.saveas(dxf_path)
            logging.info(f"Sample DXF saved to {dxf_path}")
            
            graph = builder.build_from_dxf(dxf_path)
            graph.print_summary()
            
            graph_json_path = Path("demo_graph_builder_output.json")
            graph.save_to_json(graph_json_path)
            
        finally:
            if dxf_path.exists():
                dxf_path.unlink()
                logging.info(f"Cleaned up sample file: {dxf_path}")
