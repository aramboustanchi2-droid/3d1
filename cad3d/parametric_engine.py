from __future__ import annotations
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re
import math
from pathlib import Path
import json

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from .cad_graph import CADGraph, CADElement


class ExpressionType(Enum):
    """Defines the type of a parametric expression."""
    DIRECT = "direct"        # The value is a direct constant.
    REFERENCE = "reference"  # The value is a reference to another property.
    FORMULA = "formula"      # The value is derived from a formula.
    CONSTRAINT = "constraint" # The expression represents a geometric constraint.


class ConstraintType(Enum):
    """Defines the type of a geometric constraint between elements."""
    PARALLEL = "parallel"
    PERPENDICULAR = "perpendicular"
    COINCIDENT = "coincident"
    DISTANCE = "distance"
    ANGLE = "angle"
    TANGENT = "tangent"
    CONCENTRIC = "concentric"


@dataclass
class ParametricExpression:
    """
    Represents a parametric relationship that defines a property of a CAD element.

    For example, this can define a window's width as a fraction of a wall's width:
    `window.width = wall.width * 0.3`

    Attributes:
        target_element: The ID of the element whose property is being defined.
        target_property: The name of the property being defined (e.g., "width").
        expression: The string formula to be evaluated (e.g., "wall_001.width * 0.3").
        expression_type: The type of the expression.
        dependencies: A set of element IDs that this expression depends on.
    """
    target_element: str
    target_property: str
    expression: str
    expression_type: ExpressionType = ExpressionType.FORMULA
    dependencies: Set[str] = field(default_factory=set)

    def __post_init__(self):
        """Automatically extracts dependencies from the expression string after initialization."""
        if not self.dependencies:
            self.dependencies = self._extract_dependencies()

    def _extract_dependencies(self) -> Set[str]:
        """
        Parses the expression string to find all referenced element IDs.

        For example, an expression "wall.width * 0.3 + column.height" would
        identify "wall" and "column" as dependencies.
        """
        # This regex finds all occurrences of `element_id.property_name`.
        pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)'
        matches = re.findall(pattern, self.expression)
        return {elem_id for elem_id, _ in matches}


@dataclass
class GeometricConstraint:
    """
    Represents a geometric constraint between two or more CAD elements.

    Attributes:
        elements: A list of element IDs involved in the constraint.
        constraint_type: The type of geometric constraint.
        value: An optional numeric value for the constraint (e.g., for distance or angle).
    """
    elements: List[str]
    constraint_type: ConstraintType
    value: Optional[float] = None

    def evaluate(self, graph: CADGraph) -> Tuple[bool, Optional[str]]:
        """
        Checks if the constraint is currently satisfied by the elements in the graph.

        Args:
            graph: The CADGraph containing the elements.

        Returns:
            A tuple of (is_satisfied, error_message).
        """
        if len(self.elements) < 2:
            return False, "Constraint requires at least two elements."

        elem1 = graph.get_element(self.elements[0])
        elem2 = graph.get_element(self.elements[1])

        if not elem1 or not elem2:
            return False, "One or more elements in the constraint not found."

        # Dispatch to the appropriate check method based on constraint type.
        check_methods = {
            ConstraintType.DISTANCE: self._check_distance,
            # Add other constraint checks here
        }
        
        check_method = check_methods.get(self.constraint_type)
        if check_method:
            return check_method(elem1, elem2)

        return True, None  # Assume unhandled constraints are satisfied.

    def _check_distance(self, elem1: CADElement, elem2: CADElement) -> Tuple[bool, Optional[str]]:
        """Checks if the distance between two elements matches the constraint value."""
        if self.value is None:
            return False, "Distance constraint requires a value."
        if not elem1.centroid or not elem2.centroid:
            return False, "Elements must have centroids to check distance."
        if not NUMPY_AVAILABLE:
            return True, "NumPy not available, cannot check distance."

        dist = np.linalg.norm(np.array(elem1.centroid) - np.array(elem2.centroid))
        
        # Use a small tolerance for floating-point comparison.
        tolerance = 1e-6
        if not math.isclose(dist, self.value, rel_tol=tolerance, abs_tol=tolerance):
            return False, f"Distance violation: is {dist:.2f}, should be {self.value:.2f}"
        
        return True, None


class ParametricEngine:
    """
    Manages and propagates parametric changes within a CADGraph.

    This engine allows for creating Revit-like parametric behaviors where changes
    to one element automatically update dependent elements according to defined
    expressions and constraints.
    """
    def __init__(self, graph: CADGraph):
        self.graph = graph
        self.expressions: Dict[Tuple[str, str], ParametricExpression] = {}
        self.constraints: List[GeometricConstraint] = []
        # The dependency graph maps an element ID to a set of other element IDs that depend on it.
        self._dependency_graph: Dict[str, Set[str]] = {}

    def add_expression(self, target_element: str, target_property: str, expression: str):
        """Adds a new parametric expression to the engine."""
        param_expr = ParametricExpression(
            target_element=target_element,
            target_property=target_property,
            expression=expression,
        )
        self.expressions[(target_element, target_property)] = param_expr

        # Update the dependency graph.
        for dep_id in param_expr.dependencies:
            self._dependency_graph.setdefault(dep_id, set()).add(target_element)

    def add_constraint(self, elements: List[str], constraint_type: ConstraintType, value: Optional[float] = None):
        """Adds a new geometric constraint to the engine."""
        constraint = GeometricConstraint(elements, constraint_type, value)
        self.constraints.append(constraint)

    def update_parameter(self, element_id: str, property_name: str, new_value: Any) -> Dict[str, Any]:
        """
        Updates a parameter on an element and propagates the changes to all dependent elements.

        Args:
            element_id: The ID of the element to update.
            property_name: The name of the property to change.
            new_value: The new value for the property.

        Returns:
            A dictionary summarizing the updates and any constraint violations.
        """
        element = self.graph.get_element(element_id)
        if not element:
            return {'error': f"Element '{element_id}' not found."}

        old_value = element.properties.get(property_name)
        element.properties[property_name] = new_value

        result = {
            'element_id': element_id,
            'property': property_name,
            'old_value': old_value,
            'new_value': new_value,
            'updated_elements': [],
        }

        # Use a topological sort of the dependency graph to update elements in the correct order.
        try:
            update_order = self._get_topological_sort(element_id)
        except ValueError as e: # Cycle detected
            result['error'] = str(e)
            return result

        for elem_to_update_id in update_order:
            # Find all expressions that target this element.
            for (target_elem, target_prop), expr in self.expressions.items():
                if target_elem == elem_to_update_id:
                    new_val = self._evaluate_expression(expr)
                    if new_val is not None:
                        dep_element = self.graph.get_element(target_elem)
                        if dep_element:
                            old_val = dep_element.properties.get(target_prop)
                            dep_element.properties[target_prop] = new_val
                            result['updated_elements'].append({
                                'element_id': target_elem, 'property': target_prop,
                                'old_value': old_val, 'new_value': new_val,
                            })
        
        # After all updates, check for constraint violations.
        violated_constraints = self.check_all_constraints()
        if violated_constraints:
            result['constraint_violations'] = violated_constraints

        return result

    def _evaluate_expression(self, expr: ParametricExpression) -> Optional[Any]:
        """
        Evaluates a parametric expression string.

        This method builds a safe execution context containing the properties of
        dependent elements and common math functions, then evaluates the expression.
        """
        try:
            context = {'math': math, 'abs': abs, 'min': min, 'max': max}
            for dep_id in expr.dependencies:
                element = self.graph.get_element(dep_id)
                if not element:
                    raise NameError(f"Dependency '{dep_id}' not found in graph.")
                
                # Create a simple object-like structure for the context.
                elem_props = element.properties.copy()
                context[dep_id] = type('ElementContext', (), elem_props)

            # Evaluate the expression in the sandboxed context.
            return eval(expr.expression, {"__builtins__": {}}, context)
        except Exception as e:
            print(f"⚠️  Could not evaluate expression '{expr.expression}': {e}")
            return None

    def check_all_constraints(self) -> List[Dict[str, Any]]:
        """Checks all registered geometric constraints and returns a list of violations."""
        violations = []
        for constraint in self.constraints:
            is_satisfied, error = constraint.evaluate(self.graph)
            if not is_satisfied:
                violations.append({
                    'elements': constraint.elements,
                    'type': constraint.constraint_type.value,
                    'reason': error,
                })
        return violations

    def _get_topological_sort(self, start_node: str) -> List[str]:
        """
        Performs a topological sort of the dependency graph starting from a given node.
        This determines the correct order for propagating updates.

        Args:
            start_node: The ID of the element that was initially changed.

        Returns:
            A list of element IDs in the order they should be updated.
            
        Raises:
            ValueError: If a dependency cycle is detected.
        """
        visited = set()
        recursion_stack = set()
        sorted_order = []

        def dfs(node):
            visited.add(node)
            recursion_stack.add(node)

            for neighbor in self._dependency_graph.get(node, []):
                if neighbor not in visited:
                    dfs(neighbor)
                elif neighbor in recursion_stack:
                    raise ValueError(f"Dependency cycle detected involving '{neighbor}'")
            
            recursion_stack.remove(node)
            sorted_order.append(node)

        # Build the full dependency chain starting from the modified node.
        nodes_to_visit = {start_node}
        queue = [start_node]
        while queue:
            node = queue.pop(0)
            for dependent in self._dependency_graph.get(node, []):
                if dependent not in nodes_to_visit:
                    nodes_to_visit.add(dependent)
                    queue.append(dependent)

        for node in nodes_to_visit:
            if node not in visited:
                dfs(node)
        
        return sorted_order

    def export_to_json(self, path: Path):
        """Saves the parametric setup (expressions and constraints) to a JSON file."""
        data = {
            'expressions': [
                {
                    'target': f"{expr.target_element}.{expr.target_property}",
                    'expression': expr.expression,
                    'type': expr.expression_type.value,
                } for expr in self.expressions.values()
            ],
            'constraints': [
                {
                    'elements': c.elements,
                    'type': c.constraint_type.value,
                    'value': c.value,
                } for c in self.constraints
            ]
        }
        with path.open('w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"💾 Parametric data saved to {path}")


def demo_parametric_system():
    """An example demonstrating the use of the ParametricEngine."""
    from .cad_graph import CADGraph, CADElement, ElementType
    
    print("="*70)
    print("Parametric System Demo")
    print("="*70)
    
    graph = CADGraph("Building Plan")
    
    # Add elements to the graph
    graph.add_element(CADElement(id="wall_001", element_type=ElementType.WALL, centroid=(5000, 0, 1500), properties={'width': 10000, 'height': 3000}))
    graph.add_element(CADElement(id="window_001", element_type=ElementType.WINDOW, centroid=(5000, 0, 1500), properties={'width': 3000, 'height': 1500}))
    graph.add_element(CADElement(id="door_001", element_type=ElementType.DOOR, centroid=(2000, 0, 1100), properties={'width': 900, 'height': 2200}))

    engine = ParametricEngine(graph)

    print("\n📐 Defining parametric relationships...")
    engine.add_expression("window_001", "width", "wall_001.width * 0.3")
    engine.add_expression("door_001", "width", "wall_001.width * 0.1") # Doors are 10% of wall width

    print("\n🔗 Adding geometric constraints...")
    engine.add_constraint(["window_001", "door_001"], ConstraintType.DISTANCE, value=3000)

    print("\n🔄 Updating wall width from 10m to 15m...")
    result = engine.update_parameter("wall_001", "width", 15000)

    print("\n📊 Results:")
    print(f"  Wall width: {graph.get_element('wall_001').properties['width']} mm")
    print(f"  Window width: {graph.get_element('window_001').properties['width']} mm")
    print(f"  Door width: {graph.get_element('door_001').properties['width']} mm")

    if 'constraint_violations' in result:
        print("\n⚠️ Constraint Violations:")
        for v in result['constraint_violations']:
            print(f"  - {v['elements']} ({v['type']}): {v['reason']}")

    engine.export_to_json(Path("parametric_demo.json"))
    print("\n✅ Demo complete!")


if __name__ == "__main__":
    demo_parametric_system()
