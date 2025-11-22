# راهنمای جامع سیستم گراف (Graph-Based System)

# Complete Guide to Graph-Based CAD System

## 📚 فهرست / Table of Contents

1. [معرفی / Introduction](#introduction)
2. [معماری سیستم / Architecture](#architecture)
3. [مفاهیم کلیدی / Key Concepts](#concepts)
4. [استفاده / Usage](#usage)
5. [مثال‌های کاربردی / Examples](#examples)
6. [API Reference](#api)
7. [پیشرفته / Advanced Topics](#advanced)

---

## 1. معرفی / Introduction <a name="introduction"></a>

### چرا سیستم گراف؟ / Why Graph-Based System?

سیستم گراف به شما این امکان را می‌دهد که:

- **روابط** بین عناصر را درک کنید (کدام دیوار به کدام ستون متصل است)
- **تحلیل ساختاری** انجام دهید (محاسبه نیرو، تنش، کرنش)
- **parametric updates** داشته باشید (مثل Revit - تغییر یک عنصر بقیه را update می‌کند)
- **3D دقیق‌تر** تولید کنید (با درک کامل از ساختار)

The Graph-Based System allows you to:

- **Understand relationships** between elements (which wall connects to which column)
- **Perform structural analysis** (calculate forces, stress, strain)
- **Enable parametric updates** (like Revit - changing one element updates others)
- **Generate more accurate 3D** (with complete understanding of structure)

### نرم‌افزارهای مشابه / Similar Software

این سیستم الهام‌گرفته از:

- **Autodesk Revit** - BIM و parametric modeling
- **Autodesk Civil 3D** - مهندسی عمران
- **Grasshopper** - parametric design
- **Tekla Structures** - تحلیل ساختاری

---

## 2. معماری سیستم / Architecture <a name="architecture"></a>

### نمای کلی / Overview

```
                     📐 DXF/DWG File
                            |
                            v
         🔨 CADGraphBuilder (Automatic Extraction)
                            |
                            v
                  📊 CAD Graph Structure
                  (Nodes + Edges + Properties)
                            |
            +---------------+---------------+
            |               |               |
            v               v               v
    🧠 GNN Analysis   🔗 Relationships  📈 Structural Analysis
       (GAT/GCN)       (Parametric)      (Load/Stress)
            |               |               |
            +---------------+---------------+
                            |
                            v
              🎨 3D Generation (VAE/Diffusion)
                            |
                            v
                     🏗️ 3D Model Output
```

### اجزای اصلی / Main Components

1. **CADGraph** (`cad_graph.py`):
   - Core graph representation
   - Node = CAD element
   - Edge = Relationship
   - Properties: geometry, parameters, metadata

2. **CADGraphBuilder** (`cad_graph_builder.py`):
   - Automatic graph construction from DXF
   - Spatial relationship detection
   - Connectivity analysis
   - Parametric relationship detection

3. **GNN Models** (`cad_gnn.py`):
   - CAD_GCN: Basic graph convolution
   - CAD_GAT: Graph attention (focuses on important connections)
   - EdgeConditionedGNN: Message passing with edge features
   - CADDepthReconstructionGNN: 3D reconstruction from 2D graph

4. **GraphEnhancedConverter** (`graph_enhanced_converter.py`):
   - Unified pipeline: DXF → Graph → GNN → 3D
   - Integration with VAE/Diffusion/ViT
   - Complete 2D→3D conversion

---

## 3. مفاهیم کلیدی / Key Concepts <a name="concepts"></a>

### 3.1 Element Types (انواع عناصر)

سیستم از 50+ نوع عنصر پشتیبانی می‌کند:

#### Architectural (معماری)

- `WALL` - دیوار
- `COLUMN` - ستون
- `BEAM` - تیر
- `SLAB` - دال
- `DOOR` - در
- `WINDOW` - پنجره
- `ROOF` - سقف
- `STAIR` - پله
- `RAILING` - نرده

#### Structural (سازه)

- `FOUNDATION` - پی
- `FOOTING` - پایه
- `TRUSS` - خرپا
- `BRACE` - مهاربند
- `SHEAR_WALL` - دیوار برشی

#### MEP (تاسیسات)

- `PIPE` - لوله
- `DUCT` - کانال
- `CABLE` - کابل
- `VALVE` - شیر
- `PUMP` - پمپ
- `FAN` - فن

#### Civil (عمران)

- `ROAD` - جاده
- `BRIDGE` - پل
- `TUNNEL` - تونل
- `DAM` - سد
- `RAILWAY` - راه‌آهن
- `CANAL` - کانال

#### Mechanical (مکانیک)

- `GEAR` - چرخ‌دنده
- `SHAFT` - محور
- `BEARING` - بلبرینگ
- `SPRING` - فنر

### 3.2 Relationship Types (انواع روابط)

#### Connectivity (اتصال)

- `CONNECTED_TO` - متصل به
- `SUPPORTED_BY` - تکیه‌گاه
- `HOSTED_BY` - میزبان
- `FIXED_CONNECTION` - اتصال ثابت
- `HINGED_CONNECTION` - اتصال مفصلی

#### Spatial (فضایی)

- `ABOVE` - بالای
- `BELOW` - زیر
- `ADJACENT` - مجاور
- `INSIDE` - داخل
- `OUTSIDE` - خارج

#### Parametric (پارامتریک)

- `DEPENDS_ON` - وابسته به
- `DRIVES` - محرک
- `CONSTRAINED_BY` - محدود‌شده توسط

#### Structural (ساختاری)

- `LOAD_TRANSFER` - انتقال بار
- `MOMENT_CONNECTION` - اتصال ممان
- `SHEAR_CONNECTION` - اتصال برشی

### 3.3 Properties (ویژگی‌ها)

هر عنصر دارای:

- **geometry**: مشخصات هندسی (نقاط، خطوط، ...)
- **properties**: ویژگی‌های فیزیکی (مواد، ابعاد، ...)
- **parameters**: پارامترهای قابل تنظیم
- **bounding_box**: محدوده فضایی
- **metadata**: اطلاعات اضافی (لایه، رنگ، ...)

---

## 4. استفاده / Usage <a name="usage"></a>

### 4.1 ساخت گراف از DXF / Building Graph from DXF

```python
from pathlib import Path
from cad3d.cad_graph_builder import CADGraphBuilder

# Create builder
builder = CADGraphBuilder(
    proximity_threshold=100.0,  # 100mm for adjacency detection
    parallel_angle_threshold=5.0  # 5° for parallel detection
)

# Build graph from DXF
graph = builder.build_from_dxf(Path("plan.dxf"))

# View statistics
stats = graph.get_statistics()
print(f"Elements: {stats['total_elements']}")
print(f"Relationships: {stats['total_relationships']}")
```

### 4.2 جستجوی عناصر / Querying Elements

```python
# Find all walls
walls = graph.get_elements_by_type(ElementType.WALL)
print(f"Found {len(walls)} walls")

# Find elements on specific layer
layer_elements = graph.get_elements_by_layer("WALLS")

# Spatial query: elements in bounding box
bbox = ((0, 0, 0), (1000, 1000, 500))  # x, y, z range in mm
elements_in_box = graph.get_elements_in_bbox(bbox)
```

### 4.3 تحلیل روابط / Analyzing Relationships

```python
# Get all relationships for an element
elem_id = "wall_001"
relationships = graph.get_element_relationships(elem_id)

for rel in relationships:
    print(f"{rel.source_id} --{rel.relation_type.name}--> {rel.target_id}")

# Example output:
# wall_001 --CONNECTED_TO--> column_005
# wall_001 --SUPPORTED_BY--> foundation_010
# window_002 --HOSTED_BY--> wall_001
```

### 4.4 تبدیل به 3D / Converting to 3D

```python
from cad3d.graph_enhanced_converter import GraphEnhancedCAD3DConverter

# Create converter
converter = GraphEnhancedCAD3DConverter(
    device='cuda',  # or 'cpu'
    vae_model_path=Path("checkpoints/vae_best.pth"),
    vit_model_path=Path("checkpoints/vit_best.pth")
)

# Convert DXF to 3D
result = converter.convert_dxf_to_3d(
    dxf_path=Path("plan_2d.dxf"),
    output_dxf=Path("plan_3d.dxf"),
    image_path=Path("plan_image.png"),  # optional
    normalize_range=(0, 1000)  # output range in mm
)

print(f"Generated {result['num_points']} 3D points")
```

---

## 5. مثال‌های کاربردی / Examples <a name="examples"></a>

### Example 1: Building (ساختمان)

```python
from cad3d.cad_graph import CADGraph, CADElement, ElementType, RelationType

# Create graph
graph = CADGraph()

# Add foundation
foundation = CADElement(
    id="foundation_001",
    element_type=ElementType.FOUNDATION,
    centroid=(0, 0, -500),  # 500mm below ground
    properties={
        'width': 2000,  # mm
        'length': 10000,
        'depth': 500,
        'material': 'concrete_C25'
    }
)
graph.add_element(foundation)

# Add column
column = CADElement(
    id="column_001",
    element_type=ElementType.COLUMN,
    centroid=(1000, 5000, 1500),  # 3m height
    properties={
        'width': 400,
        'height': 3000,
        'material': 'concrete_C30',
        'reinforcement': '8Φ20'
    }
)
graph.add_element(column)

# Connect column to foundation
graph.add_relationship(
    source_id="column_001",
    target_id="foundation_001",
    relation_type=RelationType.SUPPORTED_BY,
    weight=1.0
)

# Add wall
wall = CADElement(
    id="wall_001",
    element_type=ElementType.WALL,
    centroid=(5000, 5000, 1500),
    properties={
        'length': 10000,
        'height': 3000,
        'thickness': 200,
        'material': 'brick'
    }
)
graph.add_element(wall)

# Connect wall to column
graph.add_relationship(
    source_id="wall_001",
    target_id="column_001",
    relation_type=RelationType.CONNECTED_TO
)

# Save graph
graph.save_json(Path("building_graph.json"))
```

### Example 2: Bridge (پل)

```python
# Create bridge graph
graph = CADGraph()

# Abutments (پایه‌های انتهایی)
abutment1 = CADElement(
    id="abutment_001",
    element_type=ElementType.BRIDGE,
    centroid=(0, 0, 0),
    properties={
        'type': 'abutment',
        'height': 5000,
        'width': 8000
    }
)
graph.add_element(abutment1)

abutment2 = CADElement(
    id="abutment_002",
    element_type=ElementType.BRIDGE,
    centroid=(50000, 0, 0),  # 50m span
    properties={
        'type': 'abutment',
        'height': 5000,
        'width': 8000
    }
)
graph.add_element(abutment2)

# Deck (عرشه)
deck = CADElement(
    id="deck_001",
    element_type=ElementType.SLAB,
    centroid=(25000, 0, 5000),
    properties={
        'length': 50000,
        'width': 8000,
        'thickness': 300,
        'material': 'concrete_C40',
        'design_load': '40 ton'
    }
)
graph.add_element(deck)

# Connect deck to abutments
graph.add_relationship(
    source_id="deck_001",
    target_id="abutment_001",
    relation_type=RelationType.SUPPORTED_BY
)
graph.add_relationship(
    source_id="deck_001",
    target_id="abutment_002",
    relation_type=RelationType.SUPPORTED_BY
)
```

### Example 3: Tunnel (تونل)

```python
# Create tunnel graph
graph = CADGraph()

# Portal (دهانه)
portal = CADElement(
    id="portal_001",
    element_type=ElementType.TUNNEL,
    centroid=(0, 0, 0),
    properties={
        'type': 'portal',
        'width': 10000,
        'height': 8000
    }
)
graph.add_element(portal)

# Lining sections (پوشش تونل)
for i in range(10):  # 10 sections, 5m each
    lining = CADElement(
        id=f"lining_{i:03d}",
        element_type=ElementType.TUNNEL,
        centroid=(5000 * (i + 1), 0, 0),
        properties={
            'type': 'lining',
            'length': 5000,
            'thickness': 300,
            'material': 'shotcrete_C25',
            'rock_class': 'III'
        }
    )
    graph.add_element(lining)
    
    # Connect to previous section
    if i > 0:
        graph.add_relationship(
            source_id=f"lining_{i:03d}",
            target_id=f"lining_{i-1:03d}",
            relation_type=RelationType.CONNECTED_TO
        )
```

### Example 4: Dam (سد)

```python
# Create dam graph
graph = CADGraph()

# Dam body (بدنه سد)
dam_body = CADElement(
    id="dam_body_001",
    element_type=ElementType.DAM,
    centroid=(0, 0, 50000),  # 50m height
    properties={
        'type': 'gravity_dam',
        'height': 100000,  # 100m
        'crest_length': 200000,  # 200m
        'volume': 500000,  # m³
        'material': 'concrete_C35'
    }
)
graph.add_element(dam_body)

# Foundation
foundation = CADElement(
    id="dam_foundation_001",
    element_type=ElementType.FOUNDATION,
    centroid=(0, 0, -5000),
    properties={
        'type': 'rock',
        'depth': 10000
    }
)
graph.add_element(foundation)

# Spillway (سرریز)
spillway = CADElement(
    id="spillway_001",
    element_type=ElementType.DAM,
    centroid=(50000, 0, 80000),
    properties={
        'type': 'spillway',
        'width': 20000,
        'capacity': '5000 m³/s'
    }
)
graph.add_element(spillway)

# Connect elements
graph.add_relationship(
    source_id="dam_body_001",
    target_id="dam_foundation_001",
    relation_type=RelationType.SUPPORTED_BY
)
graph.add_relationship(
    source_id="spillway_001",
    target_id="dam_body_001",
    relation_type=RelationType.HOSTED_BY
)
```

---

## 6. API Reference <a name="api"></a>

### CADGraph

```python
class CADGraph:
    """Main graph container"""
    
    def add_element(self, element: CADElement) -> None:
        """Add element to graph"""
    
    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relation_type: RelationType,
        weight: float = 1.0
    ) -> None:
        """Add relationship between elements"""
    
    def get_element(self, element_id: str) -> Optional[CADElement]:
        """Get element by ID"""
    
    def get_elements_by_type(
        self, element_type: ElementType
    ) -> List[CADElement]:
        """Get all elements of specific type"""
    
    def get_element_relationships(
        self, element_id: str
    ) -> List[CADRelationship]:
        """Get all relationships for element"""
    
    def save_json(self, path: Path) -> None:
        """Save graph to JSON"""
    
    def load_json(cls, path: Path) -> 'CADGraph':
        """Load graph from JSON"""
    
    def to_pytorch_geometric(self) -> Optional[Data]:
        """Convert to PyTorch Geometric format for GNN"""
```

### CADGraphBuilder

```python
class CADGraphBuilder:
    """Automatic graph construction from DXF"""
    
    def __init__(
        self,
        proximity_threshold: float = 100.0,
        parallel_angle_threshold: float = 5.0
    ):
        """
        Args:
            proximity_threshold: Distance threshold for adjacency (mm)
            parallel_angle_threshold: Angle threshold for parallel (degrees)
        """
    
    def build_from_dxf(self, dxf_path: Path) -> CADGraph:
        """Build graph from DXF file"""
```

### GraphEnhancedCAD3DConverter

```python
class GraphEnhancedCAD3DConverter:
    """Unified 2D→3D converter with graph analysis"""
    
    def __init__(
        self,
        device: str = "cpu",
        vit_model_path: Optional[Path] = None,
        vae_model_path: Optional[Path] = None,
        diffusion_model_path: Optional[Path] = None
    ):
        """Initialize converter with optional model checkpoints"""
    
    def convert_dxf_to_3d(
        self,
        dxf_path: Path,
        output_dxf: Path,
        image_path: Optional[Path] = None,
        use_diffusion: bool = False,
        normalize_range: Tuple[float, float] = (0, 1000)
    ) -> Dict[str, Any]:
        """
        Convert DXF to 3D
        
        Returns:
            Dict with:
                - graph_stats: Graph statistics
                - gnn_analysis: GNN analysis results
                - num_points: Number of 3D points generated
        """
```

---

## 7. پیشرفته / Advanced Topics <a name="advanced"></a>

### 7.1 Parametric Updates

```python
# Define parametric relationship
graph.add_relationship(
    source_id="window_001",
    target_id="wall_001",
    relation_type=RelationType.DEPENDS_ON,
    parameter_expression="window.width = wall.width * 0.3"
)

# Update wall width
wall = graph.get_element("wall_001")
wall.properties['width'] = 5000  # Change to 5m

# TODO: Implement automatic propagation
# This will automatically update window width to 1.5m
```

### 7.2 Structural Analysis with GNN

```python
from cad3d.cad_gnn import CAD_GAT

# Create GNN model
gnn = CAD_GAT(
    node_features=56,
    edge_features=21,
    hidden_dim=256,
    num_layers=4,
    num_heads=8
)

# Convert graph to PyTorch Geometric
graph_data = graph.to_pytorch_geometric()

# GNN inference
output = gnn(
    graph_data.x,
    graph_data.edge_index,
    graph_data.edge_attr
)

# Get structural analysis
structural = output['structural_analysis']  # [num_nodes, 3]
stress = structural[:, 0]  # Stress per element
strain = structural[:, 1]  # Strain per element
displacement = structural[:, 2]  # Displacement per element
```

### 7.3 Custom Element Types

```python
# Add custom element type to enum
from enum import Enum

class CustomElementType(Enum):
    MY_CUSTOM_ELEMENT = "my_custom_element"
    SPECIAL_COMPONENT = "special_component"

# Use in CADElement
element = CADElement(
    id="custom_001",
    element_type=CustomElementType.MY_CUSTOM_ELEMENT,
    properties={'special_property': 123}
)
```

### 7.4 Performance Optimization

```python
# For large graphs, use NetworkX acceleration
import networkx as nx

# Enable NetworkX integration
graph = CADGraph(use_networkx=True)

# Fast shortest path
path = graph.shortest_path("elem_001", "elem_100")

# Fast connected components
components = graph.connected_components()
```

---

## 📝 Next Steps

1. Train GNN models on your specific data
2. Integrate with your existing pipeline
3. Add industry-specific element types
4. Implement parametric update engine
5. Add load calculation for structural analysis

## 🆘 Support

For questions or issues, see:

- GitHub Issues
- Documentation: `docs/`
- Examples: `examples/`

---

**نکات مهم / Important Notes:**

- این سیستم در حال توسعه است
- برای استفاده در پروژه‌های واقعی، حتماً تست کنید
- برای تحلیل ساختاری دقیق، با مهندس سازه مشورت کنید
- پیشنهاد می‌شود از GPU برای GNN استفاده کنید

This system is under development. Always test before using in real projects. For accurate structural analysis, consult with structural engineers. GPU recommended for GNN inference.
