# CRF + GNN Integration Guide

# راهنمای یکپارچه‌سازی CRF و GNN

## 🎯 نمای کلی / Overview

این سند راهنمای استفاده از سیستم یکپارچه **CRF + GNN** برای تحلیل نقشه‌های CAD در تمام صنایع است.

This guide explains how to use the unified **CRF + GNN** system for analyzing CAD drawings across all industries.

---

## 🏗️ معماری سیستم / System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT IMAGE                              │
│                    (نقشه 2D / 2D Drawing)                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             v
┌─────────────────────────────────────────────────────────────────┐
│                    STEP 1: CNN/U-Net                            │
│                 Initial Segmentation                             │
│   (تشخیص اولیه عناصر: دیوار، ستون، تیر، ...)                   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             v
┌─────────────────────────────────────────────────────────────────┐
│                    STEP 2: CRF Refinement                       │
│              Conditional Random Fields                           │
│   (بهبود مرزها و حذف نویز / Boundary refinement)               │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             v
┌─────────────────────────────────────────────────────────────────┐
│                  STEP 3: Graph Construction                     │
│         Convert segmentation → CAD Graph                         │
│   (ساخت گراف: node=element, edge=relationship)                 │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             v
┌─────────────────────────────────────────────────────────────────┐
│              STEP 4: Industry-Specific GNN                      │
│          Graph Neural Network Analysis                           │
│   (تحلیل با GNN مخصوص صنعت)                                    │
│   - Building: Load analysis                                      │
│   - Bridge: Stress analysis                                      │
│   - Road: Traffic capacity                                       │
│   - Dam: Stability analysis                                      │
│   - Tunnel: Support requirements                                 │
│   - Machinery: Tolerance & material                              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             v
┌─────────────────────────────────────────────────────────────────┐
│                    STEP 5: 3D Generation                        │
│                     (Optional)                                   │
│   GNN embeddings → VAE/Diffusion → 3D Model                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📦 ماژول‌های جدید / New Modules

### 1. `crf_segmentation.py` - Conditional Random Fields

**قابلیت‌ها:**

- `LinearChainCRF`: برای sequence labeling (دنبال کردن خطوط)
- `DenseCRF2D`: برای image segmentation (مرزبندی دقیق)
- `CRFEnhancedSegmentation`: ترکیب CNN + CRF

**استفاده:**

```python
from cad3d.crf_segmentation import CRFEnhancedSegmentation, create_simple_unet

# Create model
backbone = create_simple_unet(num_classes=10)
model = CRFEnhancedSegmentation(
    backbone=backbone,
    num_classes=10,
    use_crf=True,
    crf_params={
        'sxy_gaussian': 3.0,      # Spatial smoothness
        'compat_gaussian': 3.0,
        'sxy_bilateral': 80.0,    # Color-based smoothness
        'srgb_bilateral': 13.0,
        'compat_bilateral': 10.0,
        'num_iterations': 5
    }
)

# Predict with CRF refinement
segmentation = model.predict(image, images_rgb=image_rgb, use_crf=True)
```

**چه موقع استفاده کنیم:**

- ✅ وقتی مرزها دقیق نیستند
- ✅ وقتی نویز زیاد است
- ✅ برای بهبود خروجی CNN/U-Net
- ✅ برای segmentation نقشه‌های پیچیده

### 2. `industrial_gnn.py` - Industry-Specific GNN

**قابلیت‌ها:**

- `IndustrySpecificGNN`: GNN مخصوص هر صنعت
- `HierarchicalGNN`: درک سلسله‌مراتب (ساختمان → طبقه → اتاق)
- `UncertaintyAwareGNN`: با اندازه‌گیری عدم‌قطعیت

**صنایع پشتیبانی‌شده:**

```python
class IndustryType(Enum):
    BUILDING = "building"        # ساختمان‌سازی
    BRIDGE = "bridge"            # پل‌سازی
    ROAD = "road"                # جاده‌سازی
    DAM = "dam"                  # سدسازی
    TUNNEL = "tunnel"            # تونل‌سازی
    FACTORY = "factory"          # کارخانه
    MACHINERY = "machinery"      # ماشین‌سازی
    MEP = "mep"                  # تاسیسات
    ELECTRICAL = "electrical"    # برق
    PLUMBING = "plumbing"        # لوله‌کشی
    HVAC = "hvac"                # تهویه مطبوع
    RAILWAY = "railway"          # راه‌آهن
    AIRPORT = "airport"          # فرودگاه
    SHIPBUILDING = "shipbuilding" # کشتی‌سازی
```

**استفاده:**

```python
from cad3d.industrial_gnn import create_industry_gnn

# Create GNN for building industry
model = create_industry_gnn(
    industry="building",
    node_features=56,
    edge_features=21,
    hidden_dim=256
)

# Forward pass
output = model(node_features, edge_index, edge_attr)

# Building-specific outputs:
# - element_type: wall, column, beam, ...
# - structural_role: load-bearing, partition, ...
# - load_capacity: vertical, horizontal, lateral loads
```

**خروجی‌های مخصوص هر صنعت:**

| Industry | Outputs |
|----------|---------|
| Building | element_type, structural_role, load_capacity |
| Bridge | component_type, stress, deflection |
| Road | lane_type, traffic, condition |
| Dam | section_type, pressure, stability |
| Tunnel | lining_type, rock_class, support |
| Machinery | part_type, tolerance, material |

### 3. `unified_crf_gnn.py` - Unified System

**قابلیت:**
سیستم یکپارچه که همه چیز را به هم متصل می‌کند.

**استفاده:**

```python
from cad3d.unified_crf_gnn import UnifiedCADAnalyzer
from PIL import Image
import numpy as np

# Load image
image = Image.open("plan.png").convert('RGB')
image_np = np.array(image)

# Create analyzer
analyzer = UnifiedCADAnalyzer(
    industry="building",  # or "bridge", "road", "dam", "tunnel", "machinery"
    num_classes=10,
    hidden_dim=256,
    device="cuda",
    use_crf=True
)

# Analyze
result = analyzer.analyze_image(
    image=image_np,
    generate_3d=True
)

# Results:
# - segmentation: Refined segmentation map
# - graph: CAD graph with elements and relationships
# - gnn_analysis: Industry-specific analysis
# - points_3d: 3D point cloud (if generate_3d=True)
```

---

## 🎯 مثال‌های کاربردی / Use Cases

### Example 1: ساختمان‌سازی / Building Construction

```python
from cad3d.unified_crf_gnn import UnifiedCADAnalyzer
import numpy as np
from PIL import Image

# Load architectural plan
plan_image = Image.open("building_plan.png").convert('RGB')
plan_np = np.array(plan_image)

# Create building analyzer
analyzer = UnifiedCADAnalyzer(
    industry="building",
    device="cuda"
)

# Analyze
result = analyzer.analyze_image(plan_np, generate_3d=True)

# Extract building analysis
building_analysis = result['gnn_analysis']

print(f"Max Load: {building_analysis['max_load']:.2f} kN")
print(f"Avg Load: {building_analysis['avg_load']:.2f} kN")

# Graph statistics
stats = result['graph_stats']
print(f"Walls: {stats['elements_by_type'].get('WALL', 0)}")
print(f"Columns: {stats['elements_by_type'].get('COLUMN', 0)}")
print(f"Beams: {stats['elements_by_type'].get('BEAM', 0)}")

# Save 3D model
if 'points_3d' in result:
    np.save("building_3d.npy", result['points_3d'])
```

### Example 2: پل‌سازی / Bridge Engineering

```python
analyzer = UnifiedCADAnalyzer(
    industry="bridge",
    device="cuda"
)

result = analyzer.analyze_image(bridge_image, generate_3d=True)

# Bridge-specific analysis
bridge_analysis = result['gnn_analysis']

print(f"Max Stress: {bridge_analysis['max_stress']:.2f} MPa")
print(f"Max Shear: {bridge_analysis['max_shear']:.2f} MPa")

# Check if stress is within limits
if bridge_analysis['max_stress'] < 350:  # Steel S355
    print("✅ Stress within limits")
else:
    print("⚠️  Stress exceeds limit!")
```

### Example 3: جاده‌سازی / Road Construction

```python
analyzer = UnifiedCADAnalyzer(
    industry="road",
    device="cuda"
)

result = analyzer.analyze_image(road_image)

# Road analysis
road_analysis = result['gnn_analysis']

print(f"Traffic Capacity: {road_analysis['avg_capacity']:.0f} vehicles/hour")

# Detect lanes from graph
graph = result['graph']
lanes = graph.get_elements_by_type(ElementType.ROAD)
print(f"Number of lanes: {len(lanes)}")
```

### Example 4: سدسازی / Dam Construction

```python
analyzer = UnifiedCADAnalyzer(
    industry="dam",
    device="cuda"
)

result = analyzer.analyze_image(dam_image)

# Dam stability analysis
dam_analysis = result['gnn_analysis']

# Check stability factors (should be > 1.5 for safety)
if 'stability' in dam_analysis:
    print("Stability factors:")
    print(f"  Sliding: {dam_analysis['stability'][0]:.2f}")
    print(f"  Overturning: {dam_analysis['stability'][1]:.2f}")
    print(f"  Bearing: {dam_analysis['stability'][2]:.2f}")
```

### Example 5: تونل‌سازی / Tunnel Construction

```python
analyzer = UnifiedCADAnalyzer(
    industry="tunnel",
    device="cuda"
)

result = analyzer.analyze_image(tunnel_image)

# Tunnel support requirements
tunnel_analysis = result['gnn_analysis']

# Determine rock class and support
graph = result['graph']
for element in graph.elements.values():
    if element.element_type == ElementType.TUNNEL:
        rock_class = element.properties.get('rock_class', 'Unknown')
        print(f"Section {element.id}: Rock Class {rock_class}")
```

### Example 6: ماشین‌سازی / Machinery Manufacturing

```python
analyzer = UnifiedCADAnalyzer(
    industry="machinery",
    device="cuda"
)

result = analyzer.analyze_image(machine_drawing)

# Part analysis
machinery_analysis = result['gnn_analysis']

# Check tolerances
graph = result['graph']
for element in graph.elements.values():
    if element.element_type == ElementType.GEAR:
        tolerance = element.properties.get('tolerance', [0, 0, 0])
        print(f"Gear {element.id}: Tolerance ±{tolerance[0]:.3f} mm")
```

---

## 🔬 تفاوت CRF با سایر روش‌ها / CRF vs Other Methods

| Method | Boundary Quality | Speed | Context-Aware | Post-Processing |
|--------|-----------------|-------|---------------|----------------|
| CNN only | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ❌ | ❌ |
| CNN + Post-processing | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ❌ | ✅ |
| **CNN + CRF** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ✅ | ✅ |
| U-Net only | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ | ❌ |
| **U-Net + CRF** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ✅ | ✅ |

**چرا CRF؟**

- ✅ مرزهای دقیق‌تر (smoother boundaries)
- ✅ استفاده از context (neighboring pixels)
- ✅ حذف نویز
- ✅ بهبود consistency

---

## 🚀 Performance Tips

### 1. GPU Acceleration

```python
# Use GPU for faster processing
analyzer = UnifiedCADAnalyzer(
    industry="building",
    device="cuda"  # instead of "cpu"
)
```

### 2. Batch Processing

```python
# Process multiple images
images = [img1, img2, img3, ...]  # List of images

results = []
for img in images:
    result = analyzer.analyze_image(img)
    results.append(result)
```

### 3. CRF Parameters Tuning

برای نقشه‌های مختلف، پارامترهای CRF را تنظیم کنید:

```python
# For noisy images (increase smoothness)
crf_params = {
    'sxy_gaussian': 5.0,      # Increase spatial smoothness
    'compat_gaussian': 5.0,
    'num_iterations': 10      # More iterations
}

# For high-detail images (less smoothness)
crf_params = {
    'sxy_gaussian': 1.0,      # Less smoothing
    'compat_gaussian': 1.0,
    'num_iterations': 3
}
```

### 4. Memory Management

برای نقشه‌های بزرگ:

```python
# Process in tiles
from cad3d.unified_crf_gnn import UnifiedCADAnalyzer

def process_large_image(image, tile_size=512):
    h, w = image.shape[:2]
    
    results = []
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            tile = image[y:y+tile_size, x:x+tile_size]
            result = analyzer.analyze_image(tile)
            results.append(result)
    
    # Merge results
    # ...
    return merged_result
```

---

## 📊 Model Training

### Training CRF-Enhanced Segmentation

```python
from cad3d.crf_segmentation import CRFEnhancedSegmentation, create_simple_unet
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Create model
backbone = create_simple_unet(num_classes=10)
model = CRFEnhancedSegmentation(
    backbone=backbone,
    num_classes=10,
    use_crf=False  # CRF only in inference
)

# Training (no CRF, just CNN)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(100):
    for batch in dataloader:
        images, labels = batch
        
        # Forward
        logits = model(images)
        loss = criterion(logits, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Save
torch.save({
    'model_state_dict': model.state_dict()
}, 'segmentation_model.pth')
```

### Training Industry-Specific GNN

```python
from cad3d.industrial_gnn import create_industry_gnn
import torch
import torch.optim as optim

# Create model
model = create_industry_gnn(
    industry="building",
    node_features=56,
    edge_features=21,
    hidden_dim=256
)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(100):
    for graph_data in graph_dataloader:
        # Forward
        output = model(
            graph_data.x,
            graph_data.edge_index,
            graph_data.edge_attr
        )
        
        # Compute loss (industry-specific)
        loss = 0
        
        if 'element_type' in output:
            loss += F.cross_entropy(output['element_type'], labels['element_type'])
        
        if 'load_capacity' in output:
            loss += F.mse_loss(output['load_capacity'], labels['load_capacity'])
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

---

## 🔧 Troubleshooting

### Problem 1: CRF Too Slow

**Solution**: کاهش تعداد تکرار یا استفاده از CPU چندهسته‌ای

```python
crf_params = {
    'num_iterations': 3  # Reduce from 5 to 3
}
```

### Problem 2: Over-Smoothing

**Solution**: کاهش پارامترهای smoothness

```python
crf_params = {
    'sxy_gaussian': 1.0,    # Reduce from 3.0
    'compat_gaussian': 1.0
}
```

### Problem 3: GNN Out of Memory

**Solution**: کاهش hidden_dim یا استفاده از gradient checkpointing

```python
model = create_industry_gnn(
    industry="building",
    hidden_dim=128  # Reduce from 256
)
```

---

## 📖 References

### Academic Papers

1. **CRF**: Lafferty et al. (2001) - Conditional Random Fields
2. **Dense CRF**: Krähenbühl & Koltun (2011) - Efficient Inference in Fully Connected CRFs
3. **GNN**: Kipf & Welling (2017) - Semi-Supervised Classification with Graph Convolutional Networks
4. **GAT**: Veličković et al. (2018) - Graph Attention Networks

### Tools & Libraries

- **pydensecrf**: <https://github.com/lucasb-eyer/pydensecrf>
- **PyTorch Geometric**: <https://pytorch-geometric.readthedocs.io/>
- **ezdxf**: <https://ezdxf.readthedocs.io/>

---

## ✅ Next Steps

1. ✅ **Train Models**: آموزش مدل‌های Segmentation و GNN با داده‌های واقعی
2. ✅ **Fine-tune CRF**: تنظیم پارامترهای CRF برای هر صنعت
3. ✅ **Collect Data**: جمع‌آوری dataset برای تمام صنایع
4. ✅ **Benchmark**: مقایسه با روش‌های دیگر
5. ✅ **Deploy**: استقرار سیستم برای استفاده تولیدی

---

**Status**: ✅ **READY FOR TRAINING AND DEPLOYMENT**

This unified system combines the best of both worlds:

- **CRF**: For precise boundary detection
- **GNN**: For structural understanding

Perfect for all industries requiring technical drawings! 🎉
