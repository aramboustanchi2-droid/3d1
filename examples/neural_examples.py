"""
مثال‌های کاربردی - استفاده از Neural CAD System
Examples for using Neural CAD Processing
"""

# ============================================================================
# Example 1: تبدیل PDF به DXF با CLI
# ============================================================================

"""
Command Line:
------------
python -m cad3d.cli pdf-to-dxf \
  --input architectural_plan.pdf \
  --output plan_vectorized.dxf \
  --dpi 400 \
  --confidence 0.6 \
  --scale 2.0 \
  --device cuda

این دستور:
- PDF را با وضوح 400 DPI پردازش می‌کند
- المان‌ها را با حداقل اطمینان 60% تشخیص می‌دهد
- مقیاس 2mm per pixel استفاده می‌کند
- از GPU برای سرعت بیشتر استفاده می‌کند
"""


# ============================================================================
# Example 2: تبدیل عکس به DXF با Python API
# ============================================================================

def example_image_to_dxf():
    """تبدیل عکس نقشه به فایل DXF با API"""
    from cad3d.neural_cad_detector import NeuralCADDetector
    from pathlib import Path
    
    # ساخت detector
    print("🔧 Initializing Neural CAD Detector...")
    detector = NeuralCADDetector(device="auto")  # auto = GPU if available
    
    # مسیر فایل‌ها
    input_image = "floor_plan.jpg"
    output_dxf = "floor_plan.dxf"
    
    # Vectorization کامل
    print(f"📐 Vectorizing: {input_image}")
    vectorized = detector.vectorize_drawing(
        input_image,
        scale_mm_per_pixel=3.0,  # 3mm = 1 pixel
        detect_lines=True,
        detect_circles=True,
        detect_text=True
    )
    
    # نمایش نتایج
    print(f"\n✅ Detection Results:")
    print(f"   Lines: {len(vectorized.lines)}")
    print(f"   Circles: {len(vectorized.circles)}")
    print(f"   Texts: {len(vectorized.texts)}")
    print(f"   Elements: {len(vectorized.elements)}")
    
    # ذخیره DXF
    detector.convert_to_dxf(vectorized, output_dxf)
    print(f"\n💾 DXF saved: {output_dxf}")
    
    return vectorized


# ============================================================================
# Example 3: پردازش PDF با تنظیمات پیشرفته
# ============================================================================

def example_advanced_pdf_processing():
    """پردازش پیشرفته PDF با کنترل کامل"""
    from cad3d.neural_cad_detector import NeuralCADDetector
    from cad3d.pdf_processor import PDFToImageConverter, CADPipeline
    
    # تنظیمات PDF converter
    pdf_converter = PDFToImageConverter(
        dpi=600,  # بالاترین کیفیت
        enhance_quality=True,  # بهبود کیفیت تصویر
        detect_cad_pages=True  # فقط صفحات حاوی نقشه
    )
    
    # تنظیمات Neural detector
    detector = NeuralCADDetector(
        detection_model=None,  # استفاده از pre-trained
        segmentation_model=None,  # استفاده از pre-trained
        device="cuda"  # GPU
    )
    
    # ساخت pipeline کامل
    pipeline = CADPipeline(
        neural_detector=detector,
        pdf_converter=pdf_converter
    )
    
    # پردازش
    print("🚀 Processing PDF with advanced settings...")
    pipeline.process_pdf_to_dxf(
        pdf_path="complex_plan.pdf",
        output_dxf="complex_plan.dxf",
        confidence_threshold=0.7,  # اطمینان بالا
        scale_mm_per_pixel=1.5
    )
    
    print("✅ Processing complete!")


# ============================================================================
# Example 4: تبدیل 2D به 3D با هوش مصنوعی
# ============================================================================

def example_2d_to_3d_conversion():
    """تبدیل خودکار نقشه 2D به مدل 3D"""
    from cad3d.neural_cad_detector import NeuralCADDetector, ImageTo3DExtruder
    from cad3d.pdf_processor import PDFToImageConverter, CADPipeline
    
    print("🏗️ 2D to 3D Conversion with AI")
    
    # Components
    detector = NeuralCADDetector(device="auto")
    pdf_converter = PDFToImageConverter(dpi=300)
    extruder = ImageTo3DExtruder()
    
    # Pipeline
    pipeline = CADPipeline(
        neural_detector=detector,
        pdf_converter=pdf_converter,
        extruder_3d=extruder
    )
    
    # تبدیل به 3D
    pipeline.process_pdf_to_3d(
        pdf_path="floor_plan.pdf",
        output_dxf="floor_plan_3d.dxf",
        intelligent_height=True  # پیش‌بینی خودکار ارتفاع
    )
    
    print("✅ 3D model generated!")


# ============================================================================
# Example 5: Batch Processing - پردازش دسته‌ای چند فایل
# ============================================================================

def example_batch_processing():
    """پردازش دسته‌ای چندین PDF/Image"""
    from pathlib import Path
    from cad3d.neural_cad_detector import NeuralCADDetector
    from cad3d.pdf_processor import PDFToImageConverter, CADPipeline
    
    # مسیرها
    input_dir = Path("input_pdfs")
    output_dir = Path("output_dxfs")
    output_dir.mkdir(exist_ok=True)
    
    # Setup
    detector = NeuralCADDetector(device="cuda")
    pdf_converter = PDFToImageConverter(dpi=300)
    pipeline = CADPipeline(detector, pdf_converter)
    
    # پردازش تمام PDF ها
    pdf_files = list(input_dir.glob("*.pdf"))
    print(f"📁 Found {len(pdf_files)} PDF files")
    
    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"\n[{i}/{len(pdf_files)}] Processing: {pdf_path.name}")
        
        output_path = output_dir / f"{pdf_path.stem}.dxf"
        
        try:
            pipeline.process_pdf_to_dxf(
                pdf_path,
                output_path,
                confidence_threshold=0.5
            )
            print(f"  ✅ Success: {output_path.name}")
        except Exception as e:
            print(f"  ❌ Failed: {e}")
    
    print(f"\n✅ Batch processing complete! Output: {output_dir}")


# ============================================================================
# Example 6: استخراج اطلاعات المان‌ها
# ============================================================================

def example_element_detection():
    """تشخیص و استخراج اطلاعات المان‌ها"""
    from cad3d.neural_cad_detector import NeuralCADDetector
    import cv2
    
    detector = NeuralCADDetector(device="auto")
    
    # تشخیص المان‌ها
    elements = detector.detect_from_image(
        "floor_plan.jpg",
        confidence_threshold=0.6,
        enable_segmentation=True  # mask پیکسل به پیکسل
    )
    
    # تحلیل نتایج
    print(f"\n🔍 Detected {len(elements)} elements:")
    
    element_counts = {}
    for elem in elements:
        element_counts[elem.element_type] = element_counts.get(elem.element_type, 0) + 1
    
    for elem_type, count in sorted(element_counts.items()):
        print(f"   {elem_type:15s}: {count:3d}")
    
    # جزئیات اولین المان
    if elements:
        first = elements[0]
        print(f"\nFirst element details:")
        print(f"   Type: {first.element_type}")
        print(f"   Confidence: {first.confidence:.2%}")
        print(f"   Bounding Box: {first.bbox}")
        print(f"   Has Mask: {first.mask is not None}")
    
    return elements


# ============================================================================
# Example 7: تنظیمات سفارشی برای انواع مختلف نقشه
# ============================================================================

def example_drawing_type_specific():
    """تنظیمات بهینه برای انواع مختلف نقشه"""
    from cad3d.pdf_processor import PDFToImageConverter, CADPipeline
    from cad3d.neural_cad_detector import NeuralCADDetector
    
    detector = NeuralCADDetector(device="auto")
    
    # ===== نقشه پلان (Floor Plan) =====
    print("📐 Processing Floor Plan...")
    pdf_conv_plan = PDFToImageConverter(dpi=400, enhance_quality=True)
    pipeline_plan = CADPipeline(detector, pdf_conv_plan)
    pipeline_plan.process_pdf_to_dxf(
        "floor_plan.pdf",
        "floor_plan.dxf",
        confidence_threshold=0.6,
        scale_mm_per_pixel=2.0
    )
    
    # ===== نقشه نما (Elevation) =====
    print("\n🏛️ Processing Elevation...")
    pdf_conv_elev = PDFToImageConverter(dpi=300, enhance_quality=True)
    pipeline_elev = CADPipeline(detector, pdf_conv_elev)
    pipeline_elev.process_pdf_to_dxf(
        "elevation.pdf",
        "elevation.dxf",
        confidence_threshold=0.5,
        scale_mm_per_pixel=1.5
    )
    
    # ===== نقشه جزئیات (Detail) =====
    print("\n🔬 Processing Detail Drawing...")
    pdf_conv_detail = PDFToImageConverter(dpi=600, enhance_quality=True)  # بالاترین کیفیت
    pipeline_detail = CADPipeline(detector, pdf_conv_detail)
    pipeline_detail.process_pdf_to_dxf(
        "detail.pdf",
        "detail.dxf",
        confidence_threshold=0.7,  # دقت بالا
        scale_mm_per_pixel=0.5  # جزئیات بیشتر
    )


# ============================================================================
# Example 8: یکپارچه‌سازی با Architectural Analyzer
# ============================================================================

def example_integration_with_analyzer():
    """استفاده ترکیبی: Neural Detection + Architectural Analysis"""
    from cad3d.neural_cad_detector import NeuralCADDetector
    from cad3d.architectural_analyzer import ArchitecturalAnalyzer
    from pathlib import Path
    
    # مرحله 1: تبدیل Image/PDF به DXF با Neural
    print("🤖 Step 1: Neural conversion...")
    detector = NeuralCADDetector(device="auto")
    
    vectorized = detector.vectorize_drawing(
        "scanned_plan.jpg",
        scale_mm_per_pixel=2.0
    )
    
    temp_dxf = Path("temp_converted.dxf")
    detector.convert_to_dxf(vectorized, temp_dxf)
    
    # مرحله 2: تحلیل معماری DXF با Analyzer
    print("\n📊 Step 2: Architectural analysis...")
    analyzer = ArchitecturalAnalyzer(str(temp_dxf))
    analysis = analyzer.analyze()
    
    # نمایش نتایج
    print(f"\n✅ Combined Analysis Results:")
    print(f"   Drawing Type: {analysis.drawing_type.value}")
    print(f"   Walls: {len(analysis.walls)}")
    print(f"   Doors: {len(analysis.doors)}")
    print(f"   Windows: {len(analysis.windows)}")
    print(f"   Structural: {len(analysis.structural_elements)}")
    print(f"   MEP: {len(analysis.mep_elements)}")
    print(f"   Total Area: {analysis.total_area:.2f} m²")
    
    # حذف فایل موقت
    temp_dxf.unlink()


# ============================================================================
# Run Examples
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("🤖 NEURAL CAD SYSTEM - EXAMPLES")
    print("="*70)
    
    import sys
    
    print("\nNote: این مثال‌ها نیاز به نصب dependencies دارند:")
    print("  pip install -r requirements-neural.txt")
    print("\nبرای اجرا:")
    print("  1. Uncomment example function call below")
    print("  2. python examples/neural_examples.py")
    print("\n" + "="*70)
    
    # Uncomment to run:
    # example_image_to_dxf()
    # example_advanced_pdf_processing()
    # example_2d_to_3d_conversion()
    # example_batch_processing()
    # example_element_detection()
    # example_drawing_type_specific()
    # example_integration_with_analyzer()
