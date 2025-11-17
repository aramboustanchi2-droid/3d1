from __future__ import annotations

from dataclasses import dataclass
from typing import List
from pathlib import Path
import tempfile
import os

import fitz  # PyMuPDF
import cv2


@dataclass
class PDFToImageConverter:
    dpi: int = 300

    def pdf_to_images(self, pdf_path: str) -> List[str]:
        """Render each page of the PDF to a temporary PNG file and return the paths."""
        pdf = fitz.open(pdf_path)
        img_paths: List[str] = []
        scale = self.dpi / 72.0
        try:
            for i in range(len(pdf)):
                page = pdf[i]
                pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
                fd, out_path = tempfile.mkstemp(suffix=f"_p{i+1}.png", prefix="cad3d_pdf_")
                os.close(fd)
                pix.save(out_path)
                img_paths.append(out_path)
        finally:
            pdf.close()
        return img_paths


class CADPipeline:
    def __init__(self, neural_detector, pdf_converter: PDFToImageConverter):
        self.detector = neural_detector
        self.pdf = pdf_converter

    def process_pdf_to_dxf(
        self,
        pdf_path: str,
        out_dxf_path: str,
        confidence_threshold: float = 0.5,  # unused in minimal pipeline
        scale_mm_per_pixel: float = 1.0,
    ) -> None:
        """
        Minimal PDF->DXF: render pages to images, vectorize with NeuralCADDetector,
        then combine into a single DXF by appending entities.
        """
        from .neural_cad_detector import VectorizedResult
        import ezdxf

        image_paths = self.pdf.pdf_to_images(pdf_path)
        try:
            # Create final DXF
            final_doc = ezdxf.new(setup=True)
            final_msp = final_doc.modelspace()

            for img_path in image_paths:
                vec: VectorizedResult = self.detector.vectorize_drawing(img_path, scale_mm_per_pixel)
                # Write to a temporary DXF to reuse conversion code
                import tempfile as _tf
                fd, tmp_dxf = _tf.mkstemp(suffix=".dxf", prefix="cad3d_pdf_vec_")
                os.close(fd)
                try:
                    self.detector.convert_to_dxf(vec, tmp_dxf)
                    # Append entities to final doc
                    doc = ezdxf.readfile(tmp_dxf)
                    msp = doc.modelspace()
                    for e in msp:
                        final_msp.add_entity(e.copy())
                finally:
                    try:
                        os.unlink(tmp_dxf)
                    except OSError:
                        pass

            Path(out_dxf_path).parent.mkdir(parents=True, exist_ok=True)
            final_doc.saveas(out_dxf_path)
        finally:
            for p in image_paths:
                try:
                    os.unlink(p)
                except OSError:
                    pass
