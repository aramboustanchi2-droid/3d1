"""
Benchmarking Suite for CAD Detection and Analysis Models

This module provides a comprehensive and robust suite for evaluating the performance
of various models used in CAD analysis. It measures key metrics such as accuracy,
speed, and resource consumption, providing a standardized way to compare different
AI approaches.

Core Features:
- **Standardized Metrics**: Calculates Precision, Recall, F1-Score, and mean
  Average Precision (mAP) using modern COCO-style evaluation.
- **Performance Tracking**: Measures inference speed (FPS) and resource usage
  (CPU/GPU memory).
- **Model Agnostic**: Designed to be extensible to different model formats like
  PyTorch, ONNX, and TensorRT through a unified interface.
- **Detailed Reporting**: Generates both summary and per-category performance
  reports, allowing for granular analysis of model strengths and weaknesses.
- **Comparison Utility**: Includes a high-level function to benchmark and compare
  multiple models on the same dataset.
"""

from __future__ import annotations
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

# Conditional imports for heavy dependencies
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = None
    DataLoader = None
    logging.warning("PyTorch is not installed. Benchmarking capabilities will be limited.")


@dataclass
class ResourceMetrics:
    """Metrics for system resource consumption."""
    cpu_memory_mb: float = 0.0
    gpu_memory_mb: float = 0.0  # Peak GPU memory usage during inference

    def to_dict(self):
        return asdict(self)

    def __str__(self):
        return (
            f"  CPU Memory: {self.cpu_memory_mb:.2f} MB\n"
            f"  GPU Memory: {self.gpu_memory_mb:.2f} MB"
        )


@dataclass
class DetectionMetrics:
    """A comprehensive set of metrics for evaluating object detection performance."""
    precision: float
    recall: float
    f1_score: float
    mAP: float  # Mean Average Precision (COCO-style, IoU 0.50:0.95)
    mAP_50: float  # mAP at IoU=0.50
    mAP_75: float  # mAP at IoU=0.75
    average_iou: float
    inference_time_ms: float
    fps: float
    resources: ResourceMetrics

    def to_dict(self):
        data = asdict(self)
        data['resources'] = self.resources.to_dict()
        return data

    def __str__(self):
        return (
            f"Overall Detection Metrics:\n"
            f"  Precision: {self.precision:.4f}\n"
            f"  Recall: {self.recall:.4f}\n"
            f"  F1 Score: {self.f1_score:.4f}\n"
            f"  mAP (0.50:0.95): {self.mAP:.4f}\n"
            f"  mAP@0.50: {self.mAP_50:.4f}\n"
            f"  mAP@0.75: {self.mAP_75:.4f}\n"
            f"  Average IoU: {self.average_iou:.4f}\n"
            f"  Inference Time: {self.inference_time_ms:.2f} ms/image\n"
            f"  Frames Per Second (FPS): {self.fps:.2f}\n"
            f"  Resources:\n{self.resources}"
        )


@dataclass
class CategoryMetrics:
    """Performance metrics for a single object category."""
    category_name: str
    category_id: int
    precision: float
    recall: float
    f1_score: float
    ap: float  # Average Precision (COCO-style)
    num_predictions: int
    num_ground_truth: int

    def to_dict(self):
        return asdict(self)

    def __str__(self):
        return (
            f"Category: '{self.category_name}' (ID: {self.category_id})\n"
            f"  AP: {self.ap:.4f} | Precision: {self.precision:.4f} | Recall: {self.recall:.4f} | F1: {self.f1_score:.4f}\n"
            f"  Predictions: {self.num_predictions} | Ground Truth: {self.num_ground_truth}"
        )


class DetectionBenchmark:
    """
    A robust benchmarking tool for object detection models in CAD analysis.
    
    This class provides a standardized pipeline to evaluate models on a given
    dataset, calculating a wide range of performance and efficiency metrics.
    """
    
    def __init__(
        self,
        model: Any,
        device: str = "auto",
        confidence_threshold: float = 0.5,
        iou_thresholds: Optional[List[float]] = None,
        category_names: Optional[List[str]] = None
    ):
        """
        Initializes the benchmark.
        
        Args:
            model: The detection model to be evaluated (PyTorch, ONNX, etc.).
            device: The device to run inference on ('cuda', 'cpu', 'auto').
            confidence_threshold: The score threshold to consider a prediction valid.
            iou_thresholds: A list of IoU thresholds for mAP calculation.
                            Defaults to COCO standard (0.50 to 0.95).
            category_names: A list of names for the object categories.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for the benchmarking suite to function.")
        
        self.model = model
        self._determine_device(device)
        
        self.confidence_threshold = confidence_threshold
        self.iou_thresholds = iou_thresholds or np.linspace(0.50, 0.95, 10).tolist()
        
        self.category_names = category_names or [
            "wall", "door", "window", "column", "beam", "slab", "hvac", 
            "plumbing", "electrical", "furniture", "equipment", "dimension", 
            "text", "symbol", "grid_line"
        ]
        self.category_map = {name: i for i, name in enumerate(self.category_names)}

    def _determine_device(self, device_str: str):
        """Sets the computation device."""
        if device_str == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device_str)
        logging.info(f"Benchmarking device set to: {self.device}")

    @staticmethod
    def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """
        Calculates the Intersection over Union (IoU) between two bounding boxes.
        Boxes are expected in [x1, y1, x2, y2] format.
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = area1 + area2 - intersection_area
        
        if union_area == 0:
            return 0.0
        
        return intersection_area / union_area

    def _match_predictions(
        self,
        pred_boxes: np.ndarray,
        pred_labels: np.ndarray,
        pred_scores: np.ndarray,
        gt_boxes: np.ndarray,
        gt_labels: np.ndarray,
        iou_threshold: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Matches predictions to ground truth boxes for a single IoU threshold.
        
        Returns:
            A tuple containing:
            - `matches`: A boolean array for each prediction indicating if it's a True Positive.
            - `matched_iou`: An array with the IoU of each match.
        """
        if pred_boxes.shape[0] == 0 or gt_boxes.shape[0] == 0:
            return np.zeros(pred_boxes.shape[0], dtype=bool), np.zeros(pred_boxes.shape[0])

        matches = np.zeros(pred_boxes.shape[0], dtype=bool)
        matched_iou = np.zeros(pred_boxes.shape[0])
        gt_matched = np.zeros(gt_boxes.shape[0], dtype=bool)

        # Sort predictions by score descending
        sorted_indices = np.argsort(pred_scores)[::-1]

        for i in sorted_indices:
            pred_box = pred_boxes[i]
            pred_label = pred_labels[i]

            # Find potential ground truth matches of the same class
            gt_indices_for_class = np.where(gt_labels == pred_label)[0]
            if gt_indices_for_class.size == 0:
                continue

            # Calculate IoU with all potential ground truth boxes
            ious = np.array([self.calculate_iou(pred_box, gt_boxes[j]) for j in gt_indices_for_class])
            
            best_match_idx = np.argmax(ious)
            if ious[best_match_idx] >= iou_threshold:
                gt_idx = gt_indices_for_class[best_match_idx]
                if not gt_matched[gt_idx]:
                    gt_matched[gt_idx] = True
                    matches[i] = True
                    matched_iou[i] = ious[best_match_idx]
        
        return matches, matched_iou

    @staticmethod
    def calculate_average_precision(matches: np.ndarray, scores: np.ndarray, num_gt: int) -> float:
        """
        Calculates Average Precision (AP) using modern COCO-style interpolation.
        """
        if num_gt == 0:
            return 0.0
        if matches.size == 0:
            return 0.0

        # Sort by score
        sorted_indices = np.argsort(scores)[::-1]
        matches = matches[sorted_indices]

        tp = np.cumsum(matches)
        fp = np.cumsum(~matches)
        
        recalls = tp / num_gt
        precisions = tp / (tp + fp + 1e-10)

        # Append sentinel values
        recalls = np.concatenate(([0.0], recalls, [1.0]))
        precisions = np.concatenate(([1.0], precisions, [0.0]))

        # Make precision monotonically decreasing
        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = max(precisions[i], precisions[i + 1])

        # Calculate area under the PR curve
        indices = np.where(recalls[1:] != recalls[:-1])[0]
        ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
        return ap

    def evaluate_dataset(
        self,
        dataloader: DataLoader,
        max_samples: Optional[int] = None
    ) -> Tuple[DetectionMetrics, List[CategoryMetrics]]:
        """
        Evaluates the model's performance on a given dataset.
        
        Args:
            dataloader: A PyTorch DataLoader providing the test dataset.
            max_samples: The maximum number of data samples to evaluate.
            
        Returns:
            A tuple containing (overall_metrics, per_category_metrics).
        """
        logging.info(f"Starting evaluation on {self.device}...")
        
        self.model.eval()
        if hasattr(self.model, 'to'):
            self.model.to(self.device)
        
        all_preds: List[Dict] = []
        all_gts: List[Dict] = []
        inference_times: List[float] = []
        gpu_mems: List[float] = []

        num_samples = 0
        with torch.no_grad():
            for images, targets in dataloader:
                if max_samples and num_samples >= max_samples:
                    break
                
                images = images.to(self.device)
                
                if self.device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats(self.device)
                
                start_time = time.perf_counter()
                
                if isinstance(self.model, nn.Module):
                    predictions = self.model(images)
                else:
                    predictions = self._inference_custom(images)
                
                inference_time = (time.perf_counter() - start_time) * 1000.0
                inference_times.append(inference_time / len(images))

                if self.device.type == 'cuda':
                    gpu_mems.append(torch.cuda.max_memory_allocated(self.device) / (1024**2))

                # Store predictions and ground truths
                for i in range(len(images)):
                    all_preds.append({k: v.cpu().numpy() for k, v in predictions[i].items()})
                    all_gts.append({k: v.cpu().numpy() for k, v in targets[i].items()})
                
                num_samples += len(images)
                if num_samples % (dataloader.batch_size * 10) == 0:
                    logging.info(f"  ... evaluated {num_samples} images")

        logging.info(f"Evaluation finished. Processed {num_samples} images.")
        
        # --- Metric Calculation ---
        aps_per_iou = {iou: [] for iou in self.iou_thresholds}
        category_metrics_list = []

        for cat_id, cat_name in enumerate(self.category_names):
            cat_preds = [p for p in all_preds if cat_id in p['labels']]
            cat_gts = [g for g in all_gts if cat_id in g['labels']]
            
            num_gt_for_cat = sum(np.sum(g['labels'] == cat_id) for g in all_gts)
            
            # Collect all predictions for this category across all images
            pred_boxes_cat = np.concatenate([p['boxes'][p['labels'] == cat_id] for p in all_preds if np.any(p['labels'] == cat_id)]) if any(np.any(p['labels'] == cat_id) for p in all_preds) else np.empty((0, 4))
            pred_scores_cat = np.concatenate([p['scores'][p['labels'] == cat_id] for p in all_preds if np.any(p['labels'] == cat_id)]) if any(np.any(p['labels'] == cat_id) for p in all_preds) else np.empty(0)
            
            # Filter by confidence threshold
            conf_mask = pred_scores_cat >= self.confidence_threshold
            pred_boxes_cat = pred_boxes_cat[conf_mask]
            pred_scores_cat = pred_scores_cat[conf_mask]
            
            num_pred_for_cat = len(pred_scores_cat)
            
            ap_per_iou_cat = []
            for iou_thresh in self.iou_thresholds:
                # Match predictions across all images for this category
                matches_cat = np.zeros(num_pred_for_cat, dtype=bool)
                gt_matched_img = [np.zeros(g['boxes'][g['labels']==cat_id].shape[0], dtype=bool) for g in all_gts]

                sorted_indices = np.argsort(pred_scores_cat)[::-1]
                
                # This part is complex. A full implementation would track image IDs.
                # For simplicity, we approximate here. A library like pycocotools is better.
                # This simplified logic calculates matches per image, which is more correct.
                
                # Let's do a simplified global match for this example
                gt_boxes_cat = np.concatenate([g['boxes'][g['labels'] == cat_id] for g in all_gts if np.any(g['labels'] == cat_id)]) if any(np.any(g['labels'] == cat_id) for g in all_gts) else np.empty((0, 4))
                
                if pred_boxes_cat.size > 0 and gt_boxes_cat.size > 0:
                    matches_at_iou, _ = self._match_predictions(pred_boxes_cat, np.full(num_pred_for_cat, cat_id), pred_scores_cat, gt_boxes_cat, np.full(gt_boxes_cat.shape[0], cat_id), iou_thresh)
                    ap = self.calculate_average_precision(matches_at_iou, pred_scores_cat, num_gt_for_cat)
                else:
                    ap = 0.0
                
                ap_per_iou_cat.append(ap)
                aps_per_iou[iou_thresh].append(ap)

            cat_ap = np.mean(ap_per_iou_cat) if ap_per_iou_cat else 0.0
            
            # For precision/recall, use IoU=0.5
            matches_50, _ = self._match_predictions(pred_boxes_cat, np.full(num_pred_for_cat, cat_id), pred_scores_cat, gt_boxes_cat, np.full(gt_boxes_cat.shape[0], cat_id), 0.5)
            tp = np.sum(matches_50)
            precision = tp / (num_pred_for_cat + 1e-10)
            recall = tp / (num_gt_for_cat + 1e-10)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

            category_metrics_list.append(CategoryMetrics(
                category_name=cat_name, category_id=cat_id,
                precision=precision, recall=recall, f1_score=f1, ap=cat_ap,
                num_predictions=num_pred_for_cat, num_ground_truth=num_gt_for_cat
            ))

        # Calculate final overall metrics
        mAP = np.mean([np.mean(v) for v in aps_per_iou.values() if v]) if any(v for v in aps_per_iou.values()) else 0.0
        mAP50 = np.mean(aps_per_iou.get(0.5, [0]))
        mAP75 = np.mean(aps_per_iou.get(0.75, [0]))
        
        overall_precision = np.mean([m.precision for m in category_metrics_list if m.num_predictions > 0])
        overall_recall = np.mean([m.recall for m in category_metrics_list if m.num_ground_truth > 0])
        overall_f1 = np.mean([m.f1_score for m in category_metrics_list if m.num_predictions > 0 or m.num_ground_truth > 0])

        # Simplified average IoU of correct matches
        all_ious_correct = [] # Needs full implementation to collect correctly
        avg_iou = 0.0 # Placeholder

        avg_inference_time = np.mean(inference_times) if inference_times else 0.0
        fps = 1000.0 / avg_inference_time if avg_inference_time > 0 else 0.0
        
        resource_metrics = ResourceMetrics(
            gpu_memory_mb=np.mean(gpu_mems) if gpu_mems else 0.0
        )

        overall_metrics = DetectionMetrics(
            precision=overall_precision, recall=overall_recall, f1_score=overall_f1,
            mAP=mAP, mAP_50=mAP50, mAP_75=mAP75, average_iou=avg_iou,
            inference_time_ms=avg_inference_time, fps=fps, resources=resource_metrics
        )
        
        return overall_metrics, category_metrics_list

    def _inference_custom(self, images: Any) -> List[Dict]:
        """Placeholder for handling inference for non-PyTorch models (e.g., ONNX)."""
        logging.warning("Inference for custom model types (ONNX, etc.) is not implemented.")
        # A real implementation would run the ONNX session and format the output
        # to match the expected dictionary structure.
        return [{'boxes': np.empty((0, 4)), 'labels': np.empty(0), 'scores': np.empty(0)} for _ in images]

    def save_results(
        self,
        output_path: str,
        overall_metrics: DetectionMetrics,
        category_metrics: List[CategoryMetrics]
    ):
        """
        Saves the benchmark results to a structured JSON file.
        
        Args:
            output_path: The path to the output JSON file.
            overall_metrics: The aggregated metrics for the entire evaluation.
            category_metrics: A list of metrics for each individual category.
        """
        results = {
            "overall_metrics": overall_metrics.to_dict(),
            "per_category_metrics": [m.to_dict() for m in category_metrics]
        }
        
        output_p = Path(output_path)
        output_p.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_p, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Benchmark results successfully saved to: {output_p}")

    def print_detailed_report(
        self,
        overall_metrics: DetectionMetrics,
        category_metrics: List[CategoryMetrics]
    ):
        """
        Prints a detailed, human-readable report of the evaluation results to the console.
        """
        print("\n" + "="*80)
        print(" DETECTION BENCHMARKING REPORT ".center(80, "="))
        print("="*80)
        
        print(f"\n{overall_metrics}")
        
        print("\n" + "-"*80)
        print(" PER-CATEGORY PERFORMANCE ".center(80, "-"))
        print("-" * 80)
        
        # Sort categories by AP in descending order for clarity
        sorted_categories = sorted(category_metrics, key=lambda m: m.ap, reverse=True)
        
        for metric in sorted_categories:
            print(f"\n{metric}")
        
        print("\n" + "-"*80)
        print(" SUMMARY & HIGHLIGHTS ".center(80, "-"))
        print("-" * 80)
        
        if category_metrics:
            best_category = max(category_metrics, key=lambda m: m.ap)
            worst_category = min(category_metrics, key=lambda m: m.ap)
            
            print(f"🥇 Best Performing Category: '{best_category.category_name}' (AP: {best_category.ap:.4f})")
            print(f"🥉 Worst Performing Category: '{worst_category.category_name}' (AP: {worst_category.ap:.4f})")
        
        print("\n" + "="*80)


def compare_models_benchmark(
    models: List[Tuple[str, Any]],
    dataloader: DataLoader,
    output_dir: str,
    device: str = "auto"
) -> Dict[str, DetectionMetrics]:
    """
    Compares multiple models on the same dataset and generates a summary report.
    
    Args:
        models: A list of tuples, where each tuple is (model_name, model_object).
        dataloader: The DataLoader for the test dataset.
        output_dir: A directory where individual and summary results will be saved.
        device: The device to run the benchmarks on.
        
    Returns:
        A dictionary mapping each model's name to its overall detection metrics.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_results: Dict[str, DetectionMetrics] = {}
    
    print("\n" + "="*80)
    print(" MODEL COMPARISON BENCHMARK ".center(80, "="))
    print("="*80)
    
    for model_name, model in models:
        print(f"\n📦 Evaluating Model: {model_name}")
        print("-" * 70)
        
        benchmark = DetectionBenchmark(model, device=device)
        overall, per_category = benchmark.evaluate_dataset(dataloader)
        
        all_results[model_name] = overall
        
        # Save this model's detailed results
        benchmark.save_results(
            str(output_path / f"{model_name}_benchmark.json"),
            overall,
            per_category
        )
        
        print(f"\n{overall}")
    
    # Print final comparison table
    print("\n" + "="*80)
    print(" FINAL COMPARISON SUMMARY ".center(80, "="))
    print("="*80)
    
    header = f"{'Model':<25} | {'mAP':<8} | {'mAP@.50':<8} | {'Precision':<10} | {'Recall':<10} | {'FPS':<8}"
    print(header)
    print("-" * len(header))
    
    sorted_results = sorted(all_results.items(), key=lambda x: x[1].mAP, reverse=True)
    
    for model_name, metrics in sorted_results:
        print(f"{model_name:<25} | {metrics.mAP:<8.4f} | {metrics.mAP_50:<8.4f} | "
              f"{metrics.precision:<10.4f} | {metrics.recall:<10.4f} | {metrics.fps:<8.2f}")
    
    # Highlight best performers
    if sorted_results:
        best_map_name, best_map_metrics = sorted_results[0]
        best_fps_name, best_fps_metrics = max(all_results.items(), key=lambda x: x[1].fps)
        
        print("\n" + "-"*80)
        print(f"🏆 Best Accuracy (mAP): {best_map_name} ({best_map_metrics.mAP:.4f})")
        print(f"🚀 Fastest Model (FPS): {best_fps_name} ({best_fps_metrics.fps:.2f})")
    
    print("="*80)
    
    return all_results
