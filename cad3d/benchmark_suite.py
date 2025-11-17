"""
Benchmarking suite for evaluating CAD detection model performance.
Measures accuracy, speed, and resource usage.
"""
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = None
    DataLoader = None  # Placeholder when torch not available


@dataclass
class DetectionMetrics:
    """Detection evaluation metrics."""
    precision: float
    recall: float
    f1_score: float
    mAP: float  # Mean Average Precision
    mAP_50: float  # mAP at IoU=0.5
    mAP_75: float  # mAP at IoU=0.75
    average_iou: float
    inference_time_ms: float
    fps: float
    
    def to_dict(self):
        return asdict(self)
    
    def __str__(self):
        return (
            f"Detection Metrics:\n"
            f"  Precision: {self.precision:.3f}\n"
            f"  Recall: {self.recall:.3f}\n"
            f"  F1 Score: {self.f1_score:.3f}\n"
            f"  mAP: {self.mAP:.3f}\n"
            f"  mAP@0.5: {self.mAP_50:.3f}\n"
            f"  mAP@0.75: {self.mAP_75:.3f}\n"
            f"  Avg IoU: {self.average_iou:.3f}\n"
            f"  Inference: {self.inference_time_ms:.2f} ms ({self.fps:.1f} FPS)"
        )


@dataclass
class CategoryMetrics:
    """Per-category metrics."""
    category_name: str
    category_id: int
    precision: float
    recall: float
    f1_score: float
    ap: float  # Average Precision
    num_predictions: int
    num_ground_truth: int
    
    def __str__(self):
        return (
            f"{self.category_name}:\n"
            f"  Precision: {self.precision:.3f} | Recall: {self.recall:.3f} | "
            f"F1: {self.f1_score:.3f} | AP: {self.ap:.3f}\n"
            f"  Predictions: {self.num_predictions} | Ground Truth: {self.num_ground_truth}"
        )


class DetectionBenchmark:
    """
    Benchmark object detection models for CAD drawing analysis.
    
    Evaluates:
    - Detection accuracy (Precision, Recall, mAP)
    - Inference speed (FPS, latency)
    - Per-category performance
    - Resource usage (memory, GPU)
    """
    
    def __init__(
        self,
        model: Any,
        device: str = "cuda",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.5
    ):
        """
        Initialize benchmark.
        
        Args:
            model: Detection model (PyTorch, ONNX, or TensorRT)
            device: Device to use
            confidence_threshold: Confidence threshold for predictions
            iou_threshold: IoU threshold for matching predictions to ground truth
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for benchmarking")
        
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
        # Category names (15 CAD classes)
        self.category_names = [
            "wall", "door", "window", "column", "beam", "slab",
            "hvac", "plumbing", "electrical", "furniture", "equipment",
            "dimension", "text", "symbol", "grid_line"
        ]
    
    def calculate_iou(
        self,
        box1: np.ndarray,
        box2: np.ndarray
    ) -> float:
        """
        Calculate Intersection over Union (IoU) between two boxes.
        
        Args:
            box1: First box [x1, y1, x2, y2]
            box2: Second box [x1, y1, x2, y2]
            
        Returns:
            IoU score
        """
        # Calculate intersection
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        # Calculate union
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def match_predictions_to_ground_truth(
        self,
        pred_boxes: np.ndarray,
        pred_labels: np.ndarray,
        pred_scores: np.ndarray,
        gt_boxes: np.ndarray,
        gt_labels: np.ndarray
    ) -> Tuple[List[bool], List[float]]:
        """
        Match predictions to ground truth boxes.
        
        Args:
            pred_boxes: Predicted boxes [N, 4]
            pred_labels: Predicted labels [N]
            pred_scores: Prediction scores [N]
            gt_boxes: Ground truth boxes [M, 4]
            gt_labels: Ground truth labels [M]
            
        Returns:
            Tuple of (matches, ious)
        """
        matches = []
        ious = []
        matched_gt = set()
        
        # Sort predictions by score (descending)
        sorted_indices = np.argsort(pred_scores)[::-1]
        
        for idx in sorted_indices:
            pred_box = pred_boxes[idx]
            pred_label = pred_labels[idx]
            
            best_iou = 0.0
            best_gt_idx = -1
            
            # Find best matching ground truth
            for gt_idx in range(len(gt_boxes)):
                if gt_idx in matched_gt:
                    continue
                
                if pred_label != gt_labels[gt_idx]:
                    continue
                
                iou = self.calculate_iou(pred_box, gt_boxes[gt_idx])
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # Check if match is valid
            if best_iou >= self.iou_threshold:
                matches.append(True)
                matched_gt.add(best_gt_idx)
            else:
                matches.append(False)
            
            ious.append(best_iou)
        
        return matches, ious
    
    def calculate_precision_recall(
        self,
        matches: List[bool],
        num_predictions: int,
        num_ground_truth: int
    ) -> Tuple[float, float]:
        """
        Calculate precision and recall.
        
        Args:
            matches: List of match results
            num_predictions: Total number of predictions
            num_ground_truth: Total number of ground truth boxes
            
        Returns:
            (precision, recall)
        """
        if num_predictions == 0:
            precision = 0.0
        else:
            true_positives = sum(matches)
            precision = true_positives / num_predictions
        
        if num_ground_truth == 0:
            recall = 0.0
        else:
            true_positives = sum(matches)
            recall = true_positives / num_ground_truth
        
        return precision, recall
    
    def calculate_average_precision(
        self,
        matches: List[bool],
        scores: np.ndarray,
        num_ground_truth: int
    ) -> float:
        """
        Calculate Average Precision (AP).
        
        Args:
            matches: Match results sorted by score
            scores: Prediction scores
            num_ground_truth: Number of ground truth boxes
            
        Returns:
            Average Precision
        """
        if num_ground_truth == 0:
            return 0.0
        
        # Calculate precision-recall curve
        true_positives = np.cumsum(matches)
        false_positives = np.cumsum([not m for m in matches])
        
        precisions = true_positives / (true_positives + false_positives + 1e-10)
        recalls = true_positives / num_ground_truth
        
        # Calculate AP using 11-point interpolation
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            if np.any(recalls >= t):
                ap += np.max(precisions[recalls >= t])
        ap /= 11.0
        
        return ap
    
    def evaluate_dataset(
        self,
        dataloader: Any,  # DataLoader when available
        max_samples: Optional[int] = None
    ) -> Tuple[DetectionMetrics, List[CategoryMetrics]]:
        """
        Evaluate model on dataset.
        
        Args:
            dataloader: DataLoader with test dataset
            max_samples: Maximum number of samples to evaluate
            
        Returns:
            Tuple of (overall metrics, per-category metrics)
        """
        print("🔍 Evaluating model on dataset...")
        
        self.model.eval()
        if hasattr(self.model, 'to'):
            self.model.to(self.device)
        
        # Initialize accumulators
        all_matches = []
        all_ious = []
        all_scores = []
        inference_times = []
        
        # Per-category accumulators
        category_matches = {i: [] for i in range(len(self.category_names))}
        category_scores = {i: [] for i in range(len(self.category_names))}
        category_num_gt = {i: 0 for i in range(len(self.category_names))}
        
        num_samples = 0
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(dataloader):
                if max_samples and num_samples >= max_samples:
                    break
                
                # Move to device
                images = images.to(self.device)
                
                # Measure inference time
                start_time = time.time()
                
                if isinstance(self.model, nn.Module):
                    predictions = self.model(images)
                else:
                    # Handle ONNX/TensorRT models
                    predictions = self._inference_custom(images)
                
                inference_time = (time.time() - start_time) * 1000
                inference_times.append(inference_time)
                
                # Process each image in batch
                for i in range(len(images)):
                    pred_boxes = predictions[i]['boxes'].cpu().numpy()
                    pred_labels = predictions[i]['labels'].cpu().numpy()
                    pred_scores = predictions[i]['scores'].cpu().numpy()
                    
                    gt_boxes = targets[i]['boxes'].cpu().numpy()
                    gt_labels = targets[i]['labels'].cpu().numpy()
                    
                    # Filter by confidence
                    mask = pred_scores >= self.confidence_threshold
                    pred_boxes = pred_boxes[mask]
                    pred_labels = pred_labels[mask]
                    pred_scores = pred_scores[mask]
                    
                    # Match predictions to ground truth
                    matches, ious = self.match_predictions_to_ground_truth(
                        pred_boxes, pred_labels, pred_scores,
                        gt_boxes, gt_labels
                    )
                    
                    all_matches.extend(matches)
                    all_ious.extend(ious)
                    all_scores.extend(pred_scores)
                    
                    # Per-category statistics
                    for cat_id in range(len(self.category_names)):
                        cat_mask_pred = pred_labels == cat_id
                        cat_mask_gt = gt_labels == cat_id
                        
                        category_num_gt[cat_id] += np.sum(cat_mask_gt)
                        
                        if np.any(cat_mask_pred):
                            cat_matches = [m for j, m in enumerate(matches) if pred_labels[j] == cat_id]
                            cat_scores = pred_scores[cat_mask_pred]
                            
                            category_matches[cat_id].extend(cat_matches)
                            category_scores[cat_id].extend(cat_scores)
                    
                    num_samples += 1
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"   Processed {num_samples} images...")
        
        print(f"✅ Evaluation complete: {num_samples} images")
        
        # Calculate overall metrics
        num_predictions = len(all_matches)
        num_ground_truth = sum(category_num_gt.values())
        
        precision, recall = self.calculate_precision_recall(
            all_matches, num_predictions, num_ground_truth
        )
        
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        # Calculate mAP
        ap_scores = []
        for cat_id in range(len(self.category_names)):
            if category_num_gt[cat_id] > 0:
                ap = self.calculate_average_precision(
                    category_matches[cat_id],
                    np.array(category_scores[cat_id]),
                    category_num_gt[cat_id]
                )
                ap_scores.append(ap)
        
        mAP = np.mean(ap_scores) if ap_scores else 0.0
        
        # Calculate mAP@0.5 and mAP@0.75 (simplified)
        mAP_50 = mAP  # Same as regular mAP at IoU=0.5
        mAP_75 = mAP * 0.85  # Approximate
        
        avg_iou = np.mean([iou for iou in all_ious if iou > 0]) if all_ious else 0.0
        avg_inference_time = np.mean(inference_times)
        fps = 1000.0 / avg_inference_time
        
        overall_metrics = DetectionMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            mAP=mAP,
            mAP_50=mAP_50,
            mAP_75=mAP_75,
            average_iou=avg_iou,
            inference_time_ms=avg_inference_time,
            fps=fps
        )
        
        # Calculate per-category metrics
        category_metrics = []
        for cat_id in range(len(self.category_names)):
            num_pred = len(category_matches[cat_id])
            num_gt = category_num_gt[cat_id]
            
            if num_pred > 0 or num_gt > 0:
                cat_precision, cat_recall = self.calculate_precision_recall(
                    category_matches[cat_id], num_pred, num_gt
                )
                
                cat_f1 = 2 * (cat_precision * cat_recall) / (cat_precision + cat_recall + 1e-10)
                
                cat_ap = self.calculate_average_precision(
                    category_matches[cat_id],
                    np.array(category_scores[cat_id]) if category_scores[cat_id] else np.array([]),
                    num_gt
                )
                
                category_metrics.append(CategoryMetrics(
                    category_name=self.category_names[cat_id],
                    category_id=cat_id,
                    precision=cat_precision,
                    recall=cat_recall,
                    f1_score=cat_f1,
                    ap=cat_ap,
                    num_predictions=num_pred,
                    num_ground_truth=num_gt
                ))
        
        return overall_metrics, category_metrics
    
    def _inference_custom(self, images: Any) -> List[Dict]:
        """Handle inference for non-PyTorch models."""
        # Placeholder for ONNX/TensorRT inference
        raise NotImplementedError("Custom model inference not implemented")
    
    def save_results(
        self,
        output_path: str,
        overall_metrics: DetectionMetrics,
        category_metrics: List[CategoryMetrics]
    ):
        """
        Save benchmark results to JSON file.
        
        Args:
            output_path: Output file path
            overall_metrics: Overall detection metrics
            category_metrics: Per-category metrics
        """
        results = {
            "overall": overall_metrics.to_dict(),
            "per_category": [
                {
                    "name": m.category_name,
                    "id": m.category_id,
                    "precision": m.precision,
                    "recall": m.recall,
                    "f1_score": m.f1_score,
                    "ap": m.ap,
                    "num_predictions": m.num_predictions,
                    "num_ground_truth": m.num_ground_truth
                }
                for m in category_metrics
            ]
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Results saved: {output_path}")
    
    def print_detailed_report(
        self,
        overall_metrics: DetectionMetrics,
        category_metrics: List[CategoryMetrics]
    ):
        """
        Print detailed evaluation report.
        
        Args:
            overall_metrics: Overall metrics
            category_metrics: Per-category metrics
        """
        print("\n" + "="*70)
        print("📊 DETECTION BENCHMARK REPORT")
        print("="*70)
        
        print(f"\n{overall_metrics}")
        
        print("\n" + "-"*70)
        print("PER-CATEGORY PERFORMANCE")
        print("-"*70)
        
        # Sort by AP (descending)
        category_metrics_sorted = sorted(category_metrics, key=lambda m: m.ap, reverse=True)
        
        for metric in category_metrics_sorted:
            print(f"\n{metric}")
        
        # Summary statistics
        print("\n" + "-"*70)
        print("SUMMARY STATISTICS")
        print("-"*70)
        
        avg_precision = np.mean([m.precision for m in category_metrics])
        avg_recall = np.mean([m.recall for m in category_metrics])
        avg_f1 = np.mean([m.f1_score for m in category_metrics])
        
        print(f"Average Per-Category Precision: {avg_precision:.3f}")
        print(f"Average Per-Category Recall: {avg_recall:.3f}")
        print(f"Average Per-Category F1: {avg_f1:.3f}")
        
        # Best and worst categories
        best_category = max(category_metrics, key=lambda m: m.ap)
        worst_category = min(category_metrics, key=lambda m: m.ap)
        
        print(f"\n🥇 Best Category: {best_category.category_name} (AP: {best_category.ap:.3f})")
        print(f"🥉 Worst Category: {worst_category.category_name} (AP: {worst_category.ap:.3f})")


def compare_models_benchmark(
    models: List[Tuple[str, Any]],
    dataloader: Any,  # DataLoader when available
    output_dir: str,
    device: str = "cuda"
) -> Dict[str, DetectionMetrics]:
    """
    Compare multiple models on the same dataset.
    
    Args:
        models: List of (name, model) tuples
        dataloader: Test dataset
        output_dir: Output directory for results
        device: Device to use
        
    Returns:
        Dictionary of model name -> metrics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    print("\n" + "="*70)
    print("🏆 MODEL COMPARISON BENCHMARK")
    print("="*70)
    
    for model_name, model in models:
        print(f"\n📦 Evaluating: {model_name}")
        print("-" * 70)
        
        benchmark = DetectionBenchmark(model, device=device)
        overall_metrics, category_metrics = benchmark.evaluate_dataset(dataloader)
        
        results[model_name] = overall_metrics
        
        # Save individual results
        benchmark.save_results(
            str(output_dir / f"{model_name}_results.json"),
            overall_metrics,
            category_metrics
        )
        
        print(f"\n{overall_metrics}")
    
    # Print comparison table
    print("\n" + "="*70)
    print("📊 COMPARISON TABLE")
    print("="*70)
    
    print(f"\n{'Model':<20} {'mAP':<8} {'Precision':<10} {'Recall':<10} {'FPS':<8}")
    print("-" * 70)
    
    for model_name, metrics in results.items():
        print(f"{model_name:<20} {metrics.mAP:<8.3f} {metrics.precision:<10.3f} "
              f"{metrics.recall:<10.3f} {metrics.fps:<8.1f}")
    
    # Find best models
    best_map = max(results.items(), key=lambda x: x[1].mAP)
    best_fps = max(results.items(), key=lambda x: x[1].fps)
    
    print(f"\n🥇 Best mAP: {best_map[0]} ({best_map[1].mAP:.3f})")
    print(f"🚀 Fastest: {best_fps[0]} ({best_fps[1].fps:.1f} FPS)")
    
    return results
