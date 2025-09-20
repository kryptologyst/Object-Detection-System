"""
Performance Metrics and Evaluation Tools
Provides comprehensive evaluation metrics for object detection models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import time
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


class DetectionEvaluator:
    """Comprehensive evaluation system for object detection models"""
    
    def __init__(self):
        self.results = []
        self.benchmark_data = []
    
    def calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
        x1_max = max(box1[0], box2[0])
        y1_max = max(box1[1], box2[1])
        x2_min = min(box1[2], box2[2])
        y2_min = min(box1[3], box2[3])
        
        if x2_min <= x1_max or y2_min <= y1_max:
            return 0.0
        
        intersection = (x2_min - x1_max) * (y2_min - y1_max)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_precision_recall(self, predictions: List[Dict], ground_truth: List[Dict], 
                                 iou_threshold: float = 0.5) -> Tuple[float, float]:
        """Calculate precision and recall for detections"""
        if not predictions and not ground_truth:
            return 1.0, 1.0
        if not predictions:
            return 0.0, 0.0
        if not ground_truth:
            return 0.0, 1.0
        
        # Sort predictions by confidence
        sorted_predictions = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
        
        true_positives = 0
        false_positives = 0
        matched_gt = set()
        
        for pred in sorted_predictions:
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(ground_truth):
                if gt_idx in matched_gt:
                    continue
                
                if pred['class_name'] == gt['class_name']:
                    iou = self.calculate_iou(pred['bbox'], gt['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold:
                true_positives += 1
                matched_gt.add(best_gt_idx)
            else:
                false_positives += 1
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / len(ground_truth) if len(ground_truth) > 0 else 0
        
        return precision, recall
    
    def calculate_map(self, all_predictions: List[Dict], all_ground_truth: List[Dict], 
                     iou_threshold: float = 0.5) -> float:
        """Calculate Mean Average Precision (mAP)"""
        # Group by class
        classes = set()
        for pred in all_predictions:
            classes.add(pred['class_name'])
        for gt in all_ground_truth:
            classes.add(gt['class_name'])
        
        ap_scores = []
        
        for class_name in classes:
            class_predictions = [p for p in all_predictions if p['class_name'] == class_name]
            class_ground_truth = [g for g in all_ground_truth if g['class_name'] == class_name]
            
            if not class_ground_truth:
                continue
            
            precision, recall = self.calculate_precision_recall(
                class_predictions, class_ground_truth, iou_threshold
            )
            
            # Calculate AP using 11-point interpolation
            ap = self._calculate_ap_11_point(class_predictions, class_ground_truth, iou_threshold)
            ap_scores.append(ap)
        
        return np.mean(ap_scores) if ap_scores else 0.0
    
    def _calculate_ap_11_point(self, predictions: List[Dict], ground_truth: List[Dict], 
                              iou_threshold: float) -> float:
        """Calculate AP using 11-point interpolation"""
        if not predictions:
            return 0.0
        
        # Sort predictions by confidence
        sorted_predictions = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
        
        precisions = []
        recalls = []
        
        for i in range(len(sorted_predictions)):
            subset = sorted_predictions[:i+1]
            precision, recall = self.calculate_precision_recall(subset, ground_truth, iou_threshold)
            precisions.append(precision)
            recalls.append(recall)
        
        # 11-point interpolation
        recall_points = np.linspace(0, 1, 11)
        interpolated_precisions = []
        
        for r in recall_points:
            max_precision = 0
            for i, recall_val in enumerate(recalls):
                if recall_val >= r:
                    max_precision = max(max_precision, precisions[i])
            interpolated_precisions.append(max_precision)
        
        return np.mean(interpolated_precisions)
    
    def benchmark_model(self, detector, test_images: List[str], 
                       ground_truth: Optional[List[Dict]] = None) -> Dict:
        """Benchmark a model on test images"""
        results = {
            'model_name': detector.current_model,
            'total_images': len(test_images),
            'processing_times': [],
            'detection_counts': [],
            'confidences': [],
            'class_distribution': {},
            'errors': 0
        }
        
        start_time = time.time()
        
        for i, image_path in enumerate(test_images):
            try:
                img_start = time.time()
                result = detector.detect_objects(image_path)
                img_time = (time.time() - img_start) * 1000
                
                results['processing_times'].append(img_time)
                results['detection_counts'].append(len(result['detections']))
                
                for detection in result['detections']:
                    results['confidences'].append(detection['confidence'])
                    class_name = detection['class_name']
                    results['class_distribution'][class_name] = results['class_distribution'].get(class_name, 0) + 1
                
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                results['errors'] += 1
        
        total_time = time.time() - start_time
        
        # Calculate statistics
        results['total_processing_time'] = total_time
        results['avg_processing_time'] = np.mean(results['processing_times']) if results['processing_times'] else 0
        results['avg_detections_per_image'] = np.mean(results['detection_counts']) if results['detection_counts'] else 0
        results['avg_confidence'] = np.mean(results['confidences']) if results['confidences'] else 0
        results['fps'] = len(test_images) / total_time if total_time > 0 else 0
        
        # Calculate mAP if ground truth provided
        if ground_truth:
            all_predictions = []
            for result in results.get('detection_results', []):
                all_predictions.extend(result['detections'])
            
            results['map'] = self.calculate_map(all_predictions, ground_truth)
        
        self.benchmark_data.append(results)
        return results
    
    def compare_models(self, model_results: List[Dict]) -> pd.DataFrame:
        """Compare performance across multiple models"""
        comparison_data = []
        
        for result in model_results:
            comparison_data.append({
                'Model': result['model_name'],
                'FPS': result['fps'],
                'Avg Processing Time (ms)': result['avg_processing_time'],
                'Avg Detections/Image': result['avg_detections_per_image'],
                'Avg Confidence': result['avg_confidence'],
                'Errors': result['errors'],
                'Total Images': result['total_images']
            })
        
        df = pd.DataFrame(comparison_data)
        return df.sort_values('FPS', ascending=False)
    
    def generate_performance_report(self, output_file: str = "performance_report.html"):
        """Generate comprehensive performance report"""
        if not self.benchmark_data:
            logger.warning("No benchmark data available")
            return
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Processing time comparison
        models = [r['model_name'] for r in self.benchmark_data]
        fps_values = [r['fps'] for r in self.benchmark_data]
        
        axes[0, 0].bar(models, fps_values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[0, 0].set_title('FPS Comparison')
        axes[0, 0].set_ylabel('Frames Per Second')
        
        # Average confidence comparison
        conf_values = [r['avg_confidence'] for r in self.benchmark_data]
        axes[0, 1].bar(models, conf_values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[0, 1].set_title('Average Confidence')
        axes[0, 1].set_ylabel('Confidence Score')
        
        # Detection count distribution
        all_counts = []
        for result in self.benchmark_data:
            all_counts.extend(result['detection_counts'])
        
        axes[1, 0].hist(all_counts, bins=20, alpha=0.7, color='#2ca02c')
        axes[1, 0].set_title('Detection Count Distribution')
        axes[1, 0].set_xlabel('Detections per Image')
        axes[1, 0].set_ylabel('Frequency')
        
        # Processing time distribution
        all_times = []
        for result in self.benchmark_data:
            all_times.extend(result['processing_times'])
        
        axes[1, 1].hist(all_times, bins=20, alpha=0.7, color='#ff7f0e')
        axes[1, 1].set_title('Processing Time Distribution')
        axes[1, 1].set_xlabel('Processing Time (ms)')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Generate HTML report
        html_content = self._generate_html_report()
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Performance report saved to {output_file}")
    
    def _generate_html_report(self) -> str:
        """Generate HTML performance report"""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Object Detection Performance Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { color: #1f77b4; text-align: center; }
                .metric { background-color: #f0f2f6; padding: 20px; margin: 10px 0; border-radius: 5px; }
                table { width: 100%; border-collapse: collapse; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <h1 class="header">🔍 Object Detection Performance Report</h1>
            <p>Generated on: {}</p>
        """.format(time.strftime("%Y-%m-%d %H:%M:%S"))
        
        # Add benchmark results
        for result in self.benchmark_data:
            html += f"""
            <div class="metric">
                <h2>Model: {result['model_name'].upper()}</h2>
                <p><strong>FPS:</strong> {result['fps']:.2f}</p>
                <p><strong>Average Processing Time:</strong> {result['avg_processing_time']:.2f} ms</p>
                <p><strong>Average Detections per Image:</strong> {result['avg_detections_per_image']:.2f}</p>
                <p><strong>Average Confidence:</strong> {result['avg_confidence']:.3f}</p>
                <p><strong>Total Images Processed:</strong> {result['total_images']}</p>
                <p><strong>Errors:</strong> {result['errors']}</p>
            </div>
            """
        
        html += """
        </body>
        </html>
        """
        
        return html
    
    def export_results(self, output_file: str = "evaluation_results.json"):
        """Export evaluation results to JSON"""
        with open(output_file, 'w') as f:
            json.dump(self.benchmark_data, f, indent=2, default=str)
        
        logger.info(f"Evaluation results exported to {output_file}")


# Example usage and testing
if __name__ == "__main__":
    from object_detector import ObjectDetector
    
    # Initialize evaluator
    evaluator = DetectionEvaluator()
    
    # Create sample test data
    detector = ObjectDetector()
    detector.load_model('yolo')
    
    # Sample ground truth data
    sample_ground_truth = [
        {'class_name': 'person', 'bbox': [100, 100, 200, 300], 'confidence': 1.0},
        {'class_name': 'car', 'bbox': [300, 150, 450, 250], 'confidence': 1.0}
    ]
    
    # Sample test images (you would use real image paths)
    test_images = ['sample_image_1.jpg', 'sample_image_2.jpg']  # Replace with actual paths
    
    print("🔍 Running performance evaluation...")
    
    # Benchmark the model
    results = evaluator.benchmark_model(detector, test_images, sample_ground_truth)
    
    print(f"✅ Evaluation completed!")
    print(f"📊 Results:")
    print(f"   FPS: {results['fps']:.2f}")
    print(f"   Average Processing Time: {results['avg_processing_time']:.2f} ms")
    print(f"   Average Detections per Image: {results['avg_detections_per_image']:.2f}")
    print(f"   Average Confidence: {results['avg_confidence']:.3f}")
    
    # Generate report
    evaluator.generate_performance_report()
    
    print("📈 Performance evaluation complete!")
