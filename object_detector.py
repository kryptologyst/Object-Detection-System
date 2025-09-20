"""
Modern Object Detection System
Supports YOLOv8, YOLOv9, EfficientDet, and SSD MobileNet models
"""

import os
import cv2
import numpy as np
import torch
import tensorflow as tf
import tensorflow_hub as hub
from ultralytics import YOLO
from PIL import Image
import yaml
from typing import List, Dict, Tuple, Optional, Union
import logging
from pathlib import Path
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ObjectDetector:
    """Modern object detection class supporting multiple models"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the detector with configuration"""
        self.config = self._load_config(config_path)
        self.models = {}
        self.current_model = None
        self.device = self._get_device()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'models': {
                'yolo': {'model_path': 'yolov8n.pt', 'confidence_threshold': 0.5},
                'efficientdet': {'model_path': 'efficientdet_d0', 'confidence_threshold': 0.5},
                'ssd_mobilenet': {'model_path': 'https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2', 'confidence_threshold': 0.5}
            },
            'image_processing': {'input_size': [640, 640], 'normalize': True},
            'performance': {'use_gpu': True, 'batch_size': 1}
        }
    
    def _get_device(self) -> str:
        """Determine the best available device"""
        if self.config['performance']['use_gpu'] and torch.cuda.is_available():
            return 'cuda'
        elif self.config['performance']['use_gpu'] and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'  # Apple Silicon
        else:
            return 'cpu'
    
    def load_model(self, model_type: str = 'yolo') -> bool:
        """Load a specific detection model"""
        try:
            if model_type == 'yolo':
                model_path = self.config['models']['yolo']['model_path']
                self.models['yolo'] = YOLO(model_path)
                self.current_model = 'yolo'
                logger.info(f"Loaded YOLO model: {model_path}")
                
            elif model_type == 'efficientdet':
                # EfficientDet implementation would go here
                logger.info("EfficientDet model loading not implemented yet")
                return False
                
            elif model_type == 'ssd_mobilenet':
                model_url = self.config['models']['ssd_mobilenet']['model_path']
                self.models['ssd_mobilenet'] = hub.load(model_url)
                self.current_model = 'ssd_mobilenet'
                logger.info(f"Loaded SSD MobileNet model from TensorFlow Hub")
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to load {model_type} model: {e}")
            return False
    
    def preprocess_image(self, image: Union[str, np.ndarray, Image.Image]) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess image for detection"""
        # Load image if path provided
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"Could not load image from {image}")
        
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image.copy()
        
        # Resize image
        input_size = self.config['image_processing']['input_size']
        resized = cv2.resize(image_rgb, tuple(input_size))
        
        # Normalize if required
        if self.config['image_processing']['normalize']:
            resized = resized.astype(np.float32) / 255.0
        
        return image_rgb, resized
    
    def detect_objects(self, image: Union[str, np.ndarray, Image.Image]) -> Dict:
        """Perform object detection on image"""
        if self.current_model is None:
            raise ValueError("No model loaded. Call load_model() first.")
        
        try:
            # Preprocess image
            original_image, processed_image = self.preprocess_image(image)
            
            # Run detection based on current model
            if self.current_model == 'yolo':
                results = self._detect_yolo(processed_image)
            elif self.current_model == 'ssd_mobilenet':
                results = self._detect_ssd_mobilenet(processed_image)
            else:
                raise ValueError(f"Detection not implemented for {self.current_model}")
            
            # Post-process results
            results = self._postprocess_results(results, original_image.shape)
            
            return {
                'image': original_image,
                'detections': results,
                'model': self.current_model,
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            raise
    
    def _detect_yolo(self, image: np.ndarray) -> List[Dict]:
        """Run YOLO detection"""
        model = self.models['yolo']
        confidence_threshold = self.config['models']['yolo']['confidence_threshold']
        
        # Run inference
        results = model(image, conf=confidence_threshold)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i in range(len(boxes)):
                    detection = {
                        'bbox': boxes.xyxy[i].cpu().numpy(),  # [x1, y1, x2, y2]
                        'confidence': float(boxes.conf[i].cpu().numpy()),
                        'class_id': int(boxes.cls[i].cpu().numpy()),
                        'class_name': model.names[int(boxes.cls[i].cpu().numpy())]
                    }
                    detections.append(detection)
        
        return detections
    
    def _detect_ssd_mobilenet(self, image: np.ndarray) -> List[Dict]:
        """Run SSD MobileNet detection"""
        model = self.models['ssd_mobilenet']
        confidence_threshold = self.config['models']['ssd_mobilenet']['confidence_threshold']
        
        # Prepare input tensor
        input_tensor = tf.expand_dims(image, axis=0)
        
        # Run inference
        result = model(input_tensor)
        result = {key: value.numpy() for key, value in result.items()}
        
        # Load COCO labels
        labels_path = tf.keras.utils.get_file(
            'mscoco_labels.txt',
            'https://storage.googleapis.com/download.tensorflow.org/data/mscoco_label_map.txt'
        )
        with open(labels_path, 'r') as f:
            labels = f.read().splitlines()
        
        detections = []
        num_detections = int(result['num_detections'][0])
        
        for i in range(min(num_detections, 100)):  # Limit to 100 detections
            confidence = result['detection_scores'][0][i]
            if confidence >= confidence_threshold:
                # Convert normalized coordinates to pixel coordinates
                bbox = result['detection_boxes'][0][i]  # [y1, x1, y2, x2]
                class_id = int(result['detection_classes'][0][i])
                
                detection = {
                    'bbox': [bbox[1], bbox[0], bbox[3], bbox[2]],  # Convert to [x1, y1, x2, y2]
                    'confidence': float(confidence),
                    'class_id': class_id,
                    'class_name': labels[class_id] if class_id < len(labels) else f"class_{class_id}"
                }
                detections.append(detection)
        
        return detections
    
    def _postprocess_results(self, detections: List[Dict], image_shape: Tuple) -> List[Dict]:
        """Post-process detection results"""
        height, width = image_shape[:2]
        
        for detection in detections:
            # Scale bounding boxes to original image size
            bbox = detection['bbox']
            detection['bbox_scaled'] = [
                int(bbox[0] * width),
                int(bbox[1] * height),
                int(bbox[2] * width),
                int(bbox[3] * height)
            ]
        
        return detections
    
    def draw_detections(self, image: np.ndarray, detections: List[Dict], 
                       show_confidence: bool = True, show_class_names: bool = True) -> np.ndarray:
        """Draw bounding boxes and labels on image"""
        output_image = image.copy()
        
        # Get colors from config
        bbox_color = tuple(self.config['ui']['bounding_box_color'])
        text_color = tuple(self.config['ui']['text_color'])
        
        for detection in detections:
            bbox = detection['bbox_scaled']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            # Draw bounding box
            cv2.rectangle(output_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), bbox_color, 2)
            
            # Prepare label text
            label_parts = []
            if show_class_names:
                label_parts.append(class_name)
            if show_confidence:
                label_parts.append(f"{confidence:.2f}")
            
            if label_parts:
                label = " ".join(label_parts)
                
                # Get text size for background rectangle
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 2
                (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                
                # Draw background rectangle for text
                cv2.rectangle(output_image, 
                            (bbox[0], bbox[1] - text_height - 10),
                            (bbox[0] + text_width, bbox[1]),
                            bbox_color, -1)
                
                # Draw text
                cv2.putText(output_image, label,
                          (bbox[0], bbox[1] - 5),
                          font, font_scale, text_color, thickness)
        
        return output_image
    
    def batch_detect(self, image_paths: List[str]) -> List[Dict]:
        """Perform batch detection on multiple images"""
        results = []
        
        for image_path in image_paths:
            try:
                result = self.detect_objects(image_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                results.append({'error': str(e), 'path': image_path})
        
        return results
    
    def get_model_info(self) -> Dict:
        """Get information about loaded models"""
        info = {
            'current_model': self.current_model,
            'device': self.device,
            'available_models': list(self.models.keys()),
            'config': self.config
        }
        return info


# Example usage and testing
if __name__ == "__main__":
    # Initialize detector
    detector = ObjectDetector()
    
    # Load YOLO model
    if detector.load_model('yolo'):
        print("YOLO model loaded successfully")
        
        # Test with a sample image (you'll need to provide an actual image)
        try:
            # This would work with an actual image file
            # result = detector.detect_objects('sample_image.jpg')
            # print(f"Detected {len(result['detections'])} objects")
            print("Object detector initialized successfully!")
        except Exception as e:
            print(f"Test detection failed: {e}")
    
    # Load SSD MobileNet model
    if detector.load_model('ssd_mobilenet'):
        print("SSD MobileNet model loaded successfully")
    
    print("Object detection system ready!")
