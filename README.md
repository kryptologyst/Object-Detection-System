# Object Detection System

A comprehensive, production-ready object detection system supporting multiple state-of-the-art AI models with a modern web interface, database storage, and advanced analytics.

## Features

- **Multiple AI Models**: Support for YOLOv8, YOLOv9, SSD MobileNet, and EfficientDet
- **Modern Web UI**: Interactive Streamlit interface with real-time detection
- **Database Integration**: SQLite-based storage with comprehensive analytics
- **Performance Metrics**: Detailed evaluation tools and benchmarking
- **GPU Acceleration**: Automatic GPU detection and utilization
- **Batch Processing**: Efficient processing of multiple images
- **Configuration Management**: YAML-based configuration system
- **Analytics Dashboard**: Comprehensive statistics and visualizations
- **Export Capabilities**: JSON/CSV export functionality
- **Search & Filter**: Advanced database search and filtering

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd object-detection-system

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Web Interface

```bash
streamlit run app.py
```

### 3. Or Use Programmatically

```python
from object_detector import ObjectDetector

# Initialize detector
detector = ObjectDetector()

# Load YOLOv8 model
detector.load_model('yolo')

# Detect objects in an image
result = detector.detect_objects('path/to/your/image.jpg')

# Display results
print(f"Found {len(result['detections'])} objects")
for detection in result['detections']:
    print(f"- {detection['class_name']}: {detection['confidence']:.3f}")
```

## 📁 Project Structure

```
object-detection-system/
├── 0096.py                 # Main demo script
├── object_detector.py      # Core detection functionality
├── database.py            # Database operations
├── app.py                 # Streamlit web interface
├── evaluator.py           # Performance evaluation tools
├── config.yaml           # Configuration settings
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── sample_images/       # Sample images for testing
```

## Core Components

### ObjectDetector Class

The main detection engine supporting multiple models:

```python
from object_detector import ObjectDetector

detector = ObjectDetector("config.yaml")

# Load different models
detector.load_model('yolo')        # YOLOv8/YOLOv9
detector.load_model('ssd_mobilenet')  # SSD MobileNet
detector.load_model('efficientdet')  # EfficientDet

# Detect objects
result = detector.detect_objects(image_path)

# Batch processing
results = detector.batch_detect(['img1.jpg', 'img2.jpg', 'img3.jpg'])
```

### Database System

Comprehensive storage and analytics:

```python
from database import DetectionDatabase, MockDataGenerator

# Initialize database
db = DetectionDatabase("detection_results.db")

# Add detection results
image_id = db.add_image("path/to/image.jpg")
db.add_detection_batch(image_id, "yolo", detections)

# Search and analytics
stats = db.get_detection_statistics()
results = db.search_detections(class_name="person", min_confidence=0.5)
```

### Performance Evaluation

Advanced benchmarking and evaluation:

```python
from evaluator import DetectionEvaluator

evaluator = DetectionEvaluator()

# Benchmark model performance
results = evaluator.benchmark_model(detector, test_images, ground_truth)

# Generate comprehensive report
evaluator.generate_performance_report("report.html")
```

## Web Interface

The Streamlit web interface provides:

- **Image Upload**: Drag-and-drop image upload
- **Model Selection**: Easy switching between detection models
- **Parameter Tuning**: Adjustable confidence thresholds
- **Real-time Results**: Instant detection visualization
- **Analytics Dashboard**: Comprehensive statistics
- **Database Management**: Search and export functionality

### Interface Features

1. **Detection Tab**: Upload images and run object detection
2. **Analytics Tab**: View performance metrics and statistics
3. **Database Tab**: Search, filter, and manage detection data
4. **About Tab**: System information and documentation

## Configuration

Customize the system through `config.yaml`:

```yaml
models:
  yolo:
    model_path: "yolov8n.pt"
    confidence_threshold: 0.5
    iou_threshold: 0.45
  
  ssd_mobilenet:
    model_path: "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"
    confidence_threshold: 0.5

image_processing:
  input_size: [640, 640]
  normalize: true

performance:
  use_gpu: true
  batch_size: 1
```

## Supported Models

| Model | Type | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| YOLOv8 | Real-time | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | General purpose |
| YOLOv9 | Real-time | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | High accuracy |
| SSD MobileNet | Mobile | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Mobile/Edge |
| EfficientDet | Efficient | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Balanced |

## 🔧 Advanced Usage

### Custom Model Integration

```python
class CustomDetector(ObjectDetector):
    def load_custom_model(self, model_path):
        # Implement your custom model loading
        pass
    
    def _detect_custom(self, image):
        # Implement your custom detection logic
        pass
```

### Batch Processing Pipeline

```python
# Process large datasets
image_paths = glob.glob("dataset/*.jpg")
results = detector.batch_detect(image_paths)

# Save results to database
for result in results:
    if 'error' not in result:
        image_id = db.add_image(result['image_path'])
        db.add_detection_batch(image_id, 'yolo', result['detections'])
```

### Performance Optimization

```python
# Enable GPU acceleration
detector.config['performance']['use_gpu'] = True

# Optimize batch size
detector.config['performance']['batch_size'] = 4

# Cache model predictions
detector.config['performance']['cache_predictions'] = True
```

## Performance Metrics

The system tracks comprehensive metrics:

- **FPS**: Frames per second processing speed
- **mAP**: Mean Average Precision for accuracy
- **IoU**: Intersection over Union for localization
- **Confidence**: Detection confidence scores
- **Processing Time**: Per-image processing duration

## Database Schema

### Images Table
- `id`: Primary key
- `filename`: Image filename
- `file_path`: Full file path
- `file_hash`: MD5 hash for deduplication
- `width`, `height`, `channels`: Image properties
- `created_at`: Timestamp

### Detections Table
- `id`: Primary key
- `image_id`: Foreign key to images
- `model_name`: Detection model used
- `class_name`: Detected object class
- `class_id`: Numeric class identifier
- `confidence`: Detection confidence score
- `bbox_x1`, `bbox_y1`, `bbox_x2`, `bbox_y2`: Bounding box coordinates
- `detection_time`: Timestamp

### Model Performance Table
- `id`: Primary key
- `model_name`: Model identifier
- `total_detections`: Total detections count
- `avg_confidence`: Average confidence score
- `processing_time_ms`: Average processing time
- `last_used`: Last usage timestamp

## Testing

Run the comprehensive demo:

```bash
python 0096.py
```

This will:
1. Initialize the detection system
2. Demonstrate basic detection
3. Show database integration
4. Test batch processing
5. Generate sample data

## Example Results

### Detection Output
```
MODERN OBJECT DETECTION SYSTEM DEMO
============================================================
Initializing Modern Object Detection System...
YOLOv8 model loaded successfully!
Running object detection...
Detection completed!
Found 3 objects
Processing time: 45.2 ms
Model used: yolo
  1. person (confidence: 0.892)
  2. car (confidence: 0.756)
  3. dog (confidence: 0.634)
Result saved as 'detection_result.jpg'
```

### Performance Metrics
```
Database Statistics:
   Total Images: 15
   Total Detections: 47
   Model Statistics: {'yolo': {'count': 32, 'avg_confidence': 0.78}, 'ssd_mobilenet': {'count': 15, 'avg_confidence': 0.72}}
```

## Future Enhancements

- [ ] Real-time video processing
- [ ] Custom model training pipeline
- [ ] REST API endpoints
- [ ] Docker containerization
- [ ] Cloud deployment scripts
- [ ] Advanced visualization tools
- [ ] Model ensemble methods
- [ ] Edge device optimization

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8/YOLOv9
- [TensorFlow Hub](https://tfhub.dev/) for pre-trained models
- [Streamlit](https://streamlit.io/) for the web interface
- [OpenCV](https://opencv.org/) for computer vision operations

## Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Check the documentation
- Review the example code


# Object-Detection-System
