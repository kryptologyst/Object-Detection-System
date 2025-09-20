"""
Mock Database System for Object Detection Results
Provides SQLite-based storage for detection results, images, and metadata
"""

import sqlite3
import json
import os
import hashlib
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path
import base64
import io
from PIL import Image

logger = logging.getLogger(__name__)


class DetectionDatabase:
    """SQLite database for storing object detection results"""
    
    def __init__(self, db_path: str = "detection_results.db"):
        """Initialize database connection"""
        self.db_path = db_path
        self.conn = None
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            cursor = self.conn.cursor()
            
            # Create images table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    file_path TEXT,
                    file_hash TEXT UNIQUE,
                    image_data BLOB,
                    width INTEGER,
                    height INTEGER,
                    channels INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create detections table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_id INTEGER,
                    model_name TEXT NOT NULL,
                    class_name TEXT NOT NULL,
                    class_id INTEGER,
                    confidence REAL,
                    bbox_x1 REAL,
                    bbox_y1 REAL,
                    bbox_x2 REAL,
                    bbox_y2 REAL,
                    detection_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (image_id) REFERENCES images (id)
                )
            ''')
            
            # Create model_performance table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    total_detections INTEGER DEFAULT 0,
                    avg_confidence REAL,
                    processing_time_ms REAL,
                    last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_image_hash ON images(file_hash)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_detection_image ON detections(image_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_detection_model ON detections(model_name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_detection_class ON detections(class_name)')
            
            self.conn.commit()
            logger.info(f"Database initialized: {self.db_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def add_image(self, image_path: str, store_image_data: bool = False) -> int:
        """Add image to database and return image ID"""
        try:
            # Calculate file hash
            with open(image_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            
            # Check if image already exists
            existing_id = self.get_image_by_hash(file_hash)
            if existing_id:
                return existing_id
            
            # Get image properties
            with Image.open(image_path) as img:
                width, height = img.size
                channels = len(img.getbands()) if hasattr(img, 'getbands') else 3
            
            # Read image data if requested
            image_data = None
            if store_image_data:
                with open(image_path, 'rb') as f:
                    image_data = f.read()
            
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO images (filename, file_path, file_hash, image_data, width, height, channels)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (os.path.basename(image_path), image_path, file_hash, image_data, width, height, channels))
            
            image_id = cursor.lastrowid
            self.conn.commit()
            
            logger.info(f"Added image to database: {image_path} (ID: {image_id})")
            return image_id
            
        except Exception as e:
            logger.error(f"Failed to add image {image_path}: {e}")
            raise
    
    def add_detection(self, image_id: int, model_name: str, detection_data: Dict) -> int:
        """Add detection result to database"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO detections 
                (image_id, model_name, class_name, class_id, confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                image_id,
                model_name,
                detection_data['class_name'],
                detection_data['class_id'],
                detection_data['confidence'],
                detection_data['bbox'][0],
                detection_data['bbox'][1],
                detection_data['bbox'][2],
                detection_data['bbox'][3]
            ))
            
            detection_id = cursor.lastrowid
            self.conn.commit()
            
            return detection_id
            
        except Exception as e:
            logger.error(f"Failed to add detection: {e}")
            raise
    
    def add_detection_batch(self, image_id: int, model_name: str, detections: List[Dict]) -> List[int]:
        """Add multiple detections for an image"""
        detection_ids = []
        
        try:
            cursor = self.conn.cursor()
            
            for detection in detections:
                cursor.execute('''
                    INSERT INTO detections 
                    (image_id, model_name, class_name, class_id, confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    image_id,
                    model_name,
                    detection['class_name'],
                    detection['class_id'],
                    detection['confidence'],
                    detection['bbox'][0],
                    detection['bbox'][1],
                    detection['bbox'][2],
                    detection['bbox'][3]
                ))
                detection_ids.append(cursor.lastrowid)
            
            self.conn.commit()
            logger.info(f"Added {len(detections)} detections for image {image_id}")
            
        except Exception as e:
            logger.error(f"Failed to add detection batch: {e}")
            raise
        
        return detection_ids
    
    def get_image_by_hash(self, file_hash: str) -> Optional[int]:
        """Get image ID by file hash"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT id FROM images WHERE file_hash = ?', (file_hash,))
        result = cursor.fetchone()
        return result[0] if result else None
    
    def get_detections_by_image(self, image_id: int) -> List[Dict]:
        """Get all detections for an image"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT id, model_name, class_name, class_id, confidence, 
                   bbox_x1, bbox_y1, bbox_x2, bbox_y2, detection_time
            FROM detections WHERE image_id = ?
            ORDER BY confidence DESC
        ''', (image_id,))
        
        detections = []
        for row in cursor.fetchall():
            detection = {
                'id': row[0],
                'model_name': row[1],
                'class_name': row[2],
                'class_id': row[3],
                'confidence': row[4],
                'bbox': [row[5], row[6], row[7], row[8]],
                'detection_time': row[9]
            }
            detections.append(detection)
        
        return detections
    
    def get_detection_statistics(self) -> Dict:
        """Get overall detection statistics"""
        cursor = self.conn.cursor()
        
        # Total images and detections
        cursor.execute('SELECT COUNT(*) FROM images')
        total_images = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM detections')
        total_detections = cursor.fetchone()[0]
        
        # Detections by model
        cursor.execute('''
            SELECT model_name, COUNT(*) as count, AVG(confidence) as avg_conf
            FROM detections GROUP BY model_name
        ''')
        model_stats = {}
        for row in cursor.fetchall():
            model_stats[row[0]] = {
                'count': row[1],
                'avg_confidence': row[2]
            }
        
        # Top detected classes
        cursor.execute('''
            SELECT class_name, COUNT(*) as count, AVG(confidence) as avg_conf
            FROM detections GROUP BY class_name ORDER BY count DESC LIMIT 10
        ''')
        top_classes = []
        for row in cursor.fetchall():
            top_classes.append({
                'class_name': row[0],
                'count': row[1],
                'avg_confidence': row[2]
            })
        
        return {
            'total_images': total_images,
            'total_detections': total_detections,
            'model_statistics': model_stats,
            'top_classes': top_classes
        }
    
    def search_detections(self, class_name: Optional[str] = None, 
                         model_name: Optional[str] = None,
                         min_confidence: Optional[float] = None,
                         limit: int = 100) -> List[Dict]:
        """Search detections with filters"""
        cursor = self.conn.cursor()
        
        query = '''
            SELECT d.id, d.image_id, d.model_name, d.class_name, d.class_id, 
                   d.confidence, d.bbox_x1, d.bbox_y1, d.bbox_x2, d.bbox_y2,
                   d.detection_time, i.filename, i.file_path
            FROM detections d
            JOIN images i ON d.image_id = i.id
            WHERE 1=1
        '''
        params = []
        
        if class_name:
            query += ' AND d.class_name = ?'
            params.append(class_name)
        
        if model_name:
            query += ' AND d.model_name = ?'
            params.append(model_name)
        
        if min_confidence:
            query += ' AND d.confidence >= ?'
            params.append(min_confidence)
        
        query += ' ORDER BY d.confidence DESC LIMIT ?'
        params.append(limit)
        
        cursor.execute(query, params)
        
        results = []
        for row in cursor.fetchall():
            result = {
                'detection_id': row[0],
                'image_id': row[1],
                'model_name': row[2],
                'class_name': row[3],
                'class_id': row[4],
                'confidence': row[5],
                'bbox': [row[6], row[7], row[8], row[9]],
                'detection_time': row[10],
                'filename': row[11],
                'file_path': row[12]
            }
            results.append(result)
        
        return results
    
    def update_model_performance(self, model_name: str, processing_time_ms: float, 
                                detection_count: int, avg_confidence: float):
        """Update model performance metrics"""
        cursor = self.conn.cursor()
        
        # Check if record exists
        cursor.execute('SELECT id FROM model_performance WHERE model_name = ?', (model_name,))
        existing = cursor.fetchone()
        
        if existing:
            cursor.execute('''
                UPDATE model_performance 
                SET total_detections = total_detections + ?, 
                    avg_confidence = ?, 
                    processing_time_ms = ?,
                    last_used = CURRENT_TIMESTAMP
                WHERE model_name = ?
            ''', (detection_count, avg_confidence, processing_time_ms, model_name))
        else:
            cursor.execute('''
                INSERT INTO model_performance 
                (model_name, total_detections, avg_confidence, processing_time_ms)
                VALUES (?, ?, ?, ?)
            ''', (model_name, detection_count, avg_confidence, processing_time_ms))
        
        self.conn.commit()
    
    def get_model_performance(self) -> List[Dict]:
        """Get model performance statistics"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT model_name, total_detections, avg_confidence, 
                   processing_time_ms, last_used
            FROM model_performance
            ORDER BY total_detections DESC
        ''')
        
        performance = []
        for row in cursor.fetchall():
            performance.append({
                'model_name': row[0],
                'total_detections': row[1],
                'avg_confidence': row[2],
                'processing_time_ms': row[3],
                'last_used': row[4]
            })
        
        return performance
    
    def cleanup_old_data(self, days_old: int = 30):
        """Remove old detection data"""
        cursor = self.conn.cursor()
        
        # Delete old detections
        cursor.execute('''
            DELETE FROM detections 
            WHERE detection_time < datetime('now', '-{} days')
        '''.format(days_old))
        
        deleted_count = cursor.rowcount
        self.conn.commit()
        
        logger.info(f"Cleaned up {deleted_count} old detections")
        return deleted_count
    
    def export_data(self, output_file: str, format: str = 'json'):
        """Export detection data to file"""
        cursor = self.conn.cursor()
        
        # Get all detections with image info
        cursor.execute('''
            SELECT d.*, i.filename, i.file_path, i.width, i.height
            FROM detections d
            JOIN images i ON d.image_id = i.id
            ORDER BY d.detection_time DESC
        ''')
        
        data = []
        for row in cursor.fetchall():
            record = {
                'detection_id': row[0],
                'image_id': row[1],
                'model_name': row[2],
                'class_name': row[3],
                'class_id': row[4],
                'confidence': row[5],
                'bbox': [row[6], row[7], row[8], row[9]],
                'detection_time': row[10],
                'filename': row[11],
                'file_path': row[12],
                'image_width': row[13],
                'image_height': row[14]
            }
            data.append(record)
        
        if format.lower() == 'json':
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        elif format.lower() == 'csv':
            import pandas as pd
            df = pd.DataFrame(data)
            df.to_csv(output_file, index=False)
        
        logger.info(f"Exported {len(data)} records to {output_file}")
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")


# Mock data generator for testing
class MockDataGenerator:
    """Generate mock detection data for testing and demonstration"""
    
    def __init__(self, db: DetectionDatabase):
        self.db = db
        self.sample_classes = [
            'person', 'car', 'truck', 'bus', 'motorcycle', 'bicycle',
            'dog', 'cat', 'bird', 'horse', 'cow', 'elephant',
            'chair', 'table', 'laptop', 'phone', 'book', 'bottle',
            'cup', 'banana', 'apple', 'sandwich', 'pizza', 'cake'
        ]
        self.sample_models = ['yolo', 'ssd_mobilenet', 'efficientdet']
    
    def generate_sample_images(self, count: int = 10) -> List[int]:
        """Generate sample image records"""
        image_ids = []
        
        for i in range(count):
            # Create a mock image file
            filename = f"sample_image_{i+1}.jpg"
            file_path = f"sample_images/{filename}"
            
            # Create directory if it doesn't exist
            os.makedirs("sample_images", exist_ok=True)
            
            # Generate a simple colored image
            img = Image.new('RGB', (640, 480), color=(i*25, (i*50)%255, (i*75)%255))
            img.save(file_path)
            
            # Add to database
            image_id = self.db.add_image(file_path)
            image_ids.append(image_id)
        
        return image_ids
    
    def generate_sample_detections(self, image_ids: List[int], detections_per_image: int = 3):
        """Generate sample detection records"""
        import random
        
        for image_id in image_ids:
            detections = []
            
            for _ in range(random.randint(1, detections_per_image)):
                detection = {
                    'class_name': random.choice(self.sample_classes),
                    'class_id': random.randint(0, 79),  # COCO classes
                    'confidence': random.uniform(0.3, 0.95),
                    'bbox': [
                        random.uniform(0, 0.7),  # x1
                        random.uniform(0, 0.7),  # y1
                        random.uniform(0.3, 1.0),  # x2
                        random.uniform(0.3, 1.0)   # y2
                    ]
                }
                detections.append(detection)
            
            model_name = random.choice(self.sample_models)
            self.db.add_detection_batch(image_id, model_name, detections)
            
            # Update model performance
            avg_conf = sum(d['confidence'] for d in detections) / len(detections)
            processing_time = random.uniform(50, 500)  # ms
            self.db.update_model_performance(model_name, processing_time, len(detections), avg_conf)


# Example usage
if __name__ == "__main__":
    # Initialize database
    db = DetectionDatabase("test_detection.db")
    
    # Generate mock data
    generator = MockDataGenerator(db)
    
    print("Generating sample data...")
    image_ids = generator.generate_sample_images(5)
    generator.generate_sample_detections(image_ids)
    
    # Get statistics
    stats = db.get_detection_statistics()
    print(f"Database Statistics:")
    print(f"Total Images: {stats['total_images']}")
    print(f"Total Detections: {stats['total_detections']}")
    print(f"Model Statistics: {stats['model_statistics']}")
    
    # Search example
    results = db.search_detections(class_name='person', min_confidence=0.5, limit=5)
    print(f"Found {len(results)} person detections with confidence >= 0.5")
    
    # Export data
    db.export_data("detection_export.json", "json")
    
    db.close()
    print("Mock database setup complete!")
