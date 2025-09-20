"""
Modern Streamlit Web UI for Object Detection System
Interactive interface for uploading images, running detections, and viewing results
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io
import time
import os
from pathlib import Path
import json

# Import our custom modules
from object_detector import ObjectDetector
from database import DetectionDatabase, MockDataGenerator

# Page configuration
st.set_page_config(
    page_title="Object Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .detection-result {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'detector' not in st.session_state:
    st.session_state.detector = None
if 'db' not in st.session_state:
    st.session_state.db = None
if 'detection_results' not in st.session_state:
    st.session_state.detection_results = []

def initialize_system():
    """Initialize the detection system and database"""
    try:
        # Initialize detector
        if st.session_state.detector is None:
            with st.spinner("Loading object detection models..."):
                st.session_state.detector = ObjectDetector()
                st.session_state.detector.load_model('yolo')
        
        # Initialize database
        if st.session_state.db is None:
            st.session_state.db = DetectionDatabase("detection_results.db")
            
            # Generate mock data if database is empty
            stats = st.session_state.db.get_detection_statistics()
            if stats['total_images'] == 0:
                with st.spinner("Generating sample data..."):
                    generator = MockDataGenerator(st.session_state.db)
                    generator.generate_sample_images(10)
                    generator.generate_sample_detections(list(range(1, 11)))
        
        return True
    except Exception as e:
        st.error(f"Failed to initialize system: {e}")
        return False

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">🔍 Object Detection System</h1>', unsafe_allow_html=True)
    
    # Initialize system
    if not initialize_system():
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Controls")
        
        # Model selection
        model_options = ['yolo', 'ssd_mobilenet']
        selected_model = st.selectbox(
            "Select Detection Model",
            model_options,
            index=0
        )
        
        # Load selected model
        if st.button("🔄 Load Model"):
            with st.spinner(f"Loading {selected_model} model..."):
                success = st.session_state.detector.load_model(selected_model)
                if success:
                    st.success(f"✅ {selected_model.upper()} model loaded!")
                else:
                    st.error(f"❌ Failed to load {selected_model} model")
        
        # Detection parameters
        st.subheader("⚙️ Detection Parameters")
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.05
        )
        
        show_confidence = st.checkbox("Show Confidence Scores", value=True)
        show_class_names = st.checkbox("Show Class Names", value=True)
        
        # Update config
        st.session_state.detector.config['models'][selected_model]['confidence_threshold'] = confidence_threshold
        st.session_state.detector.config['ui']['show_confidence'] = show_confidence
        st.session_state.detector.config['ui']['show_class_names'] = show_class_names
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["📸 Detection", "📊 Analytics", "🗄️ Database", "ℹ️ About"])
    
    with tab1:
        detection_tab()
    
    with tab2:
        analytics_tab()
    
    with tab3:
        database_tab()
    
    with tab4:
        about_tab()

def detection_tab():
    """Object detection interface"""
    st.header("📸 Object Detection")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload an image to detect objects"
    )
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Detection button
            if st.button("🔍 Detect Objects", type="primary"):
                with st.spinner("Running object detection..."):
                    try:
                        # Convert PIL to numpy array
                        img_array = np.array(image)
                        
                        # Run detection
                        start_time = time.time()
                        result = st.session_state.detector.detect_objects(img_array)
                        processing_time = (time.time() - start_time) * 1000
                        
                        # Draw detections
                        output_image = st.session_state.detector.draw_detections(
                            result['image'], 
                            result['detections'],
                            show_confidence=st.session_state.detector.config['ui']['show_confidence'],
                            show_class_names=st.session_state.detector.config['ui']['show_class_names']
                        )
                        
                        # Store results
                        st.session_state.detection_results.append({
                            'image': output_image,
                            'detections': result['detections'],
                            'model': result['model'],
                            'processing_time': processing_time,
                            'timestamp': result['timestamp']
                        })
                        
                        # Save to database
                        image_path = f"temp_upload_{int(time.time())}.jpg"
                        image.save(image_path)
                        image_id = st.session_state.db.add_image(image_path)
                        st.session_state.db.add_detection_batch(
                            image_id, 
                            result['model'], 
                            result['detections']
                        )
                        
                        # Update model performance
                        if result['detections']:
                            avg_conf = sum(d['confidence'] for d in result['detections']) / len(result['detections'])
                            st.session_state.db.update_model_performance(
                                result['model'], 
                                processing_time, 
                                len(result['detections']), 
                                avg_conf
                            )
                        
                        # Clean up temp file
                        os.remove(image_path)
                        
                        st.success(f"✅ Detection completed! Found {len(result['detections'])} objects in {processing_time:.1f}ms")
                        
                    except Exception as e:
                        st.error(f"❌ Detection failed: {e}")
    
    with col2:
        # Display results
        if st.session_state.detection_results:
            latest_result = st.session_state.detection_results[-1]
            st.image(latest_result['image'], caption="Detection Results", use_column_width=True)
            
            # Detection summary
            st.subheader("📋 Detection Summary")
            detections = latest_result['detections']
            
            if detections:
                # Create detection table
                detection_data = []
                for i, det in enumerate(detections):
                    detection_data.append({
                        'Object': det['class_name'],
                        'Confidence': f"{det['confidence']:.3f}",
                        'Bounding Box': f"({det['bbox_scaled'][0]}, {det['bbox_scaled'][1]}) - ({det['bbox_scaled'][2]}, {det['bbox_scaled'][3]})"
                    })
                
                df = pd.DataFrame(detection_data)
                st.dataframe(df, use_container_width=True)
                
                # Performance metrics
                st.metric("Processing Time", f"{latest_result['processing_time']:.1f} ms")
                st.metric("Objects Detected", len(detections))
                st.metric("Model Used", latest_result['model'].upper())
            else:
                st.info("No objects detected with current confidence threshold")

def analytics_tab():
    """Analytics and visualization tab"""
    st.header("📊 Detection Analytics")
    
    # Get database statistics
    stats = st.session_state.db.get_detection_statistics()
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Images", stats['total_images'])
    
    with col2:
        st.metric("Total Detections", stats['total_detections'])
    
    with col3:
        avg_detections = stats['total_detections'] / max(stats['total_images'], 1)
        st.metric("Avg Detections/Image", f"{avg_detections:.1f}")
    
    with col4:
        if stats['model_statistics']:
            best_model = max(stats['model_statistics'].items(), key=lambda x: x[1]['count'])
            st.metric("Most Used Model", best_model[0].upper())
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Model performance chart
        if stats['model_statistics']:
            model_data = []
            for model, data in stats['model_statistics'].items():
                model_data.append({
                    'Model': model.upper(),
                    'Detections': data['count'],
                    'Avg Confidence': data['avg_confidence']
                })
            
            df_models = pd.DataFrame(model_data)
            
            fig = px.bar(df_models, x='Model', y='Detections', 
                        title="Detections by Model",
                        color='Avg Confidence',
                        color_continuous_scale='viridis')
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Top detected classes
        if stats['top_classes']:
            class_data = []
            for cls in stats['top_classes'][:10]:  # Top 10
                class_data.append({
                    'Class': cls['class_name'],
                    'Count': cls['count'],
                    'Avg Confidence': cls['avg_confidence']
                })
            
            df_classes = pd.DataFrame(class_data)
            
            fig = px.pie(df_classes, values='Count', names='Class', 
                        title="Top Detected Classes")
            st.plotly_chart(fig, use_container_width=True)
    
    # Model performance details
    st.subheader("⚡ Model Performance")
    performance = st.session_state.db.get_model_performance()
    
    if performance:
        perf_df = pd.DataFrame(performance)
        st.dataframe(perf_df, use_container_width=True)
    else:
        st.info("No performance data available yet")

def database_tab():
    """Database management tab"""
    st.header("🗄️ Database Management")
    
    # Search interface
    st.subheader("🔍 Search Detections")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        search_class = st.selectbox(
            "Filter by Class",
            ["All"] + [cls['class_name'] for cls in st.session_state.db.get_detection_statistics()['top_classes']]
        )
    
    with col2:
        search_model = st.selectbox(
            "Filter by Model",
            ["All", "yolo", "ssd_mobilenet", "efficientdet"]
        )
    
    with col3:
        min_confidence = st.slider("Min Confidence", 0.0, 1.0, 0.0, 0.1)
    
    # Search button
    if st.button("🔍 Search"):
        with st.spinner("Searching database..."):
            results = st.session_state.db.search_detections(
                class_name=search_class if search_class != "All" else None,
                model_name=search_model if search_model != "All" else None,
                min_confidence=min_confidence if min_confidence > 0 else None,
                limit=50
            )
            
            if results:
                st.success(f"Found {len(results)} detections")
                
                # Display results
                search_df = pd.DataFrame(results)
                st.dataframe(search_df, use_container_width=True)
            else:
                st.info("No detections found matching criteria")
    
    # Database operations
    st.subheader("🛠️ Database Operations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export Data"):
            export_file = f"detection_export_{int(time.time())}.json"
            st.session_state.db.export_data(export_file, "json")
            st.success(f"Data exported to {export_file}")
    
    with col2:
        if st.button("🧹 Clean Old Data"):
            deleted = st.session_state.db.cleanup_old_data(30)
            st.success(f"Cleaned up {deleted} old records")
    
    with col3:
        if st.button("📈 Generate Sample Data"):
            with st.spinner("Generating sample data..."):
                generator = MockDataGenerator(st.session_state.db)
                image_ids = generator.generate_sample_images(5)
                generator.generate_sample_detections(image_ids)
            st.success("Sample data generated!")

def about_tab():
    """About and information tab"""
    st.header("ℹ️ About This System")
    
    st.markdown("""
    ## 🔍 Modern Object Detection System
    
    This is a comprehensive object detection system built with the latest AI technologies and tools.
    
    ### ✨ Features
    
    - **Multiple Models**: Support for YOLOv8, YOLOv9, SSD MobileNet, and EfficientDet
    - **Real-time Detection**: Fast inference with GPU acceleration
    - **Interactive Web UI**: Modern Streamlit interface
    - **Database Storage**: SQLite-based storage for results and analytics
    - **Batch Processing**: Process multiple images efficiently
    - **Performance Metrics**: Detailed analytics and model comparison
    - **Export Capabilities**: Export results in JSON/CSV formats
    
    ### 🛠️ Technologies Used
    
    - **Deep Learning**: PyTorch, TensorFlow, Ultralytics YOLO
    - **Computer Vision**: OpenCV, PIL
    - **Web Framework**: Streamlit
    - **Database**: SQLite with SQLAlchemy
    - **Visualization**: Plotly, Matplotlib
    - **Data Processing**: Pandas, NumPy
    
    ### 📊 Model Performance
    
    The system tracks and compares performance across different models:
    - Detection accuracy
    - Processing speed
    - Confidence scores
    - Class distribution
    
    ### 🚀 Getting Started
    
    1. Upload an image using the file uploader
    2. Select your preferred detection model
    3. Adjust confidence threshold as needed
    4. Click "Detect Objects" to run detection
    5. View results and analytics
    
    ### 📈 Analytics
    
    The system provides comprehensive analytics including:
    - Detection statistics by model
    - Most commonly detected objects
    - Performance metrics over time
    - Database search and filtering
    
    ### 🔧 Configuration
    
    All settings can be customized through the `config.yaml` file:
    - Model parameters
    - Detection thresholds
    - UI preferences
    - Performance settings
    """)
    
    # System information
    st.subheader("🔧 System Information")
    
    if st.session_state.detector:
        model_info = st.session_state.detector.get_model_info()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.json({
                "Current Model": model_info['current_model'],
                "Device": model_info['device'],
                "Available Models": model_info['available_models']
            })
        
        with col2:
            st.json(model_info['config'])

if __name__ == "__main__":
    main()
