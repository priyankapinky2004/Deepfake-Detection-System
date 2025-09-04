# deepfake_detector/app/main.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt
import io
import os
import tempfile
import sys
sys.path.append('../src')

from inference import DeepfakeInference
import plotly.graph_objects as go
import plotly.express as px

# Page config
st.set_page_config(
    page_title="Deepfake Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .real-prediction {
        background-color: #d4edda;
        border: 2px solid #28a745;
        color: #155724;
    }
    .fake-prediction {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        color: #721c24;
    }
    .confidence-bar {
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitApp:
    def __init__(self):
        self.detector = None
        self.model_loaded = False
        
    @st.cache_resource
    def load_model(_self, model_path):
        """Load the deepfake detection model"""
        try:
            detector = DeepfakeInference(model_path)
            return detector
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    
    def render_header(self):
        """Render the main header"""
        st.markdown('<h1 class="main-header">🔍 Deepfake Detection System</h1>', unsafe_allow_html=True)
        st.markdown("---")
        
        # Info section
        with st.expander("ℹ️ About this System"):
            st.markdown("""
            This AI-powered system can detect deepfake content in images and videos using advanced computer vision techniques:
            
            **Features:**
            - 🖼️ **Image Analysis**: Upload images to detect manipulated faces
            - 🎥 **Video Analysis**: Process videos frame-by-frame for comprehensive detection
            - 🔥 **Explainable AI**: View heatmaps showing which facial regions influenced the decision
            - 📊 **Confidence Scores**: Get probability scores for real vs. fake classification
            
            **Model Architecture:**
            - EfficientNet backbone with attention mechanisms
            - Trained on FaceForensics++ and Celeb-DF datasets
            - Grad-CAM visualization for explainability
            """)
    
    def render_sidebar(self):
        """Render sidebar with model settings and info"""
        st.sidebar.header("⚙️ Settings")
        
        # Model loading
        st.sidebar.subheader("Model Configuration")
        
        # Check if model file exists
        model_path = "../models/checkpoint_best.pth"
        if os.path.exists(model_path):
            if st.sidebar.button("🔄 Load Model", type="primary"):
                self.detector = self.load_model(model_path)
                if self.detector:
                    self.model_loaded = True
                    st.sidebar.success("✅ Model loaded successfully!")
                else:
                    st.sidebar.error("❌ Failed to load model")
        else:
            st.sidebar.error("❌ Model file not found. Please train the model first.")
            st.sidebar.info("Expected path: models/checkpoint_best.pth")
        
        # Model info
        if self.model_loaded and self.detector:
            st.sidebar.subheader("Model Information")
            st.sidebar.info(f"Device: {self.detector.device}")
            st.sidebar.info("Architecture: EfficientNet-B0")
            st.sidebar.info("Input Size: 224x224")
        
        # Analysis settings
        st.sidebar.subheader("Analysis Settings")
        show_heatmap = st.sidebar.checkbox("Show Attention Heatmap", value=True)
        confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)
        
        return show_heatmap, confidence_threshold
    
    def create_prediction_display(self, result, show_heatmap=True):
        """Create a nice display for prediction results"""
        prediction = result['prediction']
        confidence = result['confidence']
        
        # Main prediction box
        box_class = "real-prediction" if prediction == "Real" else "fake-prediction"
        st.markdown(f"""
        <div class="prediction-box {box_class}">
            <h2>Prediction: {prediction}</h2>
            <h3>Confidence: {confidence:.1%}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Probability bars
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Real Probability", f"{result['probabilities']['real']:.1%}")
            st.progress(result['probabilities']['real'])
        
        with col2:
            st.metric("Fake Probability", f"{result['probabilities']['fake']:.1%}")
            st.progress(result['probabilities']['fake'])
        
        # Confidence gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = confidence,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Confidence Score"},
            delta = {'reference': 0.5},
            gauge = {
                'axis': {'range': [None, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 0.5], 'color': "lightgray"},
                    {'range': [0.5, 1], 'color': "gray"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.9}}))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Heatmap visualization
        if show_heatmap and 'visualization' in result:
            st.subheader("🔥 Attention Heatmap")
            st.pyplot(result['visualization'])
            st.caption("The heatmap shows which parts of the face the AI focused on when making its decision.")
    
    def process_image(self, uploaded_file, show_heatmap, confidence_threshold):
        """Process uploaded image"""
        if not self.model_loaded:
            st.error("Please load the model first from the sidebar.")
            return
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name
        
        try:
            # Process image
            if self.detector is None:
                st.error("Model is not loaded. Please load the model from the sidebar.")
                return
            with st.spinner("🔍 Analyzing image..."):
                result = self.detector.predict_image(temp_path, return_heatmap=show_heatmap)
            
            # Display results
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📷 Original Image")
                image = Image.open(uploaded_file)
                st.image(image, use_column_width=True)
                
                # Face detection info
                if 'face_bbox' in result:
                    st.info(f"✅ Face detected and analyzed")
            
            with col2:
                st.subheader("📊 Analysis Results")
                self.create_prediction_display(result, show_heatmap)
                
                # Warning for low confidence
                if result['confidence'] < confidence_threshold:
                    st.warning(f"⚠️ Low confidence prediction (< {confidence_threshold:.1%}). Results may be unreliable.")
        
        except Exception as e:
            st.error(f"Error processing image: {e}")
        finally:
            # Clean up temp file
            os.unlink(temp_path)
    
    def process_video(self, uploaded_file, show_heatmap, confidence_threshold):
        """Process uploaded video"""
        if not self.model_loaded:
            st.error("Please load the model first from the sidebar.")
            return
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name
        
        try:
            # Process video
            if self.detector is None:
                st.error("Model is not loaded. Please load the model from the sidebar.")
                return
            with st.spinner("🎥 Analyzing video... This may take a few minutes."):
                result = self.detector.predict_video(temp_path, max_frames=30, return_frame_results=True)
            
            # Display overall results
            st.subheader("📊 Overall Video Analysis")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("🎬 Video Info")
                st.info(f"Frames analyzed: {result['frames_analyzed']}")
                st.info(f"Total frames: {result['total_frames']}")
            
            with col2:
                # Create simplified result for display
                display_result = {
                    'prediction': result['prediction'],
                    'confidence': result['confidence'],
                    'probabilities': result['avg_probabilities']
                }
                self.create_prediction_display(display_result, show_heatmap=False)
            
            # Frame-by-frame analysis
            if 'frame_results' in result and result['frame_results']:
                st.subheader("📈 Frame-by-Frame Analysis")
                
                frame_data = result['frame_results']
                frames = [f['frame_number'] for f in frame_data]
                real_probs = [f['probabilities']['real'] for f in frame_data]
                fake_probs = [f['probabilities']['fake'] for f in frame_data]
                
                # Create plotly chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=frames, y=real_probs, mode='lines+markers', 
                                       name='Real Probability', line=dict(color='green')))
                fig.add_trace(go.Scatter(x=frames, y=fake_probs, mode='lines+markers', 
                                       name='Fake Probability', line=dict(color='red')))
                
                fig.update_layout(
                    title="Probability Scores Across Frames",
                    xaxis_title="Frame Number",
                    yaxis_title="Probability",
                    hovermode='x'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Summary statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Avg Real Score", f"{np.mean(real_probs):.3f}")
                with col2:
                    st.metric("Avg Fake Score", f"{np.mean(fake_probs):.3f}")
                with col3:
                    st.metric("Max Confidence", f"{max(result['confidence'] for result in frame_data):.3f}")
                with col4:
                    st.metric("Min Confidence", f"{min(result['confidence'] for result in frame_data):.3f}")
        
        except Exception as e:
            st.error(f"Error processing video: {e}")
        finally:
            # Clean up temp file
            os.unlink(temp_path)
    
    def render_main_content(self, show_heatmap, confidence_threshold):
        """Render main content area"""
        tab1, tab2, tab3 = st.tabs(["📷 Image Analysis", "🎥 Video Analysis", "📚 Batch Processing"])
        
        with tab1:
            st.header("Image Deepfake Detection")
            st.markdown("Upload an image to analyze whether it contains a real or manipulated face.")
            
            uploaded_file = st.file_uploader(
                "Choose an image file", 
                type=['png', 'jpg', 'jpeg'],
                help="Supported formats: PNG, JPG, JPEG"
            )
            
            if uploaded_file is not None:
                self.process_image(uploaded_file, show_heatmap, confidence_threshold)
        
        with tab2:
            st.header("Video Deepfake Detection")
            st.markdown("Upload a video file to analyze multiple frames for deepfake content.")
            
            uploaded_file = st.file_uploader(
                "Choose a video file", 
                type=['mp4', 'avi', 'mov'],
                help="Supported formats: MP4, AVI, MOV. Note: Large files may take several minutes to process."
            )
            
            if uploaded_file is not None:
                self.process_video(uploaded_file, show_heatmap, confidence_threshold)
        
        with tab3:
            st.header("Batch Processing")
            st.markdown("Process multiple files at once (Coming Soon)")
            
            st.info("🚧 Batch processing feature will be available in a future update.")
            
            # Placeholder for batch processing UI
            st.subheader("Upload Multiple Files")
            uploaded_files = st.file_uploader(
                "Choose multiple files", 
                type=['png', 'jpg', 'jpeg', 'mp4'],
                accept_multiple_files=True,
                disabled=True
            )
    
    def run(self):
        """Main app runner"""
        self.render_header()
        show_heatmap, confidence_threshold = self.render_sidebar()
        self.render_main_content(show_heatmap, confidence_threshold)
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: gray;'>
            Built with ❤️ using Streamlit | Deepfake Detection System v1.0
        </div>
        """, unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    app = StreamlitApp()
    app.run()