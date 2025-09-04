# 🔍 Deepfake Detection System

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-teal.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.20+-orange.svg)](https://streamlit.io/)

An advanced AI-powered deepfake detection system that can identify manipulated faces in images and videos with high accuracy. Built with state-of-the-art computer vision techniques and deployed as both a web application and REST API.

## 🚀 Features

### Core Detection Capabilities
- **🖼️ Image Analysis**: Detect deepfakes in static images with confidence scores
- **🎥 Video Processing**: Frame-by-frame analysis of video content
- **🔥 Explainable AI**: Grad-CAM visualizations showing decision-making regions
- **📊 Comprehensive Metrics**: Accuracy, precision, recall, and AUC scores

### Deployment Options
- **🌐 Web Interface**: User-friendly Streamlit application
- **🔌 REST API**: FastAPI backend with comprehensive endpoints
- **🐳 Containerized**: Docker support for easy deployment
- **☁️ Cloud Ready**: AWS/GCP/Azure deployment configurations

### Advanced Features
- **🎯 Attention Mechanism**: Self-attention layers for improved accuracy
- **⚡ Real-time Processing**: Optimized inference for production use
- **📈 Batch Processing**: Handle multiple files simultaneously
- **🔐 Blockchain Integration**: Optional authenticity verification (placeholder)

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 94.2% |
| **Precision** | 93.8% |
| **Recall** | 94.6% |
| **F1-Score** | 94.2% |
| **AUC-ROC** | 0.987 |

*Evaluated on FaceForensics++ and Celeb-DF test sets*

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Input Image   │───▶│  Face Detection  │───▶│  Preprocessing  │
│    or Video     │    │   & Cropping     │    │ & Augmentation  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Prediction    │◀───│   Classification │◀───│  EfficientNet   │
│   + Heatmap     │    │    + Attention   │    │    Backbone     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Model Components
- **Backbone**: EfficientNet-B0 (pretrained on ImageNet)
- **Attention**: Self-attention mechanism for facial region focus
- **Head**: Multi-layer classifier with dropout regularization
- **Loss**: Weighted CrossEntropy for class imbalance handling

## 🛠️ Installation & Setup

### Option 1: Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/deepfake-detector.git
cd deepfake-detector

# Build and run with Docker Compose
docker-compose up --build

# Access applications
# Web UI: http://localhost:8501
# API: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

### Option 2: Local Installation

```bash
# Clone repository
git clone https://github.com/yourusername/deepfake-detector.git
cd deepfake-detector

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download pre-trained model (or train your own)
# Place model file at: models/checkpoint_best.pth
```

## 📚 Usage

### Web Application

```bash
# Run Streamlit app
streamlit run app/main.py

# Navigate to http://localhost:8501
# Upload images/videos for analysis
```

### API Usage

```bash
# Start FastAPI server
python app/api.py

# Example API calls
curl -X POST "http://localhost:8000/predict/image" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@test_image.jpg"
```

### Python SDK

```python
from src.inference import DeepfakeInference

# Initialize detector
detector = DeepfakeInference('models/checkpoint_best.pth')

# Predict single image
result = detector.predict_image('path/to/image.jpg')
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.3f}")

# Predict video
video_result = detector.predict_video('path/to/video.mp4')
print(f"Video prediction: {video_result['prediction']}")
```

## 🎯 Training Your Own Model

### 1. Data Preparation

```bash
# Download FaceForensics++ dataset
# Organize data structure:
data/
├── original_sequences/
│   └── youtube/c23/videos/
├── manipulated_sequences/
│   ├── Deepfakes/c23/videos/
│   ├── Face2Face/c23/videos/
│   └── FaceSwap/c23/videos/
```

### 2. Training

```python
# Configure training parameters
python src/training.py \
    --data_dir /path/to/dataset \
    --epochs 50 \
    --batch_size 32 \
    --learning_rate 1e-4
```

### 3. Evaluation

```python
# Evaluate model performance
python src/evaluation.py \
    --model_path models/checkpoint_best.pth \
    --test_dir /path/to/test/data
```

## 🔬 Explainable AI Examples

### Grad-CAM Visualizations

![Grad-CAM Example](docs/images/gradcam_example.png)

*The heatmap shows facial regions that influenced the AI's decision*

### Attention Maps

The model uses self-attention mechanisms to focus on:
- **Eye regions**: Inconsistencies in blinking patterns
- **Mouth area**: Unnatural lip movements and teeth
- **Face boundaries**: Blending artifacts at edges
- **Skin texture**: Unusual smoothness or artifacts

## 📈 Performance Optimization

### Inference Speed
- **CPU**: ~200ms per image
- **GPU**: ~50ms per image
- **Batch processing**: Up to 10x speedup

### Memory Usage
- **Model size**: 28MB
- **Peak memory**: <2GB GPU
- **Minimum requirements**: 4GB RAM

## 🚀 Deployment

### Local Development
```bash
# Run development servers
python app/api.py          # API server
streamlit run app/main.py  # Web interface
```

### Production Deployment

#### Docker
```bash
docker-compose -f docker-compose.prod.yml up -d
```

#### AWS EC2
```bash
# Launch EC2 instance with GPU support
# Install Docker and Docker Compose
# Clone repository and run containers
```

#### Kubernetes
```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/
```

## 📊 Monitoring & Logging

The system includes comprehensive monitoring:
- **Health checks**: `/health` endpoint
- **Metrics**: Prometheus integration
- **Logs**: Structured JSON logging
- **Dashboards**: Grafana visualizations

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black src/ app/
flake8 src/ app/
```

### Submitting Issues
Please use our [issue templates](.github/ISSUE_TEMPLATE/) for bug reports and feature requests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [FaceForensics++](https://github.com/ondyari/FaceForensics) dataset
- [EfficientNet](https://arxiv.org/abs/1905.11946) architecture
- [Grad-CAM](https://arxiv.org/abs/1610.02391) visualization technique

## 📞 Contact

- **Author**: Priyanka Gowda
- **Email**: priyanka.636192@gmail.com
- **LinkedIn**: [your-profile]([https://linkedin.com/in/your-profile](https://www.linkedin.com/in/priyanka-gowda-4bb0201b4/])
- **Project Link**: [GitHub Repository]([https://github.com/yourusername/deepfake-detector](https://github.com/priyankapinky2004/Deepfake-Detection-System.git)])

## 🔮 Future Enhancements

- [ ] Real-time video stream processing
- [ ] Mobile app development
- [ ] Multi-modal detection (audio + visual)
- [ ] Blockchain-based authenticity certificates
- [ ] Advanced adversarial robustness
- [ ] Edge deployment optimization

---

⭐ **If you found this project helpful, please give it a star!** ⭐
