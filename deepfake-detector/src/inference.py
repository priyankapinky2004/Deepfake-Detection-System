# deepfake_detector/src/inference.py
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import DeepfakeDetector
import face_recognition
import os
from typing import Tuple, List, Optional
import json

class GradCAM:
    """Gradient-weighted Class Activation Mapping for explainable AI"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)
    
    def generate_cam(self, input_tensor, class_idx=None):
        """Generate class activation map"""
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        loss = output[0, class_idx]
        loss.backward()
        
        # Generate CAM
        gradients = self.gradients[0].cpu().data.numpy()
        activations = self.activations[0].cpu().data.numpy()
        
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
        
        # Normalize
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        
        return cam, class_idx

class DeepfakeInference:
    """Main inference class for deepfake detection"""
    
    def __init__(self, model_path, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Setup GradCAM
        self.grad_cam = GradCAM(self.model, self._get_target_layer())
        
        # Setup transforms
        self.transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        # Class labels
        self.class_labels = {0: 'Real', 1: 'Fake'}
    
    def _load_model(self, model_path):
        """Load trained model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Initialize model with same config
        model = DeepfakeDetector(
            model_name=checkpoint['config'].get('model_name', 'efficientnet-b0'),
            num_classes=2
        )
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        return model
    
    def _get_target_layer(self):
        """Get target layer for GradCAM"""
        # For EfficientNet, use the last convolutional layer
        return self.model.backbone._conv_head
    
    def _detect_face(self, image):
        """Detect and crop face from image"""
        # Convert to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image
        
        # Detect faces
        face_locations = face_recognition.face_locations(rgb_image)
        
        if face_locations:
            # Get the largest face
            top, right, bottom, left = face_locations[0]
            
            # Add padding
            padding = 20
            height, width = rgb_image.shape[:2]
            top = max(0, top - padding)
            left = max(0, left - padding)
            bottom = min(height, bottom + padding)
            right = min(width, right + padding)
            
            # Crop face
            face = rgb_image[top:bottom, left:right]
            return face, (top, left, bottom, right)
        
        # If no face detected, return center crop
        h, w = rgb_image.shape[:2]
        center_crop = rgb_image[h//4:3*h//4, w//4:3*w//4]
        return center_crop, (h//4, w//4, 3*h//4, 3*w//4)
    
    def predict_image(self, image_path, return_heatmap=True):
        """Predict single image"""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Detect face
        face, bbox = self._detect_face(image)
        
        # Preprocess
        face_resized = cv2.resize(face, (224, 224))
        transformed = self.transform(image=face_resized)
        input_tensor = transformed['image'].unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            confidence = probabilities.max().item()
            prediction = output.argmax(dim=1).item()
        
        result = {
            'prediction': self.class_labels[prediction],
            'confidence': confidence,
            'probabilities': {
                'real': probabilities[0][0].item(),
                'fake': probabilities[0][1].item()
            },
            'face_bbox': bbox
        }
        
        # Generate heatmap if requested
        if return_heatmap:
            cam, _ = self.grad_cam.generate_cam(input_tensor, class_idx=prediction)
            result['heatmap'] = cam
            result['visualization'] = self._create_visualization(face_resized, cam, result)
        
        return result
    
    def predict_video(self, video_path, max_frames=30, return_frame_results=False):
        """Predict video by analyzing multiple frames"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        frame_results = []
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        skip_frames = max(1, total_frames // max_frames)
        
        real_scores = []
        fake_scores = []
        
        while cap.isOpened() and len(frame_results) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames for efficiency
            if frame_count % skip_frames != 0:
                frame_count += 1
                continue
            
            try:
                # Detect face
                face, bbox = self._detect_face(frame)
                
                # Preprocess
                face_resized = cv2.resize(face, (224, 224))
                transformed = self.transform(image=face_resized)
                input_tensor = transformed['image'].unsqueeze(0).to(self.device)
                
                # Inference
                with torch.no_grad():
                    output = self.model(input_tensor)
                    probabilities = F.softmax(output, dim=1)
                    prediction = output.argmax(dim=1).item()
                
                frame_result = {
                    'frame_number': frame_count,
                    'prediction': self.class_labels[prediction],
                    'confidence': probabilities.max().item(),
                    'probabilities': {
                        'real': probabilities[0][0].item(),
                        'fake': probabilities[0][1].item()
                    }
                }
                
                frame_results.append(frame_result)
                real_scores.append(probabilities[0][0].item())
                fake_scores.append(probabilities[0][1].item())
                
            except Exception as e:
                print(f"Error processing frame {frame_count}: {e}")
            
            frame_count += 1
        
        cap.release()
        
        # Aggregate results
        if real_scores and fake_scores:
            avg_real_score = np.mean(real_scores)
            avg_fake_score = np.mean(fake_scores)
            final_prediction = 'Real' if avg_real_score > avg_fake_score else 'Fake'
            confidence = max(avg_real_score, avg_fake_score)
        else:
            final_prediction = 'Unknown'
            confidence = 0.0
            avg_real_score = avg_fake_score = 0.0
        
        result = {
            'prediction': final_prediction,
            'confidence': confidence,
            'avg_probabilities': {
                'real': avg_real_score,
                'fake': avg_fake_score
            },
            'frames_analyzed': len(frame_results),
            'total_frames': total_frames
        }
        
        if return_frame_results:
            result['frame_results'] = frame_results
        
        return result
    
    def _create_visualization(self, original_face, heatmap, result):
        """Create visualization with original image and heatmap overlay"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(original_face)
        axes[0].set_title('Original Face')
        axes[0].axis('off')
        
        # Heatmap
        axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title('Attention Map')
        axes[1].axis('off')
        
        # Overlay
        overlay = original_face.copy()
        heatmap_colored = plt.cm.jet(heatmap)[:, :, :3]
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
        overlay = cv2.addWeighted(overlay, 0.6, heatmap_colored, 0.4, 0)
        
        axes[2].imshow(overlay)
        axes[2].set_title(f'Prediction: {result["prediction"]}\nConfidence: {result["confidence"]:.3f}')
        axes[2].axis('off')
        
        plt.tight_layout()
        return fig
    
    def batch_predict(self, image_paths, save_results=True):
        """Predict multiple images"""
        results = {}
        
        for image_path in image_paths:
            try:
                result = self.predict_image(image_path, return_heatmap=False)
                results[image_path] = result
                print(f"✓ {os.path.basename(image_path)}: {result['prediction']} ({result['confidence']:.3f})")
            except Exception as e:
                print(f"✗ {os.path.basename(image_path)}: Error - {e}")
                results[image_path] = {'error': str(e)}
        
        if save_results:
            with open('batch_predictions.json', 'w') as f:
                json.dump(results, f, indent=2)
        
        return results
    
    def evaluate_on_test_set(self, test_loader):
        """Evaluate model on test set"""
        self.model.eval()
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                probs = F.softmax(output, dim=1)
                preds = output.argmax(dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_preds)
        auc = roc_auc_score(all_targets, all_probs)
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test AUC: {auc:.4f}")
        
        return {
            'accuracy': accuracy,
            'auc': auc,
            'predictions': all_preds,
            'targets': all_targets,
            'probabilities': all_probs
        }

# Example usage
if __name__ == "__main__":
    # Initialize inference
    model_path = "models/checkpoint_best.pth"
    detector = DeepfakeInference(model_path)
    
    # Single image prediction
    image_path = "test_image.jpg"
    result = detector.predict_image(image_path)
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.3f}")
    
    # Video prediction
    video_path = "test_video.mp4"
    video_result = detector.predict_video(video_path)
    print(f"Video prediction: {video_result['prediction']}")
    print(f"Confidence: {video_result['confidence']:.3f}")
    
    # Batch prediction
    image_list = ["image1.jpg", "image2.jpg", "image3.jpg"]
    batch_results = detector.batch_predict(image_list)