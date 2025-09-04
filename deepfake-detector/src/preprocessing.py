# deepfake_detector/src/preprocessing.py
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import json
from PIL import Image
import face_recognition
import dlib
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

class DeepfakeDataset(Dataset):
    """
    Custom dataset for deepfake detection
    Supports both FaceForensics++ and Celeb-DF datasets
    """
    def __init__(self, data_dir, metadata_file, transform=None, max_frames=10):
        self.data_dir = data_dir
        self.transform = transform
        self.max_frames = max_frames
        
        # Load metadata
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)
        
        self.samples = []
        self._prepare_samples()
        
    def _prepare_samples(self):
        """Prepare list of samples with labels"""
        for video_id, info in self.metadata.items():
            video_path = os.path.join(self.data_dir, f"{video_id}.mp4")
            if os.path.exists(video_path):
                label = 1 if info['label'] == 'FAKE' else 0
                self.samples.append({
                    'video_path': video_path,
                    'label': label,
                    'split': info.get('split', 'train')
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Extract frames from video
        frames = self._extract_frames(sample['video_path'])
        
        if len(frames) == 0:
            # Return a blank image if no frames extracted
            frames = [np.zeros((224, 224, 3), dtype=np.uint8)]
        
        # Select random frame or use all frames
        if len(frames) > 1:
            frame_idx = np.random.randint(0, len(frames))
            frame = frames[frame_idx]
        else:
            frame = frames[0]
        
        # Apply transforms
        if self.transform:
            frame = self.transform(image=frame)['image']
        
        return frame, torch.tensor(sample['label'], dtype=torch.long)
    
    def _extract_frames(self, video_path, skip_frames=5):
        """Extract frames from video with face detection"""
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        frame_count = 0
        faces_extracted = 0
        
        while cap.isOpened() and faces_extracted < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Skip frames for efficiency
            if frame_count % skip_frames != 0:
                frame_count += 1
                continue
            
            # Detect and crop faces
            face_crops = self._detect_and_crop_faces(frame)
            frames.extend(face_crops)
            faces_extracted += len(face_crops)
            
            frame_count += 1
        
        cap.release()
        return frames[:self.max_frames]
    
    def _detect_and_crop_faces(self, frame):
        """Detect faces and return cropped regions"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces using face_recognition library
        face_locations = face_recognition.face_locations(rgb_frame)
        
        face_crops = []
        for (top, right, bottom, left) in face_locations:
            # Add padding
            padding = 20
            top = max(0, top - padding)
            left = max(0, left - padding)
            bottom = min(frame.shape[0], bottom + padding)
            right = min(frame.shape[1], right + padding)
            
            # Crop and resize face
            face_crop = rgb_frame[top:bottom, left:right]
            face_crop = cv2.resize(face_crop, (224, 224))
            face_crops.append(face_crop)
        
        # If no faces detected, return center crop
        if not face_crops:
            h, w = frame.shape[:2]
            center_crop = rgb_frame[h//4:3*h//4, w//4:3*w//4]
            center_crop = cv2.resize(center_crop, (224, 224))
            face_crops.append(center_crop)
            
        return face_crops

def get_transforms():
    """Get data augmentation transforms"""
    
    train_transform = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        A.MotionBlur(blur_limit=3, p=0.3),
        A.MedianBlur(blur_limit=3, p=0.3),
        A.Blur(blur_limit=3, p=0.3),
        A.CLAHE(p=0.5),
        A.ColorJitter(p=0.5),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])
    
    return train_transform, val_transform

def create_dataloaders(data_dir, metadata_file, batch_size=32, num_workers=4):
    """Create train and validation dataloaders"""
    
    train_transform, val_transform = get_transforms()
    
    # Create datasets
    full_dataset = DeepfakeDataset(data_dir, metadata_file, transform=None)
    
    # Split data
    train_samples, val_samples = train_test_split(
        full_dataset.samples, 
        test_size=0.2, 
        random_state=42,
        stratify=[s['label'] for s in full_dataset.samples]
    )
    
    # Create separate datasets
    train_dataset = DeepfakeDataset(data_dir, metadata_file, transform=train_transform)
    train_dataset.samples = train_samples
    
    val_dataset = DeepfakeDataset(data_dir, metadata_file, transform=val_transform)
    val_dataset.samples = val_samples
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

def prepare_faceforensics_metadata(data_root):
    """
    Prepare metadata file for FaceForensics++ dataset
    Expected structure: data_root/original_sequences/youtube/c23/videos/*.mp4
                       data_root/manipulated_sequences/Deepfakes/c23/videos/*.mp4
    """
    metadata = {}
    
    # Process original videos
    original_dir = os.path.join(data_root, "original_sequences/youtube/c23/videos")
    if os.path.exists(original_dir):
        for video_file in os.listdir(original_dir):
            if video_file.endswith('.mp4'):
                video_id = video_file.replace('.mp4', '')
                metadata[video_id] = {
                    'label': 'REAL',
                    'split': 'train',
                    'source': 'youtube'
                }
    
    # Process fake videos (multiple manipulation methods)
    manipulation_methods = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
    
    for method in manipulation_methods:
        fake_dir = os.path.join(data_root, f"manipulated_sequences/{method}/c23/videos")
        if os.path.exists(fake_dir):
            for video_file in os.listdir(fake_dir):
                if video_file.endswith('.mp4'):
                    video_id = f"{method}_{video_file.replace('.mp4', '')}"
                    metadata[video_id] = {
                        'label': 'FAKE',
                        'split': 'train',
                        'source': method
                    }
    
    return metadata

# Example usage
if __name__ == "__main__":
    # Prepare metadata for FaceForensics++
    data_root = "/path/to/faceforensics++"
    metadata = prepare_faceforensics_metadata(data_root)
    
    # Save metadata
    with open("metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        data_dir=data_root,
        metadata_file="metadata.json",
        batch_size=32
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")