# deepfake_detector/src/model.py
import torch
import torch.nn as nn
import torchvision.models as models
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F

class DeepfakeDetector(nn.Module):
    """
    CNN-based deepfake detection model using EfficientNet backbone
    """
    def __init__(self, model_name='efficientnet-b0', num_classes=2, dropout_rate=0.5):
        super(DeepfakeDetector, self).__init__()
        
        # Load pretrained EfficientNet
        self.backbone = EfficientNet.from_pretrained(model_name)
        
        # Get the number of features from backbone
        num_features = self.backbone._fc.in_features
        
        # Replace the classifier
        self.backbone._fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
        # Add attention mechanism
        self.attention = SelfAttention(num_features)
        
    def forward(self, x):
        # Extract features using backbone
        features = self.backbone.extract_features(x)
        
        # Apply attention
        attended_features = self.attention(features)
        
        # Global average pooling
        pooled = F.adaptive_avg_pool2d(attended_features, 1)
        pooled = pooled.view(pooled.size(0), -1)
        
        # Classification
        output = self.backbone._fc(pooled)
        
        return output

class SelfAttention(nn.Module):
    """Self-attention module to focus on important facial regions"""
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Generate query, key, value
        query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, width * height)
        value = self.value_conv(x).view(batch_size, -1, width * height)
        
        # Attention
        attention = torch.bmm(query, key)
        attention = self.softmax(attention)
        
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        
        return self.gamma * out + x

class XceptionBasedDetector(nn.Module):
    """Alternative Xception-based detector"""
    def __init__(self, num_classes=2):
        super(XceptionBasedDetector, self).__init__()
        
        # Load pretrained Xception (via torchvision)
        self.backbone = models.resnet50(pretrained=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        
        # Freeze early layers
        for param in list(self.backbone.parameters())[:-10]:
            param.requires_grad = False
            
    def forward(self, x):
        return self.backbone(x)