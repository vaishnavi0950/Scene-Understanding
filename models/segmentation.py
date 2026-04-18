import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from torchvision.transforms import functional as F

class SemanticSegmentation(nn.Module):
    """
    Semantic Segmentation Module using DeepLabV3 with a ResNet50 Backbone.
    Extracts semantic classes ("stuff" like road, sky, buildings, vegetation).
    """
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        # Load fully pre-trained model on COCO, which has standard classes out-of-the-box
        weights = DeepLabV3_ResNet50_Weights.DEFAULT
        self.model = deeplabv3_resnet50(weights=weights)
        self.model.to(self.device)
        self.model.eval()
        
        # ImageNet normalization parameters needed specifically for this branch if input isn't normalized yet
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
    def preprocess(self, img_tensor):
        # Apply ImageNet normalization 
        return F.normalize(img_tensor, mean=self.mean, std=self.std)

    def forward(self, img_tensor):
        """
        img_tensor: Tensor of shape (B, 3, H, W) scaled to [0, 1]
        Returns:
            semantic_map: Tensor of shape (B, H, W) with class IDs.
        """
        with torch.no_grad():
            img_normalized = self.preprocess(img_tensor)
            output = self.model(img_normalized)['out']
            # Output is (B, num_classes, H, W). We take argmax to get the class map.
            semantic_map = output.argmax(1)
        return semantic_map
