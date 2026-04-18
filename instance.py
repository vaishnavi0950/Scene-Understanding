import torch
import torch.nn as nn
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights

class InstanceSegmentation(nn.Module):
    """
    Instance Segmentation Module using Mask R-CNN.
    Predicts bounding boxes, classes, and masks for "things" (e.g., cars, pedestrians).
    """
    def __init__(self, device='cpu', score_thresh=0.5):
        super().__init__()
        self.device = device
        self.score_thresh = score_thresh
        
        # Load pre-trained Mask R-CNN on COCO
        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = maskrcnn_resnet50_fpn(weights=weights)
        self.model.to(self.device)
        self.model.eval()
        
    def forward(self, img_tensor):
        """
        img_tensor: Tensor of shape (B, 3, H, W) scaled to [0, 1]
        Returns:
            list of dicts containing 'boxes', 'labels', 'scores', 'masks'
        """
        with torch.no_grad():
            # MaskRCNN takes list of tensors or batched tensors and handles its own normalization
            predictions = self.model(img_tensor)
            
            # Filter by confidence threshold
            filtered_preds = []
            for pred in predictions:
                keep = pred['scores'] > self.score_thresh
                filtered_pred = {
                    'boxes': pred['boxes'][keep],
                    'labels': pred['labels'][keep],
                    'scores': pred['scores'][keep],
                    'masks': pred['masks'][keep]
                }
                filtered_preds.append(filtered_pred)
                
        return filtered_preds
