import torch
import torch.nn as nn
from ultralytics import YOLO

class ObjectDetection(nn.Module):
    """
    Object Detection Module using YOLOv8.
    Predicts bounding boxes and classes extremely fast.
    """
    def __init__(self, model_name='yolov8n.pt', device='cpu', conf_thresh=0.5):
        super().__init__()
        self.device = 'cuda' if device == 'cuda' and torch.cuda.is_available() else 'cpu'
        self.conf_thresh = conf_thresh
        
        # Load YOLOv8 model (downloads automatically if not present)
        # We use YOLOv8 Nano for speed, but models like yolov8m.pt can be swapped in.
        self.model = YOLO(model_name)
        
    def forward(self, img_path_or_tensor):
        """
        img_path_or_tensor: Can be a file path, PIL image, OpenCV image, or numpy array.
        Returns:
            list of Results objects containing boxes and classes.
        """
        # Ultralytics natively handles prediction formatting
        results = self.model.predict(
            source=img_path_or_tensor, 
            conf=self.conf_thresh, 
            device=self.device,
            verbose=False
        )
        return results
