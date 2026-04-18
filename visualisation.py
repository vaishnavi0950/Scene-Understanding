import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2

def plot_semantic_map(image, semantic_map, alpha=0.5):
    """
    Overlays semantic map on original image.
    """
    # Ensure image is numpy HxWx3
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()
        
    if isinstance(semantic_map, torch.Tensor):
        semantic_map = semantic_map.cpu().numpy()
        
    # Generate random colors for classes
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(256, 3), dtype=np.uint8)
    
    color_mask = colors[semantic_map]
    
    # Resize color_mask to match image if necessary
    if color_mask.shape[:2] != image.shape[:2]:
        color_mask = cv2.resize(color_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        
    # Image might be 0-1, convert to 0-255 uint8 if so
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
        
    blended = cv2.addWeighted(image, 1 - alpha, color_mask, alpha, 0)
    
    return blended

def plot_yolo_boxes(image, yolo_results):
    """
    Draws YOLO bounding boxes on image.
    """
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
    else:
        image = image.copy()
        
    res = yolo_results[0]
    boxes = res.boxes.xyxy.cpu().numpy()
    classes = res.boxes.cls.cpu().numpy()
    
    for box, cls in zip(boxes, classes):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(image, str(int(cls)), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
    return image

def display_pipeline_results(orig_img, semantic_res, yolo_img, panoptic_map, save_path="output.png"):
    """
    Displays all outputs in a grid layout and saves to disk.
    """
    plt.figure(figsize=(20, 10))
    
    plt.subplot(2, 2, 1)
    plt.title("Original Image")
    plt.imshow(orig_img)
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.title("Semantic Segmentation (DeepLabV3+)")
    plt.imshow(semantic_res)
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.title("Object Detection (YOLOv8)")
    plt.imshow(yolo_img)
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.title("Panoptic Map (Fusion)")
    # Normalize panoptic map for visualization
    if isinstance(panoptic_map, torch.Tensor):
        panoptic_map = panoptic_map.cpu().numpy()
    
    # Just modulo by 255 to create distinctive colors for instance IDs
    vis_panoptic = panoptic_map % 255
    plt.imshow(vis_panoptic, cmap='nipy_spectral')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Results saved to {save_path}")
    plt.close()
