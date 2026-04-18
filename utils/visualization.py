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

def plot_instance_map(image, instance_res, alpha=0.5):
    """
    Overlays instance segmentation masks on original image.
    """
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
    else:
        image = image.copy()
        
    masks = instance_res['masks'] # shape: (N, 1, H, W)
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy()
        
    np.random.seed(42)
    blended = image.copy()
    
    for i in range(masks.shape[0]):
        mask = masks[i, 0] > 0.5
        # Resize mask to fit original image
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            mask = mask.astype(np.uint8)
            
        color = np.random.randint(0, 255, size=(3,), dtype=np.uint8)
        
        color_mask = np.zeros_like(image)
        color_mask[mask > 0] = color
        
        mask_idx = mask > 0
        blended[mask_idx] = cv2.addWeighted(blended, 1 - alpha, color_mask, alpha, 0)[mask_idx]
        
    return blended

def display_pipeline_results(orig_img, semantic_res, instance_res_vis, yolo_img, panoptic_map, save_path="output.png"):
    """
    Displays all outputs in a grid layout and saves to disk.
    """
    plt.figure(figsize=(20, 15))
    
    plt.subplot(2, 3, 1)
    plt.title("Original Image")
    plt.imshow(orig_img)
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.title("Semantic Segmentation (DeepLabV3+)")
    plt.imshow(semantic_res)
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.title("Instance Segmentation (Mask R-CNN)")
    plt.imshow(instance_res_vis)
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.title("Object Detection (YOLOv8)")
    plt.imshow(yolo_img)
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.title("Panoptic Map (Fusion)")
    # Normalize panoptic map for visualization
    if isinstance(panoptic_map, torch.Tensor):
        panoptic_map = panoptic_map.cpu().numpy()
    
    # Just modulo by 255 to create distinctive colors for instance IDs
    vis_panoptic = panoptic_map % 255
    if vis_panoptic.shape[:2] != orig_img.shape[:2]:
        vis_panoptic = cv2.resize(vis_panoptic.astype(np.uint8), (orig_img.shape[1], orig_img.shape[0]), interpolation=cv2.INTER_NEAREST)

    plt.imshow(vis_panoptic, cmap='nipy_spectral')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Results saved to {save_path}")
    plt.close()

def plot_confusion_matrix_to_disk(conf_matrix, save_path="confusion_matrix.png"):
    """
    Plots a confusion matrix and saves it to disk.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    plt.figure(figsize=(10, 8))
    # Normalize by row (ground truth)
    row_sums = conf_matrix.sum(axis=1, keepdims=True)
    # Avoid division by zero
    norm_conf_matrix = np.divide(conf_matrix, row_sums, out=np.zeros_like(conf_matrix, dtype=float), where=row_sums!=0)
    
    sns.heatmap(norm_conf_matrix, annot=False, cmap='Blues')
    plt.title('Normalized Confusion Matrix')
    plt.ylabel('Ground Truth')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix plot saved to {save_path}")
