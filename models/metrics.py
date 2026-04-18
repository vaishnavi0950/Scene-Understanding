import torch
import numpy as np

def calculate_iou(pred_mask, gt_mask, num_classes):
    """
    Calculate Intersection over Union (IoU) for semantic segmentation.
    """
    ious = []
    pred_mask = pred_mask.view(-1)
    gt_mask = gt_mask.view(-1)
    
    # Ignore background class if needed, here we do all
    for cls in range(num_classes):
        pred_inds = pred_mask == cls
        target_inds = gt_mask == cls
        
        intersection = (pred_inds[target_inds]).long().sum().item()
        union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
        
        if union == 0:
            ious.append(float('nan'))  # Class not present in GT or Pred
        else:
            ious.append(float(intersection) / float(max(union, 1)))
            
    return np.array(ious)

def calculate_miou(ious):
    """
    Calculate mean IoU ignoring NaNs.
    """
    return np.nanmean(ious)

def pixel_accuracy(pred_mask, gt_mask):
    """
    Calculate pixel accuracy: TP / Total Pixels
    """
    correct = (pred_mask == gt_mask).sum().item()
    total = gt_mask.numel()
    return correct / total
