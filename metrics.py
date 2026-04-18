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

def calculate_confusion_matrix(pred_mask, gt_mask, num_classes):
    """
    Calculate confusion matrix for semantic segmentation.
    Returns: numpy array of shape (num_classes, num_classes)
    """
    pred = pred_mask.flatten()
    gt = gt_mask.flatten()
    
    valid_idx = (gt >= 0) & (gt < num_classes)
    pred_valid = pred[valid_idx]
    gt_valid = gt[valid_idx]
    
    conf_matrix = np.bincount(
        num_classes * gt_valid.cpu().numpy() + pred_valid.cpu().numpy(),
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)
    
    return conf_matrix
