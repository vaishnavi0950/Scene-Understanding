import torch
import numpy as np
from tqdm import tqdm
from utils.metrics import calculate_iou, calculate_miou, pixel_accuracy
from models.segmentation import SemanticSegmentation
from data.dataset import CityscapesWrapper
from torch.utils.data import DataLoader

def evaluate_segmentation():
    """
    Example evaluation loop for calculating mIoU and Pixel Accuracy.
    (Requires Cityscapes Dataset to actually run with targets).
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Needs valid root_dir where Cityscapes dataset is downloaded
    # e.g., 'C:/Datasets/Cityscapes'
    root_dir = './data_root/cityscapes'
    
    print(f"Attempting to load dataset from {root_dir}")
    try:
        dataset = CityscapesWrapper(root_dir, split='val')
        if len(dataset) == 0:
            print("Dataset empty or not downloaded. Skipping evaluation.")
            return
            
        loader = DataLoader(dataset, batch_size=4, shuffle=False)
    except Exception as e:
        print(e)
        return
        
    model = SemanticSegmentation(device=device)
    
    all_ious = []
    all_pa = []
    
    print("Evaluating over validation set...")
    for images, targets in tqdm(loader):
        images = images.to(device)
        targets = targets.to(device) # Shape: (B, H, W)
        
        preds = model(images)
        
        for i in range(preds.shape[0]):
            pred = preds[i].cpu()
            target = targets[i].cpu()
            
            # Cityscapes has 19 eval classes usually, but our pre-trained model has 21 Pascol VOC classes
            ious_batch = calculate_iou(pred, target, num_classes=21) 
            pa_batch = pixel_accuracy(pred, target)
            
            all_ious.append(ious_batch)
            all_pa.append(pa_batch)
            
    # Calculate global metrics
    ious_matrix = np.array(all_ious) # Shape (N, num_classes)
    class_ious = np.nanmean(ious_matrix, axis=0)
    miou = calculate_miou(class_ious)
    
    mean_pa = np.mean(all_pa)
    
    print(f"Evaluation Results:")
    print(f"mIoU: {miou:.4f}")
    print(f"Pixel Accuracy: {mean_pa:.4f}")

if __name__ == '__main__':
    evaluate_segmentation()
