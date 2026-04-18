import torch
import numpy as np

class PanopticFusion:
    """
    Merges outputs from Semantic Segmentation branch and Instance Segmentation branch
    to create a single Panoptic Map where "stuff" and "things" are combined.
    """
    def __init__(self, overlap_threshold=0.5):
        self.overlap_threshold = overlap_threshold

    def fuse(self, semantic_map, instance_outputs):
        """
        semantic_map: Tensor of shape (H, W) containing class indices.
        instance_outputs: dict with 'masks', 'labels', 'scores' for instances.
        
        Returns:
            panoptic_map: Tensor of shape (H, W) where each pixel has a unique ID.
                          (class_id * 1000 + instance_id)
            segments_info: list of dicts describing each segment found.
        """
        P = torch.zeros_like(semantic_map, dtype=torch.long)
        segments_info = []
        
        # 1. Base map initialized with "stuff" (semantic segmentation)
        # In a real Cityscapes dataset, we'd filter out "thing" classes from the semantic map
        P = semantic_map.clone().long()
        
        # Add stuff to segments info (unique classes in semantic map)
        unique_classes = torch.unique(P)
        for cls in unique_classes:
            segments_info.append({
                'id': int(cls.item()),          # For stuff, ID is just the class ID
                'category_id': int(cls.item()),
                'isthing': False
            })

        # 2. Overlay instances ("things") using confident instance masks
        if len(instance_outputs) == 0 or len(instance_outputs['masks']) == 0:
            return P, segments_info
            
        masks = instance_outputs['masks'].squeeze(1) # (N, H, W)
        labels = instance_outputs['labels']          # (N,)
        scores = instance_outputs['scores']          # (N,)
        
        # Sort by confidence descending
        sorted_indices = torch.argsort(scores, descending=True)
        
        current_instance_id = 1
        
        for idx in sorted_indices:
            mask = masks[idx] > 0.5 # binary mask
            label = labels[idx].item()
            
            # Combine category id and instance id: e.g., category 3, instance 1 => 3001
            panoptic_id = label * 1000 + current_instance_id
            
            # Find overlap with existing things
            # (In a strict approach, we check if it heavily overlaps another mask)
            # For simplicity, we directly overwrite background pixels 
            # or optionally overwrite based on depth/confidence
            
            P[mask] = panoptic_id
            
            segments_info.append({
                'id': panoptic_id,
                'category_id': label,
                'isthing': True,
                'score': scores[idx].item()
            })
            
            current_instance_id += 1
            
        return P, segments_info
