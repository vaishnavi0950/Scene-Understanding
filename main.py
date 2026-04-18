import os
import torch
import cv2
import requests
from PIL import Image
from models.segmentation import SemanticSegmentation
from models.instance import InstanceSegmentation
from models.detection import ObjectDetection
from models.panoptic import PanopticFusion
from data.dataset import get_inference_loader
from utils.visualization import plot_semantic_map, plot_yolo_boxes, plot_instance_map, display_pipeline_results

def download_sample_image(samples_dir):
    """
    Downloads a sample street scene image if none exist to test the pipeline out of the box.
    """
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)
        
    sample_path = os.path.join(samples_dir, "sample_street.jpg")
    if not os.path.exists(sample_path):
        print("Downloading sample street image...")
        url = "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg"
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        with open(sample_path, 'wb') as f:
            f.write(response.content)
        print("Downloaded sample.")
    return sample_path
        
def main():
    print("Initializing Multi-Task Scene Understanding Pipeline...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize models
    semantic_model = SemanticSegmentation(device=device)
    instance_model = InstanceSegmentation(device=device, score_thresh=0.7)
    detection_model = ObjectDetection(device=device, conf_thresh=0.5)
    panoptic_fusion = PanopticFusion(overlap_threshold=0.5)
    
    # Get Data
    samples_dir = 'samples'
    download_sample_image(samples_dir)
    loader = get_inference_loader(samples_dir, batch_size=1)
    
    for idx, (img_tensor, path, orig_imgs) in enumerate(loader):
        print(f"Processing image: {path[0]}")
        img_tensor = img_tensor.to(device)
        
        # 1. Semantic Segmentation
        print("- Running DeepLabV3+ Semantic Segmentation...")
        semantic_out = semantic_model(img_tensor) # shape: (1, H, W)
        semantic_map = semantic_out[0]
        
        # 2. Instance Segmentation
        print("- Running Mask R-CNN Instance Segmentation...")
        instance_out = instance_model(img_tensor)
        instance_res = instance_out[0] # dict of masks, labels, scores for image 1
        
        # 3. Object Detection
        print("- Running YOLOv8 Object Detection...")
        # ultralytics can take file paths directly for ease
        yolo_res = detection_model(path[0])
        
        # 4. Panoptic Fusion
        print("- Running Panoptic Fusion...")
        panoptic_map, segments = panoptic_fusion.fuse(semantic_map, instance_res)
        
        print(f"  -> Found {len(segments)} panoptic segments.")
        
        # 5. Visualizer
        print("- Generating visualizations...")
        orig_img_cv = cv2.cvtColor(cv2.imread(path[0]), cv2.COLOR_BGR2RGB)
        
        # Vis semantic
        sem_vis = plot_semantic_map(orig_img_cv, semantic_map, alpha=0.6)
        
        # Vis instance
        inst_vis = plot_instance_map(orig_img_cv, instance_res, alpha=0.6)
        
        # Vis yolo
        yolo_vis = plot_yolo_boxes(orig_img_cv, yolo_res)
        
        # Display all
        display_pipeline_results(
            orig_img=orig_img_cv,
            semantic_res=sem_vis,
            instance_res_vis=inst_vis,
            yolo_img=yolo_vis,
            panoptic_map=panoptic_map,
            save_path=f"result_{idx}.png"
        )
        
    print("Pipeline execution complete.")

if __name__ == '__main__':
    main()
