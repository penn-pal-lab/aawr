import os
import numpy as np
import cv2
import supervision as sv
from pathlib import Path
import re

from dinox_utils import _rle2mask, _string2rle

from dds_cloudapi_sdk import Config, Client
from dds_cloudapi_sdk.tasks.v2_task import V2Task

class DINOX:
    def __init__(self):
        token = os.getenv("DDS_CLOUDAPI_TEST_TOKEN")
        # print(f"token: {token}")
        if not token:
            raise ValueError("API Token not found. Please set DDS_CLOUDAPI_TEST_TOKEN environment variable.")
        self.config = Config(token)
        self.client = Client(self.config)
        
        
    def get_dinox(self, image_path, input_prompts=None): 
        if image_path.startswith(('http://', 'https://')):
            infer_image_url = image_path
        else:
            # print(f"Uploading image {image_path}")
            infer_image_url = self.client.upload_file(image_path)
        
        text_prompt = "<prompt_free>" if input_prompts is None else input_prompts
        
        task = V2Task(api_path="/v2/task/dinox/detection", api_body={
            "model": "DINO-X-1.0",
            "image": infer_image_url,
            "prompt": {
                "type": "text",
                "text": text_prompt,
            },
            "targets": ["bbox", "mask"],
            "bbox_threshold": 0.25,
            "iou_threshold": 0.85
        })
        task.set_request_timeout(15)
        self.client.run_task(task)
        # import ipdb; ipdb.set_trace()
        return task.result['objects']

    def get_dinox_seek(self, image_path, input_prompts=None):
        # Warning: DINO-XSeek-1.0 is not available in the public version of DDS Cloud API
        # TODO contact IDEA for access
        if image_path.startswith(('http://', 'https://')):
            infer_image_url = image_path
        else:
            # print(f"Uploading image {image_path}")
            infer_image_url = self.client.upload_file(image_path)
            
        text_prompt = "<prompt_free>" if input_prompts is None else input_prompts
        
        task = V2Task(api_path="/v2/task/dinox_xseek/detection", api_body={
            "model": "DINO-XSeek-1.0",
            "image": infer_image_url,
            "prompt": {
                "type": "text",
                "text": text_prompt,
            },
            "targets": ["bbox", "mask"],
            "bbox_threshold": 0.20,
            "iou_threshold": 0.85
        })
        task.set_request_timeout(15)
        self.client.run_task(task)
        # import ipdb; ipdb.set_trace()
        return task.result['objects']


    def visualize_bbox_and_mask(self, predictions, img_path, output_dir, img_name):
        os.makedirs(output_dir, exist_ok=True)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            return None, None
        
        if not predictions:
            cv2.putText(img, "Nothing detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2)
            cv2.imwrite(os.path.join(output_dir, f"{img_name}_bbox_mask.jpg"), img)
            return None, None

        classes, boxes, masks, confidences = [], [], [], []

        # Extract bounding boxes, masks, and other information from predictions
        for obj in predictions:
            if 'bbox' not in obj or 'mask' not in obj:
                print(f"No bbox or mask in {obj}, continue")
                continue
            
            bbox = obj['bbox']
            mask_rle = obj['mask']
            
            if bbox and len(bbox) == 4 and mask_rle:
                x1, y1, x2, y2 = bbox
                boxes.append([x1, y1, x2, y2])
                
                rle_counts = _string2rle(mask_rle['counts'])
                mask = _rle2mask(rle_counts, mask_rle['size']).astype(bool)

                masks.append(mask)
                
                # Add class and confidence
                class_name = obj.get('category')
                classes.append(class_name)
                confidences.append(obj.get('score', 1.0))
        
        # Check if we have any valid detections
        if not boxes:
            print(f"No valid detections in {img_path}")
            return None, None
        
        # Convert to numpy arrays
        boxes_np = np.array(boxes)
        masks_np = np.array(masks)
        
        # Create unique class IDs
        unique_classes = list(set(classes))
        class_id_map = {cls: i for i, cls in enumerate(unique_classes)}
        class_ids = np.array([class_id_map[cls] for cls in classes])
        
        # Create labels
        labels = [
            f"{class_name} {confidence:.2f}"
            for class_name, confidence
            in zip(classes, confidences)
        ]
        
        # Create detections object for visualization
        detections = sv.Detections(
            xyxy=boxes_np,
            mask=masks_np,
            class_id=class_ids,
        )
        
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        mask_annotator = sv.MaskAnnotator()
        annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
        cv2.imwrite(os.path.join(output_dir, f"{img_name}_bbox_mask.jpg"), annotated_frame)
        
        print(f"visualized DINOX results to {output_dir}/{img_name}_bbox_mask.jpg")

        return boxes, masks


if __name__ == "__main__":
    dinox = DINOX()

    frame_path = "~/projects/aawr/data/offline_data/shelf_cabinet/data_0520/success/2025-04-15/2025-04-15_17-27-00/recordings/frames/hand_camera"
    output_dir = "~/projects/aawr/data/offline_data/shelf_cabinet/data_0520/vis/dinox"

    os.makedirs(output_dir, exist_ok=True)

    image_files = [f for f in os.listdir(frame_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    def get_frame_index(filename):
        match = re.search(r'(\d+)', filename)
        return int(match.group(1)) if match else 0
    
    image_files.sort(key=get_frame_index, reverse=True)

    for image_file in image_files:
        input_image = os.path.join(frame_path, image_file)
        print(f"Processing {input_image}")
        
        predictions = dinox.get_dinox(input_image, "pineapple toy . purple object")
        dinox.visualize_bbox_and_mask(predictions, input_image, output_dir, os.path.splitext(image_file)[0])
 