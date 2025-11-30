import os
import sys
import numpy as np
import json
import torch
import cv2
from PIL import Image
import supervision as sv
from pathlib import Path
import warnings
import logging


# Silence specific warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")
warnings.filterwarnings("ignore", category=UserWarning, message="torch.utils.checkpoint")
warnings.filterwarnings("ignore", category=UserWarning, message="None of the inputs have requires_grad=True")
warnings.filterwarnings("ignore", category=UserWarning, message="torch.meshgrid:")
warnings.filterwarnings("ignore", category=FutureWarning, message="You are using `torch.load` with `weights_only=False`")
warnings.filterwarnings("ignore", category=FutureWarning, message="`torch.cuda.amp.autocast(args...)`")
warnings.filterwarnings("ignore", category=FutureWarning, message="final text_encoder_type: bert-base-uncased")

# Silence specific logging messages
logging.getLogger("groundingdino").setLevel(logging.WARNING)

# Add paths for the required models
sys.path.append(os.path.join(os.path.dirname(__file__), "GroundingDINO"))
sys.path.append(os.path.join(os.path.dirname(__file__), "segment-anything"))

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
GROUNDED_CHECKPOINT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights/groundingdino_swint_ogc.pth")
SAM_CHECKPOINT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights/sam_vit_h_4b8939.pth")
# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# Segment Anything Model
from segment_anything import (
    sam_model_registry,
    SamPredictor
)

class GSAMDetector:
    def __init__(
        self,
        config_file = CONFIG_FILE,
        grounded_checkpoint = GROUNDED_CHECKPOINT,
        sam_checkpoint = SAM_CHECKPOINT,
        box_threshold=0.3,
        text_threshold=0.25,
        device="cuda" if torch.cuda.is_available() else "cpu",
        bert_base_uncased_path=None
    ):
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.device = device
        
        self.grounding_dino_model = self._load_grounding_dino_model(
            config_file, 
            grounded_checkpoint, 
            bert_base_uncased_path
        )
        
        self.predictor = SamPredictor(
            sam_model_registry["vit_h"](checkpoint=sam_checkpoint).to(device)
        )
    
    def _load_image(self, image_path):
        # Load image
        image_pil = Image.open(image_path).convert("RGB")
        
        # Transform for Grounding DINO
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image_tensor, _ = transform(image_pil, None)  # 3, h, w
        return image_pil, image_tensor

    def _load_grounding_dino_model(self, model_config_path, model_checkpoint_path, bert_base_uncased_path):
        args = SLConfig.fromfile(model_config_path)
        args.device = self.device
        if bert_base_uncased_path:
            args.bert_base_uncased_path = bert_base_uncased_path
        model = build_model(args)
        checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
        model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        model.eval()
        return model

    def _get_grounding_output(self, image_tensor, text_prompt):
        caption = text_prompt.lower().strip()
        if not caption.endswith("."):
            caption = caption + "."
        
        print(f"Processing prompt: '{caption}'")
        
        # Move model and image to device
        self.grounding_dino_model = self.grounding_dino_model.to(self.device)
        image_tensor = image_tensor.to(self.device)
        
        with torch.no_grad():
            outputs = self.grounding_dino_model(image_tensor[None], captions=[caption])
        
        logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
        
        # Filter out NaN values
        max_logits = logits.max(dim=1)[0]
        valid_mask = ~torch.isnan(max_logits)
        logits = logits[valid_mask]
        boxes = boxes[valid_mask]
        max_logits = max_logits[valid_mask]
        
        # Handle empty tensor case
        if len(max_logits) == 0:
            print("No valid detections after NaN filtering")
            return torch.zeros((0, 4)), []
            
        # Debug: print initial detection counts
        print(f"Initial detections: {len(max_logits)}, max confidence: {max_logits.max().item():.4f}")
        
        # Apply threshold and filter
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > self.box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
        
        # Debug: print filtered counts
        print(f"After filtering (threshold={self.box_threshold}): {boxes_filt.size(0)} boxes remain")
        
        # Get phrases
        tokenlizer = self.grounding_dino_model.tokenizer
        tokenized = tokenlizer(caption)
        pred_phrases = []
        
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(
                logit > self.text_threshold, 
                tokenized, 
                tokenlizer
            )
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        
        # Debug: print detected phrases
        if pred_phrases:
            print(f"Detected phrases: {', '.join(pred_phrases)}")
        else:
            print(f"No phrases detected for threshold={self.text_threshold}")
        
        return boxes_filt, pred_phrases

    def detect(self, image_path, text_prompt):
        """
        Perform object detection and segmentation using Grounding DINO and SAM.
        
        Args:
            image_path: Path to the input image
            text_prompt: Text prompt describing objects to detect
            
        Returns:
            Dict containing detection results:
                - boxes: Bounding boxes in xyxy format
                - masks: Segmentation masks
                - labels: Class labels with confidence scores
                - image: Original image
        """
        image_pil, image_tensor = self._load_image(image_path)
        original_size = image_pil.size  # (width, height)
        
        boxes_filt, pred_phrases = self._get_grounding_output(image_tensor, text_prompt)
        
        # Check if GroundingDINO found any boxes
        if boxes_filt.size(0) == 0:
            print("GroundingDINO found no boxes, skipping SAM.")
            return {
                "image": np.array(image_pil),
                "boxes": np.array([]),
                "masks": np.array([]),
                "class_names": [],
                "confidences": [],
                "labels": []
            }
        
        H, W = original_size[1], original_size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]
        
        boxes_xyxy = boxes_filt.cpu()
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Generate masks with SAM
        self.predictor.set_image(image)
        transformed_boxes = self.predictor.transform.apply_boxes_torch(
            boxes_xyxy, image.shape[:2]
        ).to(self.device)
        
        # Handle potential empty tensor after transformation or if prediction fails
        try:
            masks, _, _ = self.predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )
            masks_np = masks.cpu().numpy()[:, 0, :, :]  # Remove batch dimension
        except Exception as e:
            print(f"Error during SAM prediction: {e}")
            print("Returning empty masks.")
            masks_np = np.zeros((0, H, W), dtype=bool)
        
        # Extract class names and confidences
        class_names = []
        confidences = []
        
        for phrase in pred_phrases:
            try:
                class_name, conf_str = phrase.split('(')
                confidence = float(conf_str[:-1])  # remove the ')'
                class_names.append(class_name.strip())
                confidences.append(confidence)
            except ValueError:
                print(f"Warning: Could not parse phrase '{phrase}'. Skipping.")
                continue
        
        boxes_np = boxes_xyxy.numpy()
        
        if masks_np.shape[0] != boxes_np.shape[0]:
            print(f"Warning: Mismatch between number of boxes ({boxes_np.shape[0]}) and masks ({masks_np.shape[0]}). Using empty masks.")
            masks_np = np.zeros((boxes_np.shape[0], H, W), dtype=bool)
        
        return {
            "image": image,
            "boxes": boxes_np,
            "masks": masks_np,
            "class_names": class_names,
            "confidences": confidences,
            "labels": pred_phrases
        }
    
    def visualize_bbox_and_mask(self, results, output_dir, img_name):
        """Visualize detection results and save them to disk"""
        os.makedirs(output_dir, exist_ok=True)
        
        img = results["image"]
        boxes = results["boxes"]
        masks = results["masks"]
        class_names = results["class_names"]
        confidences = results["confidences"]
        
        if len(boxes) == 0 or np.isnan(confidences).any():
            # Save empty image with text indicating no detection
            annotated_frame = img.copy()
            annotated_frame = cv2.putText(
                annotated_frame, 
                "No Detection", 
                (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 0, 255), 
                2
            )
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(output_dir, f"{img_name}_bbox_mask.jpg"), annotated_frame)
            print(f"No valid detections found, saving empty image to {output_dir}/{img_name}_bbox_mask.jpg")
            return [], []
        
        class_ids = np.array([i for i, _ in enumerate(class_names)])
        labels = [f"{cls} {conf:.2f}" for cls, conf in zip(class_names, confidences)]
        detections = sv.Detections(
            xyxy=boxes,
            mask=masks, 
            class_id=class_ids,
            confidence=np.array(confidences)
        )
        
        box_annotator = sv.BoxAnnotator()
        mask_annotator = sv.MaskAnnotator()
        label_annotator = sv.LabelAnnotator()
        
        annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(output_dir, f"{img_name}_bbox_mask.jpg"), annotated_frame)
        
        print(f"visualized DINOX results to {output_dir}/{img_name}_bbox_mask.jpg")

        return boxes, masks
# Test
if __name__ == "__main__":
    detector = GSAMDetector(
        box_threshold=0.3,
        text_threshold=0.25
    )
    
    image_path = "~/projects/aawr/reward_pipeline/color_seg_config/example_pineapple.jpg"
    text_prompt = "yellow pineapple toy. toy"
    output_dir = "results/gsam"
    
    results = detector.detect(image_path, text_prompt)
    
    image_name = Path(image_path).stem
    detector.visualize_bbox_and_mask(results, output_dir, image_name)
