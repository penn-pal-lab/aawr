#!/usr/bin/env python3
"""
Prepare reward mask for each episode

Output:
  - fused_masks/*.npy           (uint8 0/1)
  - reward_info.json            (time-series reward info)

Author: Tony (2025-04-19)
"""

from __future__ import annotations
from pathlib import Path
import numpy as np, json, cv2, logging
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

import os
import sys

# Prepare reward masks for each episode, for downstream reward estimation.

## really simple, just : 
# - fuse mask to generate the fuse_mask
# - prepare bbox  & cxcy
# - prepare conf

'''
"mask_info": {
    "1": {
      "category": "yellow pineapple toy",
      "bbox": [],
      "cxcy": [], 
      "fused_mask": "fused_masks/00001.npy",
      "fused_source": "color",
      "mask_area": 17894,
      "conf": 0.5
    },
    "2": {
      "category": "yellow pineapple toy",
      "bbox": [100, 100, 200, 200], # xyxy x1, y1, x2, y2
      "cxcy": [150,150],
      "fused_mask": "fused_masks/00002.npy",
      "fused_source": "dinox",
      "mask_area": 5751,
      "conf": 0.5
    },
    "3": {
      "category": "yellow pineapple toy",
      "bbox": [], 
      "cxcy": [],
      "fused_mask": "fused_masks/00003.npy",
      "fused_source": "none",
      "mask_area": 0,
      "conf": 0.0
    },
    "4": {
      "category": "yellow pineapple toy",
      "bbox": [100, 100, 200, 200], # from dinox
      "cxcy": [150,150],
      "fused_mask": "fused_masks/00004.npy",
      "fused_source": "fused",
      "mask_area": 47041,
      "conf": 0.5
    },

'''


# --------------------------------------------------------------------------- #
# -------------------------  FUSION & SCORING LOGIC  ------------------------ #
# --------------------------------------------------------------------------- #

AREA_THR = 500          # Pixel threshold, too small is considered invalid color mask
COLOR_CONF = 0.3       # Confidence value when only color mask is available

# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #
def _bbox_from_mask(mask: np.ndarray) -> list[int]:
    """Return [x1,y1,x2,y2] ‑‑ empty list if mask is empty."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return []
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

def _center_from_bbox(bbox: list[int]) -> list[float]:
    if not bbox:
        return []
    x1, y1, x2, y2 = bbox
    return [(x1 + x2) / 2.0, (y1 + y2) / 2.0]

# --------------------------------------------------------------------------- #
# Main fusion function
# --------------------------------------------------------------------------- #
def fuse_mask_bbox(
    dinfo_frame: list[dict],
    color: np.ndarray,
    masks_dir: Path,
) -> tuple[np.ndarray, float, str, list[int], list[float], int]:
    """
    Returns
    -------
    mask       : uint8 {0,1}
    conf       : float
    src        : "dinox" | "color" | "fused" | "none"
    bbox       : [x1,y1,x2,y2]   (empty list if none)
    cxcy       : [cx,cy]         (empty list if none)
    mask_area  : int             (#pixels ==1)
    """
    color_ok = color.sum() >= AREA_THR

    # -------------------- collect DINO‑X candidates -------------------- #
    candidates = []          # each = {"score":float, "mask":ndarray}
    for obj in dinfo_frame:
        mf = obj.get("mask_file", "")
        if not mf:
            continue
        dmask = np.load(masks_dir / mf)
        candidates.append({"score": obj.get("score", 0.0), "mask": dmask})

    # --------------------------- no DINO‑X ------------------------------ #
    if not candidates:
        if color_ok:
            mask = color.astype(np.uint8)
            src  = "color"
        else:
            mask = np.zeros_like(color, dtype=np.uint8)
            src  = "none"
        conf = 0.0
        bbox  = []
        cxcy  = []
        area  = int(mask.sum())
        return mask, conf, src, bbox, cxcy, area

    # ----------------------- choose best DINO‑X mask -------------------- #
    # Step‑1: intersect criterion (if colour exists)
    if len(candidates) > 1 and color_ok:
        # pick cand with the largest intersection (#pixel) with colour mask
        best_idx, best_inter = -1, -1
        for i, c in enumerate(candidates):
            inter = int((c["mask"] & color).sum())
            if inter > best_inter:
                best_inter = inter
                best_idx   = i
        if best_inter > 0:
            best_mask = candidates[best_idx]["mask"]
            best_conf = candidates[best_idx]["score"]
        else:
            # fall back to highest score from DINO-X
            best_mask = max(candidates, key=lambda c: c["score"])["mask"]
            best_conf = max(candidates, key=lambda c: c["score"])["score"]
    else:
        # only one candidate OR no colour reference
        best_mask = max(candidates, key=lambda c: c["score"])["mask"]
        best_conf = max(candidates, key=lambda c: c["score"])["score"]

    # --------------------------- colour fusion -------------------------- #
    if color_ok:
        inter_ratio = (best_mask & color).sum() / best_mask.sum()
        if 0: # inter_ratio > 0.30: # try later
            
            fused = np.logical_or(best_mask, color).astype(np.uint8)
            src   = "fused"
            mask  = fused
        else:
            src   = "dinox"
            mask  = best_mask.astype(np.uint8)
    else:
        src  = "dinox"
        mask = best_mask.astype(np.uint8)

    bbox = _bbox_from_mask(mask)
    cxcy = _center_from_bbox(bbox)
    area = int(mask.sum())
    return mask, best_conf, src, bbox, cxcy, area



# --------------------------------------------------------------------------- #
# ----------------------  EPISODE-LEVEL PROCESSOR  -------------------------- #
# --------------------------------------------------------------------------- #

def build_reward_masks(episode_root: Path):
    # ---- 1. Load Data ---- #
    dinox_info = json.load(open(episode_root / "dinox_info.json"))
    white_list = json.load(open(episode_root / "white_list.json"))
    packed_data = np.load(episode_root / "color_segmentation_npy/new_masks_packed.npz")
    packed_masks = packed_data['packed_masks']
    original_shape = packed_data['original_shape']
    packed_mask_idxs = packed_data['idxs']
    color_masks = np.unpackbits(packed_masks).reshape(original_shape)
    masks_dir   = episode_root / "masks"
    fused_dir   = episode_root / "fused_masks"
    fused_dir.mkdir(exist_ok=True)

    # Track additional data for visualization
    areas = {}
    sources = {}
    confidences = {}
    
    # Store mask metadata in reward_mask.json
    reward_data = {}

    # ---- 2. Main Loop ---- #
    color_mask_dict = dict(zip(packed_mask_idxs, color_masks))
    for idx in white_list:
        frame_key = str(idx)
        dinfo     = dinox_info.get(frame_key, [])
        color    = color_mask_dict[idx].astype(bool)
        # TO Edward: here is the index bug
        
        mask, conf, src, bbox, cxcy, area = fuse_mask_bbox(dinfo, color, masks_dir)

        # Save the packed masks and the original shape efficiently
        output_npz_path = os.path.join(episode_root, "fused_masks", f"{idx:05d}.npz")
        original_shape = mask.shape
        packed_masks = np.packbits(mask, axis=-1)
        # Save the packed masks and the original shape efficiently
        np.savez_compressed(output_npz_path, packed_masks=packed_masks, original_shape=original_shape)

        areas[frame_key] = area
        sources[frame_key] = src
        confidences[frame_key] = conf
        
        reward_data[frame_key] = {
            "category": "yellow pineapple toy",
            "fused_mask": output_npz_path,
            "fused_source": src,
            "mask_area": area,
            "mask_conf": conf,
            "bbox": bbox,
            "cxcy": cxcy
        }
    with open(episode_root / "reward_mask_info.json", "w") as fp:
        json.dump(reward_data, fp, indent=2)
    

    logger.info("Episode %s  →  reward mask done (%d frames)",
                episode_root.name, len(white_list))
                
    return white_list, areas, sources, confidences


def visualize_reward_masks(episode_root, white_list, areas, sources, confidences):
    # load the reward_mask_info.json
    reward_mask_info = json.load(open(episode_root / "reward_mask_info.json"))

    os.makedirs(episode_root / "fused_masks_vis", exist_ok=True)
    images_path = os.path.join(episode_root, "recordings", "frames", "hand_camera")
    sorted_images = sorted(os.listdir(images_path))
    idx = 0 
    for i, image_dir in enumerate(sorted_images):
        if i not in white_list:
            continue
        info = reward_mask_info[str(i)]
        bbox = info['bbox']
        image_path = os.path.join(images_path, image_dir)
        fused_mask_path = os.path.join(episode_root, "fused_masks", f"{i:05d}.npz")
        data = np.load(fused_mask_path)
        packed_mask = data['packed_masks']
        original_shape = data['original_shape']
        # Unpack the masks using unpackbits and reshape to original dimensions
        mask = np.unpackbits(packed_mask).reshape(original_shape).astype(bool)
        # load the image 
        image = cv2.imread(image_path)
        image[mask == 0] //= 5

        if len(bbox) > 0:
            # import ipdb; ipdb.set_trace()
            corner1, corner2 = (bbox[0], bbox[1]), (bbox[2], bbox[3])
            cv2.rectangle(image, corner1, corner2, (0, 0, 255), 2)
            # write the confidence value next to the bbox
            cv2.putText(image, f"{confidences[str(i)]}", (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imwrite(os.path.join(episode_root, "fused_masks_vis", f"{i:05d}.jpg"), image)
        idx += 1
    

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    TRAJ_ROOT = sys.argv[1]
    DATA_DIRS = ["success", "failure"]
    for data_dir in DATA_DIRS:
        for traj_dir in os.listdir(os.path.join(TRAJ_ROOT, data_dir)):
            for sub_traj_dir in os.listdir(os.path.join(TRAJ_ROOT, data_dir, traj_dir)):
                EP = Path(os.path.join(TRAJ_ROOT, data_dir, traj_dir, sub_traj_dir))
                white_list, areas, sources, confidences = build_reward_masks(EP)
                visualize_reward_masks(EP, white_list, areas, sources, confidences)
