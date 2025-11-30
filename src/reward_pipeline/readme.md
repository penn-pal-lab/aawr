# Reward Pipeline for Active Perception Training
Jie Wang

- 04/18/2025 

This document outlines the reward preparation process for the AAWR on DROID. The pipeline generates rewards for training an active perception model using wrist camera data.

## Input Data Structure

The reward pipeline expects each episode to have the following structure:

```
episode_root/
├── calibration.json                 # Camera calibration information
├── color_segmentation_npy/          
│   └── masks.npy                    # Color segmentation masks (T×H×W)
├── dinox_info.json                  # DINO-X object detection results
├── masks/                           # Individual DINO-X masks
├── recordings/                      # Raw camera recordings
│   ├── frames/
│   ├── hand_camera.mp4
│   └── ...
├── trajectory.h5                    # Robot trajectory data
└── white_list.json                  # Valid frame indices
```

### Key Input Files

1. **white_list.json**: Contains a list of valid frame indices to process.
   ```json
   [3, 4, 5, 6, 7, ..., 28]
   ```

2. **dinox_info.json**: Contains DINO-X detection results for each frame.
   ```json
   {
     "3": [
       {
         "score": 0.85,
         "category": "yellow pineapple toy",
         "mask_file": "00003.npy"
       }
     ],
     "4": [...],
     ...
   }
   ```

3. **color_segmentation_npy/masks.npy**: Raw color segmentation masks.

4. **trajectory.h5**: Contains robot state information including cartesian positions.

## Output Data Structure

The pipeline generates the following outputs:

```
episode_root/
├── fused_masks/                     # Fused masks (DINO-X + color)
│   ├── 00003.npz                    # Compressed binary masks
│   ├── 00004.npz
│   └── ...
├── fused_masks_vis/                 # Optional visualizations
├── reward_mask_info.json            # Metadata for masks
├── reward_components.json           # Individual reward components
└── reward.npy                       # Final weighted rewards
```

### Key Output Files

1. **fused_masks/*.npz**: Compressed binary masks from fusing DINO-X and color segmentation.

2. **reward_info.json**: Contains mask metadata for each frame.
   ```json
   {
     "3": {
       "category": "yellow pineapple toy",
       "fused_mask": "fused_masks/00003.npz",
       "fused_source": "dinox",
       "mask_area": 17894,
       "mask_conf": 0.85,
       "bbox": [100, 100, 300, 350],
       "cxcy": [200, 225]
     },
     "4": {...},
     ...
   }
   ```

   ```

4. **reward.npy**: Final weighted rewards as a numpy array with shape (T,) where T is the number of frames in white_list.json.