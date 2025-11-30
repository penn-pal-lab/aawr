DATA_ROOT="~/projects/aawr/data/offline_data/shelf_cabinet/data_complex_0520"


TRUNCATION_VIS_DIR="$DATA_ROOT/truncation_vis_new_comp"
COLOR_SEG_EXAMPLE_IMAGE="~/projects/aawr/reward_pipeline/color_seg_config/example_pineapple.jpg"
OBJECT="yellow pineapple toy"
set -e

# arguments for defining the target region for reward features.
TARGET_CXCY="640,200" # center of target region for reward
TARGET_SIDE_LENGTH=400

# 3. Compute color segmentation masks for each valid frame.
python reward_pipeline/vision/create_color_segmentation.py $DATA_ROOT $COLOR_SEG_EXAMPLE_IMAGE

# 4. Fuse together DINOX mask, bbox and color segmentaiton mask into final mask / bbox
python reward_pipeline/reward/prepare_reward_masks.py $DATA_ROOT

# 5. Given the fused masks/bbox, compute the reward features 
python reward_pipeline/compute_reward_features.py $DATA_ROOT $TARGET_CXCY $TARGET_SIDE_LENGTH

# 6. Now compute reward by weighting the reward features
python reward_pipeline/compute_reward.py $DATA_ROOT
