DATA_ROOT="~/projects/aawr/data/offline_data/shelf_cabinet/data_complex_0520"
set -e


TRUNCATION_VIS_DIR="$DATA_ROOT/truncation_vis_new"
COLOR_SEG_EXAMPLE_IMAGE="~/projects/aawr/reward_pipeline/color_seg_config/example_pineapple.jpg"
NUM_WORKERS=4
# arguments for defining the target region for reward features.
TARGET_CXCY="640,200" # center of target region for reward
TARGET_SIDE_LENGTH=400

# OBJECT_NAME="yellow duck toy"
OBJECT_NAME="yellow pineapple toy"
# 1. drop out actions with low magnitude and create whitelist of valid frames in white_list.json
python reward_pipeline/truncate_droid_traj.py $DATA_ROOT $TRUNCATION_VIS_DIR
# 2. label the valid frames with DINOX
python reward_pipeline/vision/get_dinox_info.py $DATA_ROOT $NUM_WORKERS "$OBJECT_NAME"

