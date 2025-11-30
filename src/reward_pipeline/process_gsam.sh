# We also provide a GSAM version to generate the reward.
DATA_ROOT="~/projects/aawr/data/offline_data/shelf_cabinet/data_complex_0520"
set -e
OBJECT_NAME="yellow pineapple toy"
TRUNCATION_VIS_DIR="$DATA_ROOT/truncation_vis_new"
COLOR_SEG_EXAMPLE_IMAGE="~/projects/aawr/reward_pipeline/color_seg_config/example_pineapple.jpg"
NUM_WORKERS=1
# arguments for defining the target region for reward features.
TARGET_CXCY="640,200" # center of target region for reward
TARGET_SIDE_LENGTH=400

echo "DATA_ROOT: ${DATA_ROOT}"
echo "OBJECT_NAME: ${OBJECT_NAME}"
echo "start processing gsam...! "
read -p "Press Enter to continue"
# 1. drop out actions with low magnitude and create whitelist of valid frames in white_list.json
python reward_pipeline/truncate_droid_traj.py $DATA_ROOT $TRUNCATION_VIS_DIR
# 2. label the valid frames with GSAM
python reward_pipeline/vision/get_gsam_info.py $DATA_ROOT $NUM_WORKERS "$OBJECT_NAME"