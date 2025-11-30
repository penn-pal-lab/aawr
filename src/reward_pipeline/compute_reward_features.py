import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt


def compute_bbox_dist_to_target(cxcy, target_cxcy):
    distance = np.linalg.norm(cxcy - target_cxcy, axis=-1)
    return distance

def compute_bbox_conf(cxcy, conf):
    # conf is a 2D array of shape (N, 1)
    # TODO check this
    return conf


def compute_sufficient_mask_overlap(target_cxcy, mask, target_side_length=200):
    # compute the number of pixels in the mask that are inside a square of side length target_side_length centered at target_cxcy
    target_x, target_y = target_cxcy
    half_side = target_side_length / 2.0

    # Calculate the boundaries of the square region
    min_x = int(round(target_x - half_side))
    max_x = int(round(target_x + half_side)) # Exclusive index for slicing
    min_y = int(round(target_y - half_side))
    max_y = int(round(target_y + half_side)) # Exclusive index for slicing

    # Get mask dimensions (assuming H, W format)
    H, W = mask.shape

    # Clamp coordinates to be within mask bounds
    start_col = max(0, min_x)
    end_col = min(W, max_x)
    start_row = max(0, min_y)
    end_row = min(H, max_y)

    # Extract the sub-mask corresponding to the square region
    sub_mask = mask[start_row:end_row, start_col:end_col]

    # Compute the sum of pixel values within the sub-mask
    # This assumes mask contains boolean (True/False) or numerical (e.g., 0/1) values
    pixel_count = np.sum(sub_mask)

    # compute the overlap ratio
    overlap_ratio = pixel_count / (target_side_length * target_side_length)
    return overlap_ratio

# def compute_area_reward(all_masks):
#     return all_masks.sum((1,2))
    
if __name__ == "__main__":
    DATA_DIRS = ["success", "failure"]
    # take in command line argument as TRAJ_ROOT, VIS_DIR
    TRAJ_ROOT = sys.argv[1]
    # convert system argument string to tuple
    TARGET_CXCY = tuple(map(int, sys.argv[2].split(',')))
    TARGET_SIDE_LENGTH = int(sys.argv[3])
    for data_dir in DATA_DIRS:
        # traverse the subfolders in traj_dir/data_dir
        for traj_dir_date in os.listdir(os.path.join(TRAJ_ROOT, data_dir)):
            if os.path.isdir(os.path.join(TRAJ_ROOT, data_dir, traj_dir_date)):
                for sub_traj_dir in os.listdir(os.path.join(TRAJ_ROOT, data_dir, traj_dir_date)):
                    full_traj_path = os.path.join(TRAJ_ROOT, data_dir, traj_dir_date, sub_traj_dir)
                    if os.path.isdir(full_traj_path):
                        print("=======", full_traj_path)
                        # white_list_path = os.path.join(full_traj_path, "white_list.json")
                        # white_list = json.load(open(white_list_path))
                        reward_mask_info = json.load(open(os.path.join(full_traj_path, "reward_mask_info.json")))
                        num_pixels = []
                        mask_overlap_ratio = []
                        distance = []
                        valid_detection = []
                        conf_reward = []
                        for k, v in reward_mask_info.items():
                            valid_detection.append(float(len(v['cxcy']) > 0))
                            if len(v['cxcy']) == 0:
                                num_pixels.append(0)
                                mask_overlap_ratio.append(0)
                                distance.append(0)
                            else:
                                fused_mask_path =v['fused_mask']
                                packed_data = np.load(fused_mask_path)
                                packed_masks = packed_data['packed_masks']
                                original_shape = packed_data['original_shape']
                                mask = np.unpackbits(packed_masks).reshape(original_shape)
                                num_pixels.append(np.sum(mask).item())

                                overlap_ratio = compute_sufficient_mask_overlap(TARGET_CXCY, mask, target_side_length=TARGET_SIDE_LENGTH)
                                mask_overlap_ratio.append(overlap_ratio)

                                cxcy = np.array(v['cxcy'])
                                conf = np.array(v['mask_conf'])
                                distance.append(compute_bbox_dist_to_target(cxcy, TARGET_CXCY).item())
                                conf_reward.append(compute_bbox_conf(cxcy, conf).item())

                        reward_features = {
                            "num_pixels": num_pixels,
                            "mask_overlap_ratio": mask_overlap_ratio,
                            "distance": distance,
                            "valid_detection": valid_detection,
                            "conf_reward": conf_reward
                        }

                        with open(os.path.join(full_traj_path, "reward_features.json"), "w") as f:
                            # import ipdb; ipdb.set_trace()
                            json.dump(reward_features, f, indent=4)

                        # Create visualization plots
                        time_steps = np.arange(len(num_pixels))
                        
                        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
                        fig.suptitle(f'Reward Features - {sub_traj_dir}')
                        
                        # Convert valid_detection to boolean for easier masking
                        valid_detection_bool = np.array(valid_detection, dtype=bool)
                        
                        # Function to add gray background for invalid regions
                        def add_invalid_regions(ax, time_steps, valid_detection_bool):
                            # Find the start and end of invalid regions
                            invalid_regions = np.where(~valid_detection_bool)[0]
                            if len(invalid_regions) > 0:
                                # Group consecutive invalid regions
                                regions = []
                                start = invalid_regions[0]
                                for i in range(1, len(invalid_regions)):
                                    if invalid_regions[i] != invalid_regions[i-1] + 1:
                                        regions.append((start, invalid_regions[i-1]))
                                        start = invalid_regions[i]
                                regions.append((start, invalid_regions[-1]))
                                
                                # Add gray background for each invalid region
                                for start_idx, end_idx in regions:
                                    ax.axvspan(time_steps[start_idx], time_steps[end_idx], 
                                             color='gray', alpha=0.2)
                        
                        # Plot number of pixels
                        valid_num_pixels = np.array(num_pixels) * np.array(valid_detection)
                        valid_time_steps = time_steps[valid_detection_bool]
                        valid_num_pixels = valid_num_pixels[valid_detection_bool]
                        ax1.scatter(valid_time_steps, valid_num_pixels, c='b', s=10)
                        add_invalid_regions(ax1, time_steps, valid_detection_bool)
                        ax1.set_title('Number of Pixels in Mask')
                        ax1.set_xlabel('Time Step')
                        ax1.set_ylabel('Pixel Count')
                        ax1.grid(True)
                        
                        # Plot mask overlap
                        valid_mask_overlap_ratio = np.array(mask_overlap_ratio) * np.array(valid_detection)
                        valid_mask_overlap_ratio = valid_mask_overlap_ratio[valid_detection_bool]
                        ax2.scatter(valid_time_steps, valid_mask_overlap_ratio, c='g', s=10)
                        add_invalid_regions(ax2, time_steps, valid_detection_bool)
                        ax2.set_title('Mask Overlap with Target Region')
                        ax2.set_xlabel('Time Step')
                        ax2.set_ylabel('Overlap Ratio')
                        ax2.grid(True)
                        
                        # Plot distance to target
                        valid_distance = np.array(distance) * np.array(valid_detection)
                        valid_distance = valid_distance[valid_detection_bool]
                        ax3.scatter(valid_time_steps, valid_distance, c='r', s=10)
                        add_invalid_regions(ax3, time_steps, valid_detection_bool)
                        ax3.set_title('Distance to Target')
                        ax3.set_xlabel('Time Step')
                        ax3.set_ylabel('Distance (pixels)')
                        ax3.grid(True)
                        
                        plt.tight_layout()
                        
                        # Save the plot
                        plot_path = os.path.join(full_traj_path, "reward_features_plot.png")
                        plt.savefig(plot_path)
                        plt.close()
