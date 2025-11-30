import os
import sys
import json 
import numpy as np
import matplotlib.pyplot as plt

"""
Load the reward features for each episode, and then compute the reward and save it.
We have the following features:
reward_features = {
    "num_pixels": num_pixels,
    "mask_overlap_ratio": mask_overlap_ratio,
    "distance": distance,
    "valid_detection": valid_detection
}

And we want to define three reward terms, before scaling. 

1. distance reward - a dense reward signal between 0 and 1, based on the distance to the target.
2. mask overlap reward - a sparse reward that is 1 if mask overlap ratio is above a threshold.
3. mask area reward - a dense reward  between 0 and 1, based on number of pixels in the mask.  
"""

def compute_mask_area_reward(num_pixels, max_area=50000):
    # number of pixels is typically between 0 and 50k. 
    # clip to remove outlier cases. 
    num_pixels = np.clip(num_pixels, 1000, max_area)
    mask_area_reward = num_pixels / max_area
    return mask_area_reward # between 0 and 1.

def compute_distance_reward(distance, tanh_scale=10):
    # tanh_scale affects how fast the reward drops off.
    # since image is 1280 x 720, maximum distance between a pixel and the target is ~1000 
    distance = np.clip(distance, 0, 1000)
    try:
        distance /= 1000.0 # now it's between 0 and 1.
    except:
        import ipdb; ipdb.set_trace()
    distance_reward = 1 - np.tanh(tanh_scale * distance)
    return distance_reward # between 0 and 1.
    
def compute_mask_overlap_reward(mask_overlap_ratio, threshold=0.1):
    mask_overlap_reward = np.where(mask_overlap_ratio > threshold, 1, 0)
    return mask_overlap_reward # between 0 and 1.

def compute_conf_reward(conf_reward, threshold=0.5):
    conf_reward = np.where(conf_reward > threshold, 1, 0)
    return conf_reward # between 0 and 1.

if __name__ == "__main__":
    DATA_DIRS = ["success", "failure"]
    TRAJ_ROOT = sys.argv[1]

    tanh_scale = 5
    threshold = 0.1
    conf_threshold = 0.5
    max_mask_area = 50000

    area_reward_weight = 1
    distance_reward_weight = 1
    mask_overlap_reward_weight = 10
    conf_reward_weight = 5

    for data_dir in DATA_DIRS:
        # traverse the subfolders in traj_dir/data_dir
        for traj_dir_date in os.listdir(os.path.join(TRAJ_ROOT, data_dir)):
            if os.path.isdir(os.path.join(TRAJ_ROOT, data_dir, traj_dir_date)):
                for sub_traj_dir in os.listdir(os.path.join(TRAJ_ROOT, data_dir, traj_dir_date)):
                    full_traj_path = os.path.join(TRAJ_ROOT, data_dir, traj_dir_date, sub_traj_dir)
                    if os.path.isdir(full_traj_path):
                        print("=COMPUTING FINAL REWARD=", full_traj_path)
                        reward_features = json.load(open(os.path.join(full_traj_path, "reward_features.json")))
                        num_pixels = np.array(reward_features["num_pixels"])
                        mask_overlap_ratio = np.array(reward_features["mask_overlap_ratio"])
                        distance = np.array(reward_features["distance"])
                        valid_detection = np.array(reward_features["valid_detection"])
                        # conf_reward = np.array(reward_features["conf_reward"]) # we finally did not use this because it is noisy and not calibrated.
                        
                        # compute the reward terms, weighting them by the weights.
                        area_reward = compute_mask_area_reward(num_pixels, max_mask_area) * area_reward_weight
                        distance_reward = compute_distance_reward(distance, tanh_scale) * distance_reward_weight
                        mask_overlap_reward = compute_mask_overlap_reward(mask_overlap_ratio, threshold) * mask_overlap_reward_weight
                        # conf_reward = compute_conf_reward(conf_reward, conf_threshold) * conf_reward_weight

                        # mask out rewards for invalid detections.
                        area_reward = area_reward * valid_detection
                        distance_reward = distance_reward * valid_detection
                        mask_overlap_reward = mask_overlap_reward * valid_detection
                        # conf_reward = conf_reward * valid_detection
                        # Calculate total reward
                        total_reward = area_reward + distance_reward + mask_overlap_reward  # + conf_reward

                        np.save(os.path.join(full_traj_path, "computed_reward.npy"), total_reward)

                        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
                        time_steps = np.arange(len(area_reward))
                        
                        ax1.plot(time_steps, area_reward, label='Area Reward', marker='.')
                        ax1.plot(time_steps, distance_reward, label='Distance Reward', marker='.')
                        ax1.plot(time_steps, mask_overlap_reward, label='Mask Overlap Reward', marker='.')
                        ax1.set_title('Individual Reward Components')
                        ax1.set_xlabel('Time Step')
                        ax1.set_ylabel('Reward Value')
                        ax1.legend()
                        ax1.grid(True)
                        
                        ax2.plot(time_steps, total_reward, label='Total Reward', color='purple', marker='.')
                        ax2.set_title('Total Reward')
                        ax2.set_xlabel('Time Step')
                        ax2.set_ylabel('Reward Value')
                        ax2.legend()
                        ax2.grid(True)
                        
                        plt.tight_layout()
                        
                        plt.savefig(os.path.join(full_traj_path, 'rewards_plot.png'))
                        plt.close()