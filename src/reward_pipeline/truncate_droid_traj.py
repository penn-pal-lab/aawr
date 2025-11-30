import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import json
import sys  

def get_traj_truncated_indices(full_traj_path):
    # if is h5 file and the recordings folder has a subfolder called "frames"
    if os.path.exists(os.path.join(full_traj_path, "recordings", "frames")):
        # print(os.path.join(data_dir, traj_dir, file))
        file = os.path.join(full_traj_path, "trajectory.h5")
        data = h5py.File(file, "r")
        action_dict = data["action"]
        act_cartesian_velocity = action_dict["cartesian_velocity"][:]
        # import ipdb; ipdb.set_trace()
        act_ee_velocity = act_cartesian_velocity[:,0:3] # Update: included rpy as well.

        max_velocity = np.max(np.abs(act_ee_velocity), axis=0)
        max_velocity_5_perc = max_velocity * 0.05
        # find the first and lastindex where velocity in some axis is greater than 0.005
        greater_than_005 = np.abs(act_ee_velocity) > max_velocity_5_perc
        start_index = np.where(greater_than_005)[0][0]
        end_index = np.where(np.abs(act_ee_velocity) > max_velocity_5_perc)[0][-1]
        end_index += 1 # because we want to include the last frame.
        print(f"{end_index - start_index} frames kept out of {len(act_ee_velocity)}")
        return start_index, end_index
    else:
        raise ValueError(f"No suitable .h5 file and recordings/frames directory found for {full_traj_path}")

def get_zero_velocity_skipped_white_list(full_traj_path, traj_dir, vis_dir, traj_root):
    # if is h5 file and the recordings folder has a subfolder called "frames"
    os.makedirs(vis_dir, exist_ok=True)
    if os.path.exists(os.path.join(full_traj_path, "recordings", "frames")):
        file = os.path.join(full_traj_path, "trajectory.h5")
        data = h5py.File(file, "r")
        action_dict = data["action"]
        act_cartesian_velocity = action_dict["cartesian_velocity"][:]
        act_ee_velocity = act_cartesian_velocity[:,0:6] # Update: included rpy as well.
        # find the indices where all elements in act_ee_velocity are close to zero
        zero_velocity_indices = np.where(np.any(np.abs(act_ee_velocity) >= 1e-6, axis=1))[0]
        print(f"{len(zero_velocity_indices)} frames extracted out of {len(act_ee_velocity)}")
        old_head, old_tail = get_traj_truncated_indices(full_traj_path)
        diff_sum = abs((old_tail - old_head) - len(zero_velocity_indices))
        print(f"old white list: {old_head} to {old_tail}")
        visualize_old_new_comparison(zero_velocity_indices, list(range(old_head, old_tail)), traj_root, traj_dir, vis_dir, act_ee_velocity)
        # save the white list to a json file
        with open(os.path.join(full_traj_path, "white_list.json"), "w") as f:
            zero_velocity_indices_list = zero_velocity_indices.tolist()
            json.dump(zero_velocity_indices_list, f)
        
        return zero_velocity_indices, diff_sum
    else:
        raise ValueError(f"No suitable .h5 file and recordings/frames directory found for {full_traj_path}")

def visualize_old_new_comparison(new_white_list, old_white_list, traj_root, traj_dir, vis_dir, act_ee_velocity):
    # Create a figure for comparing old and new white lists
    plt.figure(figsize=(12, 6))
    
    # Find the range for x-axis
    min_idx = min(min(old_white_list) if len(old_white_list) > 0 else float('inf'), 
                 min(new_white_list) if len(new_white_list) > 0 else float('inf'))
    max_idx = max(max(old_white_list) if len(old_white_list) > 0 else 0, 
                 max(new_white_list) if len(new_white_list) > 0 else 0) + 1
    
    # Create x-axis range
    x_range = range(min_idx, max_idx)
    
    # Create binary arrays for old and new white lists
    old_binary = [1 if i in old_white_list else 0 for i in x_range]
    new_binary = [1 if i in new_white_list else 0 for i in x_range]
    
    # Plot the binary arrays
    plt.plot(x_range, old_binary, 'r-', label='Old White List', alpha=0.7)
    plt.plot(x_range, new_binary, 'b-', label='New White List', alpha=0.7)
    
    plt.xlabel('Frame Index')
    plt.ylabel('Included in White List (1=Yes, 0=No)')
    plt.title(f'Comparison of Old and New White Lists for {traj_dir}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Create directory if it doesn't exist
    save_path = os.path.join(vis_dir, f"{traj_dir}_WL_comparison.png")
    # print(f"Saving to: {save_path}")
    # print(f"New white list: {new_white_list}")
    # print(f"Old white list: {old_white_list}")
    # plt.show()
    plt.savefig(save_path)
    plt.close()


    fig, ax = plt.subplots(6, 1)
    # draw vertical lines at the first and last index of the old white list
    ax[0].axvline(x=old_white_list[0], color='r', linestyle='--')
    ax[0].axvline(x=old_white_list[-1], color='r', linestyle='--')
    # draw dots at the indices of the new white list
    ax[0].plot(range(len(act_ee_velocity)), act_ee_velocity[:,0])
    ax[0].plot(new_white_list, [act_ee_velocity[i,0] for i in new_white_list], 'go')
    
    ax[1].axvline(x=old_white_list[0], color='r', linestyle='--')
    ax[1].axvline(x=old_white_list[-1], color='r', linestyle='--')
    ax[1].plot(range(len(act_ee_velocity)), act_ee_velocity[:,1])
    ax[1].plot(new_white_list, [act_ee_velocity[i,1] for i in new_white_list], 'go')    

    ax[2].axvline(x=old_white_list[0], color='r', linestyle='--')
    ax[2].axvline(x=old_white_list[-1], color='r', linestyle='--')
    ax[2].plot(range(len(act_ee_velocity)), act_ee_velocity[:,2])
    ax[2].plot(new_white_list, [act_ee_velocity[i,2] for i in new_white_list], 'go')

    ax[3].axvline(x=old_white_list[0], color='r', linestyle='--')
    ax[3].axvline(x=old_white_list[-1], color='r', linestyle='--')
    ax[3].plot(range(len(act_ee_velocity)), act_ee_velocity[:,3])
    ax[3].plot(new_white_list, [act_ee_velocity[i,3] for i in new_white_list], 'go')

    ax[4].axvline(x=old_white_list[0], color='r', linestyle='--')
    ax[4].axvline(x=old_white_list[-1], color='r', linestyle='--')
    ax[4].plot(range(len(act_ee_velocity)), act_ee_velocity[:,4])
    ax[4].plot(new_white_list, [act_ee_velocity[i,4] for i in new_white_list], 'go')

    ax[5].axvline(x=old_white_list[0], color='r', linestyle='--')
    ax[5].axvline(x=old_white_list[-1], color='r', linestyle='--')
    ax[5].plot(range(len(act_ee_velocity)), act_ee_velocity[:,5])
    ax[5].plot(new_white_list, [act_ee_velocity[i,5] for i in new_white_list], 'go')
    save_path = os.path.join(vis_dir, f"{traj_dir}_WL_comparison_6dvelo.png")
    # plt.show()
    plt.savefig(save_path)
    plt.close()

def generate_white_list(full_traj_path):
    print(f"Generating white list for {full_traj_path}")
    start_index, end_index = get_traj_truncated_indices(full_traj_path)
    white_list = list(range(start_index, end_index))
    # save the white list to a json file
    with open(os.path.join(full_traj_path, "white_list.json"), "w") as f:
        json.dump(white_list, f)

def visualize_traj(traj_dir, vis_dir, traj_root):
    if os.path.exists(os.path.join(traj_dir, "recordings", "frames")):
        print(os.path.join(traj_dir, "trajectory.h5"))
        file = os.path.join(traj_dir, "trajectory.h5")
        data = h5py.File(file, "r")
        action_dict = data["action"]
        act_cartesian_velocity = action_dict["cartesian_velocity"][:]
        act_ee_velocity = act_cartesian_velocity[:,0:6] # Update: included rpy as well.
    
        max_velocity = np.max(np.abs(act_ee_velocity), axis=0)
        max_velocity_5_perc = max_velocity * 0.05
        # find the first and lastindex where velocity in some axis is greater than 0.005
        greater_than_005 = np.abs(act_ee_velocity) > max_velocity_5_perc
        start_index = np.where(greater_than_005)[0][0]
        end_index = np.where(np.abs(act_ee_velocity) > max_velocity_5_perc)[0][-1]
        print(f"{end_index - start_index} frames kept out of {len(act_ee_velocity)}")
        # plot the original xyz velocity
        fig, ax = plt.subplots(6, 1)
        ax[0].plot(act_ee_velocity[:,0])
        ax[1].plot(act_ee_velocity[:,1])
        ax[2].plot(act_ee_velocity[:,2])
        ax[3].plot(act_ee_velocity[:,3])
        ax[4].plot(act_ee_velocity[:,4])
        ax[5].plot(act_ee_velocity[:,5])
        # add vertical lines to the start and end index for each axis
        ax[0].axvline(x=start_index, color='r', linestyle='--')
        ax[0].axvline(x=end_index, color='r', linestyle='--')
        ax[1].axvline(x=start_index, color='r', linestyle='--')
        ax[1].axvline(x=end_index, color='r', linestyle='--')
        ax[2].axvline(x=start_index, color='r', linestyle='--')
        ax[2].axvline(x=end_index, color='r', linestyle='--')
        ax[3].axvline(x=start_index, color='r', linestyle='--')
        ax[3].axvline(x=end_index, color='r', linestyle='--')
        ax[4].axvline(x=start_index, color='r', linestyle='--')
        ax[4].axvline(x=end_index, color='r', linestyle='--')
        ax[5].axvline(x=start_index, color='r', linestyle='--')
        ax[5].axvline(x=end_index, color='r', linestyle='--')

        # Calculate the relative path from the trajectory root
        relative_traj_path = os.path.relpath(traj_dir, traj_root)

        # Construct the full save path
        save_dir = os.path.join(vis_dir, os.path.dirname(relative_traj_path))
        save_filename = os.path.basename(relative_traj_path) + "truncated.png"
        save_path = os.path.join(save_dir, save_filename)

        # Create the target directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        print(f"Saving plot to: {save_path}")
        # Save the plot to the constructed path
        plt.savefig(save_path)

        plt.close()
    else:
        raise ValueError(f"No suitable .h5 file and recordings/frames directory found for {traj_dir}")

if __name__ == "__main__":
    # start_index, end_index = get_traj_truncated_indices(TRAJ_ROOT, TRAJ_DIR)
    # print(start_index, end_index)
    # generate_white_list(TRAJ_ROOT, TRAJ_DIR)
    # visualize_traj(TRAJ_ROOT, TRAJ_DIR)
    DATA_DIRS = ["success", "failure"]
    # take in command line argument as TRAJ_ROOT, VIS_DIR
    TRAJ_ROOT = sys.argv[1]
    VIS_DIR = sys.argv[2]
    mean_diff_sum = 0
    count = 0
    for data_dir in DATA_DIRS:
        # traverse the subfolders in traj_dir/data_dir
        for traj_dir_date in os.listdir(os.path.join(TRAJ_ROOT, data_dir)):
            if os.path.isdir(os.path.join(TRAJ_ROOT, data_dir, traj_dir_date)):
                # traverse the subfolders in traj_dir/data_dir
                for sub_traj_dir in os.listdir(os.path.join(TRAJ_ROOT, data_dir, traj_dir_date)):
                    full_traj_path = os.path.join(TRAJ_ROOT, data_dir, traj_dir_date, sub_traj_dir)
                    if os.path.isdir(full_traj_path):
                        print(f"Processing {full_traj_path}")
                        # visualize_traj(full_traj_path, VIS_DIR, TRAJ_ROOT)
                        # generate_white_list(full_traj_path)
                        zero_velocity_indices, diff_sum = get_zero_velocity_skipped_white_list(full_traj_path, sub_traj_dir, VIS_DIR, TRAJ_ROOT)
                        mean_diff_sum += diff_sum
                        count += 1
    mean_diff_sum /= count
    print(f"Mean difference between old and new white list: {mean_diff_sum}")
