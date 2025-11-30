import os
import sys
import json 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

def load_whitelist(full_traj_path):
    """Load whitelist if it exists, otherwise return None"""
    white_list_path = os.path.join(full_traj_path, "white_list.json")
    if os.path.exists(white_list_path):
        return json.load(open(white_list_path))
    return None

def load_reward_features(full_traj_path):
    """Load reward features from existing JSON file"""
    reward_features_path = os.path.join(full_traj_path, "reward_features.json")
    if os.path.exists(reward_features_path):
        return json.load(open(reward_features_path))
    return None

def load_computed_reward(full_traj_path):
    """Load computed reward from existing NPY file"""
    computed_reward_path = os.path.join(full_traj_path, "computed_reward.npy")
    if os.path.exists(computed_reward_path):
        return np.load(computed_reward_path)
    return None

def check_mislabeling(reward_features, computed_reward, whitelist):
    """
    Check if an episode is mislabeled based on the criteria:
    - Episode has high reward > threshold in the front 30% steps of whitelist
    """
    if whitelist is None or computed_reward is None:
        return False, 0, 0
    
    whitelist_length = len(whitelist)
    front_steps = int(whitelist_length * 0.3)
    
    front_steps = min(front_steps, len(computed_reward))
    
    if front_steps == 0:
        return False, 0, 0
    
    front_rewards = computed_reward[:front_steps]
    
    avg_reward = np.mean(front_rewards)
    
    threshold = 8.0
    
    high_reward_steps = np.sum(front_rewards > threshold)
    mislabeled = high_reward_steps > 0
    
    return mislabeled, avg_reward, threshold

def analyze_mislabeling(TRAJ_ROOT):
    """Analyze mislabeling across all episodes"""
    DATA_DIRS = ["success", "failure"]
    
    # Statistics storage
    stats = {
        "success": {"total": 0, "mislabeled": 0, "details": []},
        "failure": {"total": 0, "mislabeled": 0, "details": []},
        "all": {"total": 0, "mislabeled": 0, "details": []}
    }
    
    all_rewards = []
    
    print("=== Starting Mislabeling Analysis ===")
    
    for data_dir in DATA_DIRS:
        print(f"\nProcessing {data_dir} episodes...")
        
        for traj_dir_date in os.listdir(os.path.join(TRAJ_ROOT, data_dir)):
            if os.path.isdir(os.path.join(TRAJ_ROOT, data_dir, traj_dir_date)):
                for sub_traj_dir in os.listdir(os.path.join(TRAJ_ROOT, data_dir, traj_dir_date)):
                    full_traj_path = os.path.join(TRAJ_ROOT, data_dir, traj_dir_date, sub_traj_dir)
                    if os.path.isdir(full_traj_path):
                        print(f"  Analyzing: {full_traj_path}")
                        
                        whitelist = load_whitelist(full_traj_path)
                        reward_features = load_reward_features(full_traj_path)
                        computed_reward = load_computed_reward(full_traj_path)
                        
                        if reward_features is None or computed_reward is None:
                            print(f"    Skipping - missing reward data")
                            continue
                        
                        mislabeled, avg_reward, threshold = check_mislabeling(
                            reward_features, computed_reward, whitelist
                        )
                        
                        episode_info = {
                            "path": full_traj_path,
                            "mislabeled": mislabeled,
                            "avg_reward": avg_reward,
                            "threshold": threshold,
                            "reward_length": len(computed_reward),
                            "whitelist_length": len(whitelist) if whitelist else 0
                        }
                        
                        stats[data_dir]["total"] += 1
                        stats["all"]["total"] += 1
                        
                        if mislabeled:
                            stats[data_dir]["mislabeled"] += 1
                            stats["all"]["mislabeled"] += 1
                        
                        stats[data_dir]["details"].append(episode_info)
                        stats["all"]["details"].append(episode_info)
                        
                        all_rewards.extend(computed_reward.tolist())
                        
                        print(f"    Mislabeled: {mislabeled}, Avg Reward: {avg_reward:.3f}, Threshold: {threshold:.3f}")
    
    return stats, all_rewards

def create_mislabeling_diagrams(stats, all_rewards, TRAJ_ROOT):
    """Create three diagrams showing mislabeling statistics"""
    
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Mislabeling Analysis - {os.path.basename(TRAJ_ROOT)}', fontsize=16, fontweight='bold')
    
    # 1. Mislabeling Ratio Bar Chart
    ax1 = axes[0, 0]
    categories = ['Success', 'Failure', 'All']
    mislabel_ratios = []
    
    for cat in ['success', 'failure', 'all']:
        if stats[cat]["total"] > 0:
            ratio = (stats[cat]["mislabeled"] / stats[cat]["total"]) * 100
        else:
            ratio = 0
        mislabel_ratios.append(ratio)
    
    bars = ax1.bar(categories, mislabel_ratios, color=['#2ecc71', '#e74c3c', '#3498db'], alpha=0.8)
    ax1.set_title('Mislabeling Ratio by Category', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Mislabeling Percentage (%)', fontsize=12)
    ax1.set_ylim(0, max(mislabel_ratios) * 1.2 if max(mislabel_ratios) > 0 else 10)
    
    # Add value labels on bars
    for bar, ratio in zip(bars, mislabel_ratios):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{ratio:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 2. Episode Count Comparison
    ax2 = axes[0, 1]
    categories = ['Success', 'Failure']
    total_episodes = [stats['success']['total'], stats['failure']['total']]
    mislabeled_episodes = [stats['success']['mislabeled'], stats['failure']['mislabeled']]
    correctly_labeled = [total - mislabeled for total, mislabeled in zip(total_episodes, mislabeled_episodes)]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, correctly_labeled, width, label='Correctly Labeled', color='#27ae60', alpha=0.8)
    bars2 = ax2.bar(x + width/2, mislabeled_episodes, width, label='Mislabeled', color='#e74c3c', alpha=0.8)
    
    ax2.set_title('Episode Count by Labeling Status', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Number of Episodes', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.legend()
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Reward Distribution Analysis
    ax3 = axes[1, 0]
    
    # Separate rewards by category
    success_rewards = []
    failure_rewards = []
    
    for episode_info in stats['success']['details']:
        if episode_info['avg_reward'] > 0:
            success_rewards.append(episode_info['avg_reward'])
    
    for episode_info in stats['failure']['details']:
        if episode_info['avg_reward'] > 0:
            failure_rewards.append(episode_info['avg_reward'])
    
    # Create histogram
    if success_rewards and failure_rewards:
        ax3.hist(success_rewards, bins=20, alpha=0.7, label='Success Episodes', color='#2ecc71', density=True)
        ax3.hist(failure_rewards, bins=20, alpha=0.7, label='Failure Episodes', color='#e74c3c', density=True)
        ax3.set_title('Reward Distribution by Episode Type', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Average Reward', fontsize=12)
        ax3.set_ylabel('Density', fontsize=12)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. Mislabeling Details Table
    ax4 = axes[1, 1]
    ax4.axis('tight')
    ax4.axis('off')
    
    # Create summary table
    table_data = []
    for cat in ['success', 'failure', 'all']:
        total = stats[cat]["total"]
        mislabeled = stats[cat]["mislabeled"]
        ratio = (mislabeled / total * 100) if total > 0 else 0
        table_data.append([cat.title(), total, mislabeled, f"{ratio:.1f}%"])
    
    table = ax4.table(cellText=table_data,
                     colLabels=['Category', 'Total Episodes', 'Mislabeled', 'Mislabel %'],
                     cellLoc='center',
                     loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    for i in range(len(table_data) + 1):
        for j in range(4):
            cell = table[(i, j)]
            if i == 0:  # Header row
                cell.set_facecolor('#34495e')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#ecf0f1' if i % 2 == 0 else '#bdc3c7')
    
    ax4.set_title('Summary Statistics', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    output_path = os.path.join(TRAJ_ROOT, 'mislabeling_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/mislabeling_{os.path.basename(TRAJ_ROOT)}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nMislabeling analysis diagrams saved to: {output_path}")
    print(f"Local copy saved to: results/mislabeling_{os.path.basename(TRAJ_ROOT)}.png")
    
    return output_path

def print_detailed_statistics(stats):
    """Print detailed statistics to console"""
    print("\n" + "="*60)
    print("DETAILED MISLABELING STATISTICS")
    print("="*60)
    
    for category in ['success', 'failure', 'all']:
        print(f"\n{category.upper()} EPISODES:")
        print(f"  Total episodes: {stats[category]['total']}")
        print(f"  Mislabeled episodes: {stats[category]['mislabeled']}")
        if stats[category]['total'] > 0:
            ratio = (stats[category]['mislabeled'] / stats[category]['total']) * 100
            print(f"  Mislabeling percentage: {ratio:.2f}%")
        else:
            print(f"  Mislabeling percentage: 0.00%")
    
    print(f"\nEXAMPLES OF MISLABELED EPISODES:")
    for category in ['success', 'failure']:
        mislabeled_episodes = [ep for ep in stats[category]['details'] if ep['mislabeled']]
        if mislabeled_episodes:
            print(f"\n{category.upper()} mislabeled episodes:")
            for i, episode in enumerate(mislabeled_episodes[:5]):  # Show first 5
                print(f"  {i+1}. {os.path.basename(episode['path'])}")
                print(f"     Avg Reward: {episode['avg_reward']:.3f}, Threshold: {episode['threshold']:.3f}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python calculate_mislabel.py <TRAJ_ROOT>")
        sys.exit(1)
    
    TRAJ_ROOT = sys.argv[1]
    
    if not os.path.exists(TRAJ_ROOT):
        print(f"Error: TRAJ_ROOT {TRAJ_ROOT} does not exist")
        sys.exit(1)
    
    stats, all_rewards = analyze_mislabeling(TRAJ_ROOT)
    
    output_path = create_mislabeling_diagrams(stats, all_rewards, TRAJ_ROOT)
    
    print_detailed_statistics(stats)
    
    print(f"\nAnalysis complete! Results saved to: {output_path}")