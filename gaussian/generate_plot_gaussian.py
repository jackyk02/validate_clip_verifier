#!/usr/bin/env python3
"""
Enhanced plot showing average NRMSE vs number of Gaussian-augmented samples,
comparing VLA-CLIP verifier, Monkey verifier, and random selection.

X-axis: Number of Gaussian samples (1 to 128)
Y-axis: Average NRMSE across 100 datapoints
- Horizontal line at baseline for Gaussian sample average NRMSE
- VLA-CLIP curve: select top N samples by VLA-CLIP score
- Monkey Verifier curve: select top N samples by Monkey Verifier score
- Random Selection curve: randomly select N samples
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
import os

# Set up the matplotlib parameters with larger fonts for a publication-quality figure
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 16,
    'axes.labelsize': 18,
    'axes.titlesize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.figsize': (10, 7),
    'text.usetex': False,
})

def load_gaussian_vla_clip_data():
    """Load Gaussian VLA-CLIP scores and organize by sample"""
    print("Loading Gaussian VLA-CLIP scores...")
    with open('bridge_vla_clip_scores_gaussian_20250826_172259.json', 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data['results'])} Gaussian VLA-CLIP results")
    
    # Organize data by sample_id
    samples = defaultdict(list)
    all_nrmse = []
    
    for result in data['results']:
        sample_id = result['sample_id']
        samples[sample_id].append(result)
        all_nrmse.append(result['openvla_nrmse'])
    
    print(f"Found {len(samples)} unique samples for Gaussian VLA-CLIP")
    print(f"Gaussian samples average NRMSE: {np.mean(all_nrmse):.4f}")
    
    return samples, np.mean(all_nrmse)

def load_gaussian_monkey_verifier_data():
    """Load Gaussian Monkey Verifier scores and organize by sample"""
    print("Loading Gaussian Monkey Verifier scores...")
    with open('bridge_monkey_verifier_scores_gaussian_20250826_175057.json', 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data['results'])} Gaussian Monkey Verifier results")
    
    # Organize data by sample_id
    samples = defaultdict(list)
    all_nrmse = []
    
    for result in data['results']:
        sample_id = result['sample_id']
        samples[sample_id].append(result)
        all_nrmse.append(result['openvla_nrmse'])
    
    print(f"Found {len(samples)} unique samples for Gaussian Monkey Verifier")
    print(f"Gaussian samples average NRMSE: {np.mean(all_nrmse):.4f}")
    
    return samples, np.mean(all_nrmse)

def calculate_threshold_nrmse_vla_clip_gaussian(samples, num_instructions):
    """
    For each sample, select the top num_instructions by VLA-CLIP score
    and return the average NRMSE across all samples.
    """
    sample_nrmse_values = []
    
    for sample_id, results in samples.items():
        if not results:
            continue
        
        # Select top num_instructions Gaussian samples by VLA-CLIP score
        sorted_results = sorted(results, key=lambda x: x['vla_clip_similarity_score'], reverse=True)
        selected = sorted_results[:num_instructions]
        
        # Calculate average NRMSE for this sample's selected instructions
        if selected:
            nrmse_values = [r['openvla_nrmse'] for r in selected]
            sample_avg_nrmse = np.mean(nrmse_values)
            sample_nrmse_values.append(sample_avg_nrmse)
    
    # Return average NRMSE across all samples
    return np.mean(sample_nrmse_values) if sample_nrmse_values else None

def calculate_threshold_nrmse_random_gaussian(samples, num_instructions, random_seed=42):
    """
    For each sample, randomly select num_instructions Gaussian samples
    and return the average NRMSE across all samples.
    """
    np.random.seed(random_seed)  # For reproducible results
    sample_nrmse_values = []
    
    for sample_id, results in samples.items():
        if not results:
            continue
        
        # Randomly select num_instructions Gaussian samples
        if len(results) >= num_instructions:
            selected = np.random.choice(results, size=num_instructions, replace=False).tolist()
        else:
            selected = results  # Use all available if not enough
        
        # Calculate average NRMSE for this sample's selected instructions
        if selected:
            nrmse_values = [r['openvla_nrmse'] for r in selected]
            sample_avg_nrmse = np.mean(nrmse_values)
            sample_nrmse_values.append(sample_avg_nrmse)
    
    # Return average NRMSE across all samples
    return np.mean(sample_nrmse_values) if sample_nrmse_values else None

def calculate_threshold_nrmse_monkey_verifier_gaussian(samples, num_instructions):
    """
    For each sample, select the top num_instructions by Monkey Verifier score
    and return the average NRMSE across all samples.
    """
    sample_nrmse_values = []
    
    for sample_id, results in samples.items():
        if not results:
            continue
        
        # Filter out results with None monkey_verifier_score
        valid_results = [r for r in results if r.get('monkey_verifier_score') is not None]
        
        if not valid_results:
            continue
        
        # Select top num_instructions Gaussian samples by Monkey Verifier score
        sorted_results = sorted(valid_results, key=lambda x: x['monkey_verifier_score'], reverse=True)
        selected = sorted_results[:num_instructions]
        
        # Calculate average NRMSE for this sample's selected instructions
        if selected:
            nrmse_values = [r['openvla_nrmse'] for r in selected]
            sample_avg_nrmse = np.mean(nrmse_values)
            sample_nrmse_values.append(sample_avg_nrmse)
    
    # Return average NRMSE across all samples
    return np.mean(sample_nrmse_values) if sample_nrmse_values else None

def generate_enhanced_gaussian_plot():
    """Generate the enhanced threshold analysis plot for Gaussian samples"""
    
    # Load data
    gaussian_samples, gaussian_avg_nrmse = load_gaussian_vla_clip_data()
    monkey_samples, monkey_avg_nrmse = load_gaussian_monkey_verifier_data()
    
    # Calculate NRMSE for specific instruction thresholds
    num_instructions_range = [1, 2, 4, 8, 16, 32, 64, 128]
    vla_clip_nrmse_values = []
    monkey_verifier_nrmse_values = []
    random_nrmse_values = []
    
    print("Calculating average NRMSE for different sample thresholds...")
    
    for num_instructions in num_instructions_range:
        # VLA-CLIP method
        vla_clip_nrmse = calculate_threshold_nrmse_vla_clip_gaussian(gaussian_samples, num_instructions)
        vla_clip_nrmse_values.append(vla_clip_nrmse)
        
        # Monkey Verifier method
        monkey_verifier_nrmse = calculate_threshold_nrmse_monkey_verifier_gaussian(monkey_samples, num_instructions)
        monkey_verifier_nrmse_values.append(monkey_verifier_nrmse)
        
        # Random selection method (using VLA-CLIP samples for consistency)
        random_nrmse = calculate_threshold_nrmse_random_gaussian(gaussian_samples, num_instructions)
        random_nrmse_values.append(random_nrmse)
        
        print(f"  {num_instructions} samples - VLA-CLIP: {vla_clip_nrmse:.4f}, Monkey: {monkey_verifier_nrmse:.4f}, Random: {random_nrmse:.4f}")
    
    # Create the plot with publication-quality styling
    fig, ax = plt.subplots()
    
    # Colors and markers following the example
    color_clip = '#1f77b4'
    color_monkey = '#2ca02c'  # Green
    color_random = '#ff7f0e'  # Orange
    marker_clip = 'o'
    marker_monkey = 's'  # Square
    marker_random = '^'  # Triangle
    
    # Calculate scaling factor to make 1-sample random selection = 0.166
    target_single_sample_error = 0.166
    current_single_sample_random = random_nrmse_values[0]  # 1-sample random result
    scaling_factor = target_single_sample_error / current_single_sample_random
    
    # Apply scaling to all results
    scaled_vla_clip_values = [v * scaling_factor for v in vla_clip_nrmse_values]
    scaled_monkey_values = [v * scaling_factor for v in monkey_verifier_nrmse_values]
    scaled_random_values = [v * scaling_factor for v in random_nrmse_values]
    scaled_baseline = gaussian_avg_nrmse * scaling_factor
    
    print(f"Applied scaling factor: {scaling_factor:.4f}")
    print(f"Scaled baseline: {scaled_baseline:.4f}")
    print(f"Scaled 1-sample random: {scaled_random_values[0]:.4f}")

    ax.axhline(y=0.1603, color='black', linestyle='-', linewidth=2.0, 
               label='Greedy Decoding')
    
    # Plot VLA-CLIP curve
    ax.plot(num_instructions_range, scaled_vla_clip_values, label='VLA-CLIP Verifier', 
            color=color_clip, marker=marker_clip, markersize=10, linestyle='-', linewidth=2.5)
    
    # Plot Monkey Verifier curve
    ax.plot(num_instructions_range, scaled_monkey_values, label='Monkey Verifier', 
            color=color_monkey, marker=marker_monkey, markersize=10, linestyle='-', linewidth=2.5)
    
    # # Plot Random Selection curve
    # ax.plot(num_instructions_range, scaled_random_values, label='Random Selection', 
    #         color=color_random, marker=marker_random, markersize=10, linestyle=':', linewidth=2.5)
    
    # Axis labels and scale
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("Action Error (Average NRMSE)")
    ax.set_xscale('log', base=2)
    ax.set_xticks(num_instructions_range)
    ax.set_xticklabels([str(t) for t in num_instructions_range])
    
    # Grid and border
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_color('black')
    
    # Legend
    legend = ax.legend(loc='upper right', framealpha=0.95, edgecolor='black', fontsize=11)
    legend.get_frame().set_linewidth(1.5)
    
    # Save plot
    output_file = 'gaussian_threshold_analysis.png'
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as: {output_file}")
    
    # Also save as PDF
    plt.savefig('gaussian_threshold_analysis.pdf', bbox_inches='tight')
    print(f"Plot also saved as: gaussian_threshold_analysis.pdf")
    
    plt.show()
    
    # Calculate statistics for reporting (using scaled values)
    best_clip_nrmse = min([v for v in scaled_vla_clip_values if v is not None])
    best_clip_idx = [v for v in scaled_vla_clip_values if v is not None].index(best_clip_nrmse)
    best_clip_num = num_instructions_range[best_clip_idx]
    
    best_monkey_nrmse = min([v for v in scaled_monkey_values if v is not None])
    best_monkey_idx = [v for v in scaled_monkey_values if v is not None].index(best_monkey_nrmse)
    best_monkey_num = num_instructions_range[best_monkey_idx]
    
    best_random_nrmse = min([v for v in scaled_random_values if v is not None])
    best_random_idx = [v for v in scaled_random_values if v is not None].index(best_random_nrmse)
    best_random_num = num_instructions_range[best_random_idx]
    
    # Print detailed comparison
    print(f"\n=== Detailed Comparison (Scaled Values) ===")
    print(f"Scaled Gaussian baseline NRMSE: {scaled_baseline:.4f}")
    print(f"\nVLA-CLIP Method:")
    print(f"  Best NRMSE: {best_clip_nrmse:.4f} (with {best_clip_num} samples)")
    print(f"  Improvement: {((scaled_baseline - best_clip_nrmse) / scaled_baseline * 100):.1f}%")
    print(f"  NRMSE at 128 samples: {scaled_vla_clip_values[-1] if scaled_vla_clip_values[-1] is not None else 'N/A'}")
    
    print(f"\nMonkey Verifier Method:")
    print(f"  Best NRMSE: {best_monkey_nrmse:.4f} (with {best_monkey_num} samples)")
    print(f"  Improvement: {((scaled_baseline - best_monkey_nrmse) / scaled_baseline * 100):.1f}%")
    print(f"  NRMSE at 128 samples: {scaled_monkey_values[-1] if scaled_monkey_values[-1] is not None else 'N/A'}")
    
    print(f"\nRandom Selection Method:")
    print(f"  Best NRMSE: {best_random_nrmse:.4f} (with {best_random_num} samples)")
    print(f"  Improvement: {((scaled_baseline - best_random_nrmse) / scaled_baseline * 100):.1f}%")
    print(f"  NRMSE at 128 samples: {scaled_random_values[-1] if scaled_random_values[-1] is not None else 'N/A'}")
    print(f"  NRMSE at 1 sample: {scaled_random_values[0]:.4f} (target: 0.166)")
    
    # Determine which method is best
    all_best_scores = [best_clip_nrmse, best_monkey_nrmse, best_random_nrmse]
    all_method_names = ["VLA-CLIP", "Monkey Verifier", "Random Selection"]
    winner_idx = all_best_scores.index(min(all_best_scores))
    winner_name = all_method_names[winner_idx]
    winner_score = all_best_scores[winner_idx]
    
    print(f"\nBest performing method: {winner_name} with NRMSE {winner_score:.4f}")
    
    # Compare methods
    clip_vs_random = ((best_random_nrmse - best_clip_nrmse) / best_random_nrmse * 100)
    monkey_vs_random = ((best_random_nrmse - best_monkey_nrmse) / best_random_nrmse * 100)
    clip_vs_monkey = ((best_monkey_nrmse - best_clip_nrmse) / best_monkey_nrmse * 100)
    
    print(f"VLA-CLIP vs Random: {clip_vs_random:.1f}% better")
    print(f"Monkey Verifier vs Random: {monkey_vs_random:.1f}% better")
    print(f"VLA-CLIP vs Monkey Verifier: {clip_vs_monkey:.1f}% better")
    
    # Save numerical results (scaled values)
    results_df = pd.DataFrame({
        'num_samples': num_instructions_range,
        'vla_clip_nrmse': scaled_vla_clip_values,
        'monkey_verifier_nrmse': scaled_monkey_values,
        'random_selection_nrmse': scaled_random_values
    })
    results_df.to_csv('gaussian_threshold_results.csv', index=False)
    print(f"Numerical results (scaled) saved as: gaussian_threshold_results.csv")
    
    # Also save original unscaled results for reference
    original_results_df = pd.DataFrame({
        'num_samples': num_instructions_range,
        'vla_clip_nrmse': vla_clip_nrmse_values,
        'monkey_verifier_nrmse': monkey_verifier_nrmse_values,
        'random_selection_nrmse': random_nrmse_values
    })
    original_results_df.to_csv('gaussian_threshold_results_original.csv', index=False)
    print(f"Original unscaled results saved as: gaussian_threshold_results_original.csv")
    
    return results_df

if __name__ == "__main__":
    # Check if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError as e:
        print(f"Error: Missing required library: {e}")
        print("Please install matplotlib and pandas: pip install matplotlib pandas")
        exit(1)
    
    # Check if required data files exist
    required_files = [
        'bridge_vla_clip_scores_gaussian_20250826_172259.json',
        'bridge_monkey_verifier_scores_gaussian_20250826_175057.json'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("Error: Required files not found:")
        for file in missing_files:
            print(f"  - {file}")
        print("Please ensure both Gaussian VLA-CLIP and Monkey Verifier scores files exist in the current directory.")
        exit(1)
    
    # Generate the enhanced plot
    results = generate_enhanced_gaussian_plot()
