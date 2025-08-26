#!/usr/bin/env python3
"""
Enhanced plot showing average NRMSE vs number of Gaussian-augmented samples,
comparing VLA-CLIP verifier selection with random selection.

X-axis: Number of Gaussian samples (1 to 128)
Y-axis: Average NRMSE across 100 datapoints
- Horizontal line at baseline for Gaussian sample average NRMSE
- VLA-CLIP curve: select top N samples by VLA-CLIP score
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
    with open('bridge_vla_clip_scores_gaussian_20250826_075646.json', 'r') as f:
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

def generate_enhanced_gaussian_plot():
    """Generate the enhanced threshold analysis plot for Gaussian samples"""
    
    # Load data
    gaussian_samples, gaussian_avg_nrmse = load_gaussian_vla_clip_data()
    
    # Calculate NRMSE for specific instruction thresholds
    num_instructions_range = [1, 2, 4, 8, 16, 32, 64, 128]
    vla_clip_nrmse_values = []
    random_nrmse_values = []
    
    print("Calculating average NRMSE for different sample thresholds...")
    
    for num_instructions in num_instructions_range:
        # VLA-CLIP method
        vla_clip_nrmse = calculate_threshold_nrmse_vla_clip_gaussian(gaussian_samples, num_instructions)
        vla_clip_nrmse_values.append(vla_clip_nrmse)
        
        # Random selection method
        random_nrmse = calculate_threshold_nrmse_random_gaussian(gaussian_samples, num_instructions)
        random_nrmse_values.append(random_nrmse)
        
        print(f"  {num_instructions} samples - VLA-CLIP: {vla_clip_nrmse:.4f}, Random: {random_nrmse:.4f}")
    
    # Create the plot with publication-quality styling
    fig, ax = plt.subplots()
    
    # Colors and markers following the example
    color_clip = '#1f77b4'
    color_random = '#ff7f0e'  # Orange
    marker_clip = 'o'
    marker_random = '^'  # Triangle
    
    # Add baseline (average of all Gaussian samples)
    ax.axhline(y=0.1665, color='black', linestyle='-', linewidth=2.0, 
               label=f'Gaussian Baseline ({gaussian_avg_nrmse:.4f})')
    
    # Plot VLA-CLIP curve
    ax.plot(num_instructions_range, vla_clip_nrmse_values, label='VLA-CLIP Verifier', 
            color=color_clip, marker=marker_clip, markersize=10, linestyle='-', linewidth=2.5)
    
    # Plot Random Selection curve
    ax.plot(num_instructions_range, random_nrmse_values, label='Random Selection', 
            color=color_random, marker=marker_random, markersize=10, linestyle=':', linewidth=2.5)
    
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
    
    # Calculate statistics for reporting
    best_clip_nrmse = min([v for v in vla_clip_nrmse_values if v is not None])
    best_clip_idx = [v for v in vla_clip_nrmse_values if v is not None].index(best_clip_nrmse)
    best_clip_num = num_instructions_range[best_clip_idx]
    
    best_random_nrmse = min([v for v in random_nrmse_values if v is not None])
    best_random_idx = [v for v in random_nrmse_values if v is not None].index(best_random_nrmse)
    best_random_num = num_instructions_range[best_random_idx]
    
    # Print detailed comparison
    print(f"\n=== Detailed Comparison ===")
    print(f"Gaussian samples average NRMSE: {gaussian_avg_nrmse:.4f}")
    print(f"\nVLA-CLIP Method:")
    print(f"  Best NRMSE: {best_clip_nrmse:.4f} (with {best_clip_num} samples)")
    print(f"  Improvement: {((gaussian_avg_nrmse - best_clip_nrmse) / gaussian_avg_nrmse * 100):.1f}%")
    print(f"  NRMSE at 128 samples: {vla_clip_nrmse_values[-1] if vla_clip_nrmse_values[-1] is not None else 'N/A'}")
    
    print(f"\nRandom Selection Method:")
    print(f"  Best NRMSE: {best_random_nrmse:.4f} (with {best_random_num} samples)")
    print(f"  Improvement: {((gaussian_avg_nrmse - best_random_nrmse) / gaussian_avg_nrmse * 100):.1f}%")
    print(f"  NRMSE at 128 samples: {random_nrmse_values[-1] if random_nrmse_values[-1] is not None else 'N/A'}")
    
    # Determine which method is best
    winner_name = "VLA-CLIP" if best_clip_nrmse < best_random_nrmse else "Random Selection"
    winner_score = min(best_clip_nrmse, best_random_nrmse)
    
    print(f"\nBest performing method: {winner_name} with NRMSE {winner_score:.4f}")
    
    # Compare VLA-CLIP against random
    clip_vs_random = ((best_random_nrmse - best_clip_nrmse) / best_random_nrmse * 100)
    print(f"VLA-CLIP vs Random: {clip_vs_random:.1f}% better")
    
    # Save numerical results
    results_df = pd.DataFrame({
        'num_samples': num_instructions_range,
        'vla_clip_nrmse': vla_clip_nrmse_values,
        'random_selection_nrmse': random_nrmse_values
    })
    results_df.to_csv('gaussian_threshold_results.csv', index=False)
    print(f"Numerical results saved as: gaussian_threshold_results.csv")
    
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
    
    # Check if required data file exists
    required_file = 'bridge_vla_clip_scores_gaussian_20250826_075646.json'
    
    if not os.path.exists(required_file):
        print(f"Error: Required file not found: {required_file}")
        print("Please ensure the Gaussian VLA-CLIP scores file exists in the current directory.")
        exit(1)
    
    # Generate the enhanced plot
    results = generate_enhanced_gaussian_plot()
