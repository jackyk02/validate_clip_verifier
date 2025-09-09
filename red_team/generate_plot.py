#!/usr/bin/env python3
"""
Red team plot showing average NRMSE vs number of rephrased instructions,
comparing VLA-CLIP and random selection methods.

X-axis: Number of rephrased instructions (1 to 128)
Y-axis: Average NRMSE across 100 datapoints
- Horizontal line at baseline for red team instruction average NRMSE
- VLA-CLIP curve: select top N instructions by VLA-CLIP score
- Random curve: select random N instructions
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

def load_red_team_vla_clip_data():
    """Load red team VLA-CLIP scores and organize by sample"""
    print("Loading red team VLA-CLIP scores...")
    with open('red_team_vla_clip_scores_20250909_053258.json', 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data['results'])} red team VLA-CLIP results")
    
    # Organize data by sample_id
    samples = defaultdict(list)
    red_team_nrmse = []
    
    for result in data['results']:
        sample_id = result['sample_id']
        samples[sample_id].append(result)
        
        # Collect red team instruction NRMSE (is_red_team_original = True)
        if result['is_red_team_original']:
            red_team_nrmse.append(result['openvla_nrmse'])
    
    print(f"Found {len(samples)} unique samples for red team VLA-CLIP")
    print(f"Red team instruction average NRMSE: {np.mean(red_team_nrmse):.4f}")
    
    return samples, np.mean(red_team_nrmse)


def calculate_threshold_nrmse_vla_clip(samples, num_instructions):
    """
    For each sample, select the top num_instructions by VLA-CLIP score
    and return the average NRMSE across all samples.
    """
    sample_nrmse_values = []
    
    for sample_id, results in samples.items():
        # Separate red team original and rephrased instructions
        red_team_original = [r for r in results if r['is_red_team_original']]
        rephrased = [r for r in results if not r['is_red_team_original']]
        
        if not red_team_original:
            continue
            
        if num_instructions == 1:
            # Only use red team original instruction
            selected = red_team_original
        else:
            # Select top (num_instructions - 1) rephrased instructions by VLA-CLIP score
            # Plus the red team original instruction
            rephrased_sorted = sorted(rephrased, key=lambda x: x['vla_clip_similarity_score'], reverse=True)
            top_rephrased = rephrased_sorted[:num_instructions-1]
            selected = red_team_original + top_rephrased
        
        # Calculate average NRMSE for this sample's selected instructions
        if selected:
            nrmse_values = [r['openvla_nrmse'] for r in selected]
            sample_avg_nrmse = np.mean(nrmse_values)
            sample_nrmse_values.append(sample_avg_nrmse)
    
    # Return average NRMSE across all samples
    return np.mean(sample_nrmse_values) if sample_nrmse_values else None


def calculate_threshold_nrmse_random(samples, num_instructions, random_seed=42):
    """
    For each sample, randomly select num_instructions (including red team original)
    and return the average NRMSE across all samples.
    """
    np.random.seed(random_seed)  # For reproducible results
    sample_nrmse_values = []
    
    for sample_id, results in samples.items():
        # Separate red team original and rephrased instructions
        red_team_original = [r for r in results if r['is_red_team_original']]
        rephrased = [r for r in results if not r['is_red_team_original']]
        
        if not red_team_original:
            continue
            
        if num_instructions == 1:
            # Only use red team original instruction
            selected = red_team_original
        else:
            # Randomly select (num_instructions - 1) rephrased instructions
            # Plus the red team original instruction
            if len(rephrased) >= num_instructions - 1:
                random_rephrased = np.random.choice(rephrased, size=num_instructions-1, replace=False).tolist()
            else:
                random_rephrased = rephrased  # Use all available if not enough
            selected = red_team_original + random_rephrased
        
        # Calculate average NRMSE for this sample's selected instructions
        if selected:
            nrmse_values = [r['openvla_nrmse'] for r in selected]
            sample_avg_nrmse = np.mean(nrmse_values)
            sample_nrmse_values.append(sample_avg_nrmse)
    
    # Return average NRMSE across all samples
    return np.mean(sample_nrmse_values) if sample_nrmse_values else None

def generate_red_team_plot():
    """Generate the red team threshold analysis plot comparing VLA-CLIP and random selection"""
    
    # Load data
    red_team_samples, red_team_avg_nrmse = load_red_team_vla_clip_data()
    
    # Calculate NRMSE for specific instruction thresholds
    num_instructions_range = [1, 2, 4, 8, 16, 32, 64, 128]
    vla_clip_nrmse_values = []
    random_nrmse_values = []
    
    print("Calculating average NRMSE for different instruction thresholds...")
    
    for num_instructions in num_instructions_range:
        # VLA-CLIP method
        vla_clip_nrmse = calculate_threshold_nrmse_vla_clip(red_team_samples, num_instructions)
        vla_clip_nrmse_values.append(vla_clip_nrmse)
        
        # Random selection method
        random_nrmse = calculate_threshold_nrmse_random(red_team_samples, num_instructions)
        random_nrmse_values.append(random_nrmse)
        
        print(f"  {num_instructions} instructions - VLA-CLIP: {vla_clip_nrmse:.4f}, Random: {random_nrmse:.4f}")
    
    # Create the plot with publication-quality styling
    fig, ax = plt.subplots()
    
    # Colors and markers
    color_clip = '#1f77b4'  # Blue
    color_random = '#ff7f0e'  # Orange
    marker_clip = 'o'
    marker_random = '^'  # Triangle
    
    # Add red team instruction baseline
    ax.axhline(y=red_team_avg_nrmse, color='black', linestyle='-', linewidth=2.0, 
               label='Red Team Instruction Baseline')
    
    # Plot VLA-CLIP curve
    ax.plot(num_instructions_range, vla_clip_nrmse_values, label='VLA-CLIP Verifier', 
            color=color_clip, marker=marker_clip, markersize=10, linestyle='-', linewidth=2.5)
    
    # Plot Random Selection curve
    ax.plot(num_instructions_range, random_nrmse_values, label='Random Selection', 
            color=color_random, marker=marker_random, markersize=10, linestyle=':', linewidth=2.5)
    
    # Axis labels and scale
    ax.set_xlabel("Number of Rephrases")
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
    legend = ax.legend(loc='lower right', framealpha=0.95, edgecolor='black', fontsize=11)
    legend.get_frame().set_linewidth(1.5)
    
    # Save plot
    output_file = 'red_team_threshold_analysis.png'
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as: {output_file}")
    
    # Also save as PDF
    plt.savefig('red_team_threshold_analysis.pdf', bbox_inches='tight')
    print(f"Plot also saved as: red_team_threshold_analysis.pdf")
    
    plt.show()
    
    # Calculate statistics for reporting
    best_clip_nrmse = min([v for v in vla_clip_nrmse_values if v is not None])
    best_clip_idx = [v for v in vla_clip_nrmse_values if v is not None].index(best_clip_nrmse)
    best_clip_num = num_instructions_range[best_clip_idx]
    
    best_random_nrmse = min([v for v in random_nrmse_values if v is not None])
    best_random_idx = [v for v in random_nrmse_values if v is not None].index(best_random_nrmse)
    best_random_num = num_instructions_range[best_random_idx]
    
    # Print detailed comparison
    print(f"\n=== Red Team Analysis Results ===")
    print(f"Red team instruction baseline NRMSE: {red_team_avg_nrmse:.4f}")
    print(f"\nVLA-CLIP Method:")
    print(f"  Best NRMSE: {best_clip_nrmse:.4f} (with {best_clip_num} instructions)")
    print(f"  Improvement vs baseline: {((red_team_avg_nrmse - best_clip_nrmse) / red_team_avg_nrmse * 100):.1f}%")
    print(f"  NRMSE at 128 instructions: {vla_clip_nrmse_values[-1] if vla_clip_nrmse_values[-1] is not None else 'N/A'}")
    
    print(f"\nRandom Selection Method:")
    print(f"  Best NRMSE: {best_random_nrmse:.4f} (with {best_random_num} instructions)")
    print(f"  Improvement vs baseline: {((red_team_avg_nrmse - best_random_nrmse) / red_team_avg_nrmse * 100):.1f}%")
    print(f"  NRMSE at 128 instructions: {random_nrmse_values[-1] if random_nrmse_values[-1] is not None else 'N/A'}")
    
    # Determine which method is best
    all_best_scores = [
        ("VLA-CLIP", best_clip_nrmse),
        ("Random Selection", best_random_nrmse)
    ]
    winner_name, winner_score = min(all_best_scores, key=lambda x: x[1])
    
    print(f"\nBest performing method: {winner_name} with NRMSE {winner_score:.4f}")
    
    # Compare VLA-CLIP against random
    clip_vs_random = ((best_random_nrmse - best_clip_nrmse) / best_random_nrmse * 100)
    print(f"VLA-CLIP vs Random: {clip_vs_random:.1f}% better")
    
    # Save numerical results
    results_df = pd.DataFrame({
        'num_instructions': num_instructions_range,
        'vla_clip_nrmse': vla_clip_nrmse_values,
        'random_selection_nrmse': random_nrmse_values
    })
    results_df.to_csv('red_team_threshold_results.csv', index=False)
    print(f"Numerical results saved as: red_team_threshold_results.csv")
    
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
        'red_team_vla_clip_scores_20250909_053258.json'
    ]
    
    missing_files = []
    for f in required_files:
        if not os.path.exists(f):
            missing_files.append(f)
    
    if missing_files:
        print("Error: Missing required files:")
        for f in missing_files:
            print(f"  - {f}")
        exit(1)
    
    # Generate the red team plot
    results = generate_red_team_plot()