#!/usr/bin/env python3
"""
Generate plot showing average NRMSE vs number of rephrased instructions,
using VLA-CLIP similarity score to select the best instructions.

X-axis: Number of rephrased instructions (1 to 128)
Y-axis: Average NRMSE across 100 datapoints
- Horizontal line at 0.1665 for original instruction average NRMSE
- For each threshold, select top N instructions by VLA-CLIP score and compute average NRMSE
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd

def load_vla_clip_data():
    """Load VLA-CLIP scores and organize by sample"""
    print("Loading VLA-CLIP scores...")
    with open('/root/bridge_dataset_extracted/bridge_vla_clip_scores_20250809_194343.json', 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data['results'])} results")
    
    # Organize data by sample_id
    samples = defaultdict(list)
    original_nrmse = []
    
    for result in data['results']:
        sample_id = result['sample_id']
        samples[sample_id].append(result)
        
        # Collect original instruction NRMSE
        if result['is_original']:
            original_nrmse.append(result['openvla_nrmse'])
    
    print(f"Found {len(samples)} unique samples")
    print(f"Original instruction average NRMSE: {np.mean(original_nrmse):.4f}")
    
    return samples, np.mean(original_nrmse)

def calculate_threshold_nrmse(samples, num_instructions):
    """
    For each sample, select the top num_instructions by VLA-CLIP score
    and return the average NRMSE across all samples.
    """
    sample_nrmse_values = []
    
    for sample_id, results in samples.items():
        # Separate original and rephrased instructions
        original = [r for r in results if r['is_original']]
        rephrased = [r for r in results if not r['is_original']]
        
        if not original:
            continue
            
        if num_instructions == 1:
            # Only use original instruction
            selected = original
        else:
            # Select top (num_instructions - 1) rephrased instructions by VLA-CLIP score
            # Plus the original instruction
            rephrased_sorted = sorted(rephrased, key=lambda x: x['vla_clip_similarity_score'], reverse=True)
            top_rephrased = rephrased_sorted[:num_instructions-1]
            selected = original + top_rephrased
        
        # Calculate average NRMSE for this sample's selected instructions
        if selected:
            nrmse_values = [r['openvla_nrmse'] for r in selected]
            sample_avg_nrmse = np.mean(nrmse_values)
            sample_nrmse_values.append(sample_avg_nrmse)
    
    # Return average NRMSE across all samples
    return np.mean(sample_nrmse_values) if sample_nrmse_values else None

def generate_plot():
    """Generate the threshold analysis plot"""
    
    # Load data
    samples, original_avg_nrmse = load_vla_clip_data()
    
    # Calculate NRMSE for different numbers of instructions (1 to 128)
    num_instructions_range = range(1, 129)  # 1 to 128
    avg_nrmse_values = []
    
    print("Calculating average NRMSE for different instruction thresholds...")
    for num_instructions in num_instructions_range:
        avg_nrmse = calculate_threshold_nrmse(samples, num_instructions)
        avg_nrmse_values.append(avg_nrmse)
        
        if num_instructions % 10 == 0 or num_instructions == 1:
            print(f"  {num_instructions} instructions: {avg_nrmse:.4f}")
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot the threshold curve
    plt.plot(num_instructions_range, avg_nrmse_values, 'b-', linewidth=2, 
             label='Average NRMSE', marker='o', markersize=3)
    
    # Add horizontal line for original instruction baseline
    plt.axhline(y=original_avg_nrmse, color='r', linestyle='--', linewidth=2,
                label=f'Original Instructions Baseline ({original_avg_nrmse:.4f})')
    
    # Formatting
    plt.xlabel('Number of Rephrased Instructions', fontsize=12)
    plt.ylabel('Average NRMSE across 100 Datapoints', fontsize=12)
    plt.title('Action Error vs Number of Rephrased Instructions', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    # Set reasonable y-axis limits
    min_nrmse = min(avg_nrmse_values)
    max_nrmse = max(avg_nrmse_values)
    y_margin = (max_nrmse - min_nrmse) * 0.1
    plt.ylim(min_nrmse - y_margin, max_nrmse + y_margin)
    
    # Add some key statistics as text
    best_nrmse = min(avg_nrmse_values)
    best_num = num_instructions_range[avg_nrmse_values.index(best_nrmse)]
    
    textstr = f'Best NRMSE: {best_nrmse:.4f} (at {best_num} instructions)\n'
    textstr += f'Original NRMSE: {original_avg_nrmse:.4f}\n'
    textstr += f'Improvement: {((original_avg_nrmse - best_nrmse) / original_avg_nrmse * 100):.1f}%'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    # Save plot
    output_file = 'vla_clip_threshold_analysis.png'
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as: {output_file}")
    
    # Also save as PDF
    plt.savefig('vla_clip_threshold_analysis.pdf', bbox_inches='tight')
    print(f"Plot also saved as: vla_clip_threshold_analysis.pdf")
    
    plt.show()
    
    # Print summary statistics
    print(f"\n=== Summary Statistics ===")
    print(f"Original instructions average NRMSE: {original_avg_nrmse:.4f}")
    print(f"Best NRMSE achieved: {best_nrmse:.4f} (with {best_num} instructions)")
    print(f"Improvement over original: {((original_avg_nrmse - best_nrmse) / original_avg_nrmse * 100):.1f}%")
    print(f"NRMSE at 128 instructions: {avg_nrmse_values[-1]:.4f}")
    
    # Save numerical results
    results_df = pd.DataFrame({
        'num_instructions': num_instructions_range,
        'avg_nrmse': avg_nrmse_values
    })
    results_df.to_csv('vla_clip_threshold_results.csv', index=False)
    print(f"Numerical results saved as: vla_clip_threshold_results.csv")
    
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
    
    # Generate the plot
    results = generate_plot()
