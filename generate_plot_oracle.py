#!/usr/bin/env python3
"""
Oracle verifier plot showing how action error decreases as the proposal distribution size increases.

X-axis: Number of rephrased instructions (proposal distribution size)
Y-axis: Action Error (Average NRMSE)

The oracle verifier has perfect knowledge and always selects the single best instruction
from the available proposal distribution. This demonstrates the theoretical upper bound
of performance achievable with perfect instruction selection.
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

def load_openvla_actions_data():
    """Load OpenVLA actions data and organize by sample"""
    print("Loading OpenVLA actions data...")
    with open('/root/validate_clip_verifier/bridge_openvla_actions_20250809_192948.json', 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data['results'])} OpenVLA action results")
    
    # Organize data by sample_id
    samples = defaultdict(list)
    original_nrmse = []
    
    for result in data['results']:
        sample_id = result['sample_id']
        samples[sample_id].append(result)
        
        # Collect original instruction NRMSE
        if result['is_original']:
            original_nrmse.append(result['nrmse'])
    
    print(f"Found {len(samples)} unique samples for OpenVLA actions")
    print(f"Original instruction average NRMSE: {np.mean(original_nrmse):.4f}")
    
    return samples, np.mean(original_nrmse)

def calculate_oracle_nrmse(samples, num_instructions):
    """
    Oracle verifier: For each sample, select the SINGLE BEST instruction from a proposal
    distribution of size num_instructions. The oracle has perfect knowledge and always
    selects the instruction with the lowest NRMSE from the available candidates.
    
    With more instructions in the proposal distribution, the oracle has more options
    to choose from and should achieve lower error (better performance).
    """
    sample_nrmse_values = []
    
    for sample_id, results in samples.items():
        # Separate original and rephrased instructions
        original = [r for r in results if r['is_original']]
        rephrased = [r for r in results if not r['is_original']]
        
        if not original:
            continue
            
        if num_instructions == 1:
            # Only consider original instruction
            candidate_pool = original
        else:
            # Consider original + (num_instructions - 1) rephrased instructions
            # This creates a proposal distribution of size num_instructions
            available_rephrased = rephrased[:num_instructions - 1] if len(rephrased) >= num_instructions - 1 else rephrased
            candidate_pool = original + available_rephrased
        
        # Oracle selects the SINGLE BEST instruction from the candidate pool
        if candidate_pool:
            best_instruction = min(candidate_pool, key=lambda x: x['nrmse'])
            sample_nrmse_values.append(best_instruction['nrmse'])
    
    # Return average NRMSE across all samples
    return np.mean(sample_nrmse_values) if sample_nrmse_values else None

def generate_oracle_plot():
    """Generate the oracle verifier plot showing the benefit of larger proposal distributions"""
    
    # Load data
    openvla_samples, original_avg_nrmse = load_openvla_actions_data()
    
    # Calculate NRMSE for different proposal distribution sizes
    num_instructions_range = [1, 2, 4, 8, 16, 32, 64, 128]
    oracle_nrmse_values = []
    
    print("Calculating oracle NRMSE for different proposal distribution sizes...")
    
    for num_instructions in num_instructions_range:
        oracle_nrmse = calculate_oracle_nrmse(openvla_samples, num_instructions)
        oracle_nrmse_values.append(oracle_nrmse)
        
        improvement = ((original_avg_nrmse - oracle_nrmse) / original_avg_nrmse * 100) if oracle_nrmse else 0
        print(f"  {num_instructions} instructions - Oracle: {oracle_nrmse:.4f} ({improvement:.1f}% improvement)")
    
    # Create the plot with publication-quality styling
    fig, ax = plt.subplots()
    
    # Colors and styling
    color_oracle = '#2ca02c'  # Green
    marker_oracle = 'D'  # Diamond
    
    # Add baseline for original instruction
    ax.axhline(y=original_avg_nrmse, color='black', linestyle='-', linewidth=2.0, 
               label='Original Instruction Baseline')
    
    # Plot Oracle Verifier curve
    ax.plot(num_instructions_range, oracle_nrmse_values, label='Oracle Verifier', 
            color=color_oracle, marker=marker_oracle, markersize=12, linestyle='-', linewidth=3.0)
    
    # Axis labels and scale
    ax.set_xlabel("Proposal Distribution Size (Number of Instructions)")
    ax.set_ylabel("Action Error (Average NRMSE)")
    ax.set_xscale('log', base=2)
    ax.set_xticks(num_instructions_range)
    ax.set_xticklabels([str(t) for t in num_instructions_range])
    
    # Set y-axis limits to better show the improvement
    y_min = min(oracle_nrmse_values) * 0.9
    y_max = original_avg_nrmse * 1.1
    ax.set_ylim(y_min, y_max)
    
    # Grid and border
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_color('black')
    
    # Legend
    legend = ax.legend(loc='upper right', framealpha=0.95, edgecolor='black', fontsize=14)
    legend.get_frame().set_linewidth(1.5)
    
    # Title
    ax.set_title('Oracle Verifier: Perfect Instruction Selection\nfrom Increasing Proposal Distribution Sizes', 
                 fontsize=18, pad=20)
    
    # Save plot
    output_file = 'oracle_verifier_analysis.png'
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as: {output_file}")
    
    # Also save as PDF
    plt.savefig('oracle_verifier_analysis.pdf', bbox_inches='tight')
    print(f"Plot also saved as: oracle_verifier_analysis.pdf")
    
    plt.show()
    
    # Calculate and print detailed statistics
    best_oracle_nrmse = min([v for v in oracle_nrmse_values if v is not None])
    best_oracle_idx = [v for v in oracle_nrmse_values if v is not None].index(best_oracle_nrmse)
    best_oracle_num = num_instructions_range[best_oracle_idx]
    
    print(f"\n=== Oracle Verifier Analysis ===")
    print(f"Original instruction baseline NRMSE: {original_avg_nrmse:.4f}")
    print(f"Best oracle NRMSE: {best_oracle_nrmse:.4f} (with {best_oracle_num} instructions)")
    print(f"Maximum improvement: {((original_avg_nrmse - best_oracle_nrmse) / original_avg_nrmse * 100):.1f}%")
    
    # Show improvement at each step
    print(f"\nImprovement by proposal distribution size:")
    for i, (num_instr, nrmse) in enumerate(zip(num_instructions_range, oracle_nrmse_values)):
        if nrmse is not None:
            improvement = ((original_avg_nrmse - nrmse) / original_avg_nrmse * 100)
            print(f"  {num_instr:3d} instructions: {nrmse:.4f} NRMSE ({improvement:5.1f}% improvement)")
    
    # Calculate relative improvement between consecutive sizes
    print(f"\nRelative improvement between consecutive sizes:")
    for i in range(1, len(oracle_nrmse_values)):
        if oracle_nrmse_values[i] is not None and oracle_nrmse_values[i-1] is not None:
            prev_nrmse = oracle_nrmse_values[i-1]
            curr_nrmse = oracle_nrmse_values[i]
            relative_improvement = ((prev_nrmse - curr_nrmse) / prev_nrmse * 100)
            print(f"  {num_instructions_range[i-1]:3d} → {num_instructions_range[i]:3d}: {relative_improvement:5.1f}% additional improvement")
    
    # Save numerical results
    results_df = pd.DataFrame({
        'proposal_distribution_size': num_instructions_range,
        'oracle_nrmse': oracle_nrmse_values,
        'improvement_percent': [((original_avg_nrmse - nrmse) / original_avg_nrmse * 100) if nrmse else 0 
                               for nrmse in oracle_nrmse_values]
    })
    results_df.to_csv('oracle_verifier_results.csv', index=False)
    print(f"\nNumerical results saved as: oracle_verifier_results.csv")
    
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
    required_file = '/root/validate_clip_verifier/bridge_openvla_actions_20250809_192948.json'
    
    if not os.path.exists(required_file):
        print(f"Error: Missing required file: {required_file}")
        exit(1)
    
    # Generate the oracle plot
    results = generate_oracle_plot()
