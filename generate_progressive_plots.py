#!/usr/bin/env python3
"""
Generate 5 progressive plots showing the addition of each curve in order:
1. Original Instruction baseline only
2. Original + Rephrased Instruction baselines
3. Original + Rephrased + RoboMonkey Verifier
4. Original + Rephrased + RoboMonkey + Random Selection
5. Original + Rephrased + RoboMonkey + Random + VLA-CLIP Verifier (complete)
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

def load_vla_clip_data():
    """Load VLA-CLIP scores and organize by sample"""
    print("Loading VLA-CLIP scores...")
    with open('/root/bridge_dataset_extracted/bridge_vla_clip_scores_20250809_194343.json', 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data['results'])} VLA-CLIP results")
    
    # Organize data by sample_id
    samples = defaultdict(list)
    original_nrmse = []
    
    for result in data['results']:
        sample_id = result['sample_id']
        samples[sample_id].append(result)
        
        # Collect original instruction NRMSE
        if result['is_original']:
            original_nrmse.append(result['openvla_nrmse'])
    
    print(f"Found {len(samples)} unique samples for VLA-CLIP")
    print(f"Original instruction average NRMSE: {np.mean(original_nrmse):.4f}")
    
    return samples, np.mean(original_nrmse)

def load_monkey_verifier_data():
    """Load monkey verifier scores and organize by sample"""
    print("Loading monkey verifier scores...")
    
    # Find the most recent monkey verifier file
    monkey_files = [f for f in os.listdir('/root/bridge_dataset_extracted/') 
                   if f.startswith('bridge_monkey_verifier_scores_') and f.endswith('.json')]
    if not monkey_files:
        raise FileNotFoundError("No monkey verifier scores file found!")
    monkey_file = sorted(monkey_files)[-1]  # Get the most recent one
    
    with open(f'/root/bridge_dataset_extracted/{monkey_file}', 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data['results'])} monkey verifier results from {monkey_file}")
    
    # Organize data by sample_id
    samples = defaultdict(list)
    original_nrmse = []
    
    for result in data['results']:
        sample_id = result['sample_id']
        samples[sample_id].append(result)
        
        # Collect original instruction NRMSE
        if result['is_original']:
            original_nrmse.append(result['openvla_nrmse'])
    
    print(f"Found {len(samples)} unique samples for monkey verifier")
    print(f"Original instruction average NRMSE (monkey data): {np.mean(original_nrmse):.4f}")
    
    return samples, np.mean(original_nrmse)

def calculate_threshold_nrmse_vla_clip(samples, num_instructions):
    """Calculate NRMSE using VLA-CLIP score selection"""
    sample_nrmse_values = []
    
    for sample_id, results in samples.items():
        original = [r for r in results if r['is_original']]
        rephrased = [r for r in results if not r['is_original']]
        
        if not original:
            continue
            
        if num_instructions == 1:
            selected = original
        else:
            rephrased_sorted = sorted(rephrased, key=lambda x: x['vla_clip_similarity_score'], reverse=True)
            top_rephrased = rephrased_sorted[:num_instructions-1]
            selected = original + top_rephrased
        
        if selected:
            nrmse_values = [r['openvla_nrmse'] for r in selected]
            sample_avg_nrmse = np.mean(nrmse_values)
            sample_nrmse_values.append(sample_avg_nrmse)
    
    return np.mean(sample_nrmse_values) if sample_nrmse_values else None

def calculate_threshold_nrmse_monkey(samples, num_instructions):
    """Calculate NRMSE using monkey verifier score selection"""
    sample_nrmse_values = []
    
    for sample_id, results in samples.items():
        original = [r for r in results if r['is_original']]
        rephrased = [r for r in results if not r['is_original'] and r['monkey_verifier_score'] is not None]
        
        if not original:
            continue
            
        if num_instructions == 1:
            selected = original
        else:
            rephrased_sorted = sorted(rephrased, key=lambda x: x['monkey_verifier_score'], reverse=True)
            top_rephrased = rephrased_sorted[:num_instructions-1]
            selected = original + top_rephrased
        
        if selected:
            nrmse_values = [r['openvla_nrmse'] for r in selected]
            sample_avg_nrmse = np.mean(nrmse_values)
            sample_nrmse_values.append(sample_avg_nrmse)
    
    return np.mean(sample_nrmse_values) if sample_nrmse_values else None

def calculate_threshold_nrmse_random(samples, num_instructions, random_seed=42):
    """Calculate NRMSE using random instruction selection"""
    np.random.seed(random_seed)
    sample_nrmse_values = []
    
    for sample_id, results in samples.items():
        original = [r for r in results if r['is_original']]
        rephrased = [r for r in results if not r['is_original']]
        
        if not original:
            continue
            
        if num_instructions == 1:
            selected = original
        else:
            if len(rephrased) >= num_instructions - 1:
                random_rephrased = np.random.choice(rephrased, size=num_instructions-1, replace=False).tolist()
            else:
                random_rephrased = rephrased
            selected = original + random_rephrased
        
        if selected:
            nrmse_values = [r['openvla_nrmse'] for r in selected]
            sample_avg_nrmse = np.mean(nrmse_values)
            sample_nrmse_values.append(sample_avg_nrmse)
    
    return np.mean(sample_nrmse_values) if sample_nrmse_values else None

def calculate_all_curves():
    """Calculate all the curve data"""
    print("Loading data and calculating all curves...")
    
    # Load data
    vla_clip_samples, _ = load_vla_clip_data()
    monkey_samples, _ = load_monkey_verifier_data()
    
    # Calculate NRMSE for specific instruction thresholds
    num_instructions_range = [1, 2, 4, 6, 8, 16, 32, 64, 128]
    
    vla_clip_nrmse_values = []
    monkey_nrmse_values = []
    random_nrmse_values = []
    
    print("Calculating curves...")
    for num_instructions in num_instructions_range:
        vla_clip_nrmse = calculate_threshold_nrmse_vla_clip(vla_clip_samples, num_instructions)
        vla_clip_nrmse_values.append(vla_clip_nrmse)
        
        monkey_nrmse = calculate_threshold_nrmse_monkey(monkey_samples, num_instructions)
        monkey_nrmse_values.append(monkey_nrmse)
        
        random_nrmse = calculate_threshold_nrmse_random(vla_clip_samples, num_instructions)
        random_nrmse_values.append(random_nrmse)
    
    return num_instructions_range, vla_clip_nrmse_values, monkey_nrmse_values, random_nrmse_values

def create_plot_base():
    """Create base plot with common styling"""
    fig, ax = plt.subplots()
    
    # Axis labels and scale
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("Action Error (Average NRMSE)")
    ax.set_xscale('log', base=2)
    
    # Grid and border
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_color('black')
    
    return fig, ax

def save_plot_1(output_dir, num_instructions_range):
    """Plot 1: Original Instruction baseline only"""
    fig, ax = create_plot_base()
    
    # Add original instruction baseline
    ax.axhline(y=0.1665, color='black', linestyle='-', linewidth=2.0, 
               label='Original Instruction (Greedy)')
    
    # Set x-axis
    ax.set_xticks(num_instructions_range)
    ax.set_xticklabels([str(t) for t in num_instructions_range])
    
    # Set y-axis limits
    ax.set_ylim(0.150, 0.185)
    
    # Legend
    legend = ax.legend(loc='lower right', framealpha=0.95, edgecolor='black', fontsize=11)
    legend.get_frame().set_linewidth(1.5)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/plot_1_original_only.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/plot_1_original_only.pdf', bbox_inches='tight')
    plt.close()
    print("Generated Plot 1: Original Instruction baseline")

def save_plot_2(output_dir, num_instructions_range):
    """Plot 2: Original + Rephrased Instruction baselines"""
    fig, ax = create_plot_base()
    
    # Add both baselines
    ax.axhline(y=0.1665, color='black', linestyle='-', linewidth=2.0, 
               label='Original Instruction (Greedy)')
    ax.axhline(y=0.1793, color='gray', linestyle='-', linewidth=2.0, 
               label='Rephrased Instruction (Greedy)')
    
    # Set x-axis
    ax.set_xticks(num_instructions_range)
    ax.set_xticklabels([str(t) for t in num_instructions_range])
    
    # Set y-axis limits
    ax.set_ylim(0.150, 0.185)
    
    # Legend
    legend = ax.legend(loc='lower right', framealpha=0.95, edgecolor='black', fontsize=11)
    legend.get_frame().set_linewidth(1.5)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/plot_2_both_baselines.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/plot_2_both_baselines.pdf', bbox_inches='tight')
    plt.close()
    print("Generated Plot 2: Original + Rephrased baselines")

def save_plot_3(output_dir, num_instructions_range, monkey_nrmse_values):
    """Plot 3: Original + Rephrased + RoboMonkey Verifier"""
    fig, ax = create_plot_base()
    
    # Add baselines
    ax.axhline(y=0.1665, color='black', linestyle='-', linewidth=2.0, 
               label='Original Instruction (Greedy)')
    ax.axhline(y=0.1793, color='gray', linestyle='-', linewidth=2.0, 
               label='Rephrased Instruction (Greedy)')
    
    # Add RoboMonkey curve
    ax.plot(num_instructions_range, monkey_nrmse_values, label='RoboMonkey Verifier', 
            color='#d62728', marker='s', markersize=10, linestyle='--', linewidth=2.5)
    
    # Set x-axis
    ax.set_xticks(num_instructions_range)
    ax.set_xticklabels([str(t) for t in num_instructions_range])
    
    # Set y-axis limits
    ax.set_ylim(0.150, 0.185)
    
    # Legend
    legend = ax.legend(loc='lower right', framealpha=0.95, edgecolor='black', fontsize=11)
    legend.get_frame().set_linewidth(1.5)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/plot_3_add_robomonkey.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/plot_3_add_robomonkey.pdf', bbox_inches='tight')
    plt.close()
    print("Generated Plot 3: + RoboMonkey Verifier")

def save_plot_4(output_dir, num_instructions_range, monkey_nrmse_values, random_nrmse_values):
    """Plot 4: Original + Rephrased + RoboMonkey + Random Selection"""
    fig, ax = create_plot_base()
    
    # Add baselines
    ax.axhline(y=0.1665, color='black', linestyle='-', linewidth=2.0, 
               label='Original Instruction (Greedy)')
    ax.axhline(y=0.1793, color='gray', linestyle='-', linewidth=2.0, 
               label='Rephrased Instruction (Greedy)')
    
    # Add curves
    ax.plot(num_instructions_range, monkey_nrmse_values, label='RoboMonkey Verifier', 
            color='#d62728', marker='s', markersize=10, linestyle='--', linewidth=2.5)
    ax.plot(num_instructions_range, random_nrmse_values, label='Random Selection', 
            color='#ff7f0e', marker='^', markersize=10, linestyle=':', linewidth=2.5)
    
    # Set x-axis
    ax.set_xticks(num_instructions_range)
    ax.set_xticklabels([str(t) for t in num_instructions_range])
    
    # Set y-axis limits
    ax.set_ylim(0.150, 0.185)
    
    # Legend
    legend = ax.legend(loc='lower right', framealpha=0.95, edgecolor='black', fontsize=11)
    legend.get_frame().set_linewidth(1.5)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/plot_4_add_random.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/plot_4_add_random.pdf', bbox_inches='tight')
    plt.close()
    print("Generated Plot 4: + Random Selection")

def save_plot_5(output_dir, num_instructions_range, vla_clip_nrmse_values, monkey_nrmse_values, random_nrmse_values):
    """Plot 5: Complete plot with all curves"""
    fig, ax = create_plot_base()
    
    # Add baselines
    ax.axhline(y=0.1665, color='black', linestyle='-', linewidth=2.0, 
               label='Original Instruction (Greedy)')
    ax.axhline(y=0.1793, color='gray', linestyle='-', linewidth=2.0, 
               label='Rephrased Instruction (Greedy)')
    
    # Add all curves
    ax.plot(num_instructions_range, monkey_nrmse_values, label='RoboMonkey Verifier', 
            color='#d62728', marker='s', markersize=10, linestyle='--', linewidth=2.5)
    ax.plot(num_instructions_range, random_nrmse_values, label='Random Selection', 
            color='#ff7f0e', marker='^', markersize=10, linestyle=':', linewidth=2.5)
    ax.plot(num_instructions_range, vla_clip_nrmse_values, label='VLA-CLIP Verifier', 
            color='#1f77b4', marker='o', markersize=10, linestyle='-', linewidth=2.5)
    
    # Set x-axis
    ax.set_xticks(num_instructions_range)
    ax.set_xticklabels([str(t) for t in num_instructions_range])
    
    # Set y-axis limits
    ax.set_ylim(0.150, 0.185)
    
    # Legend
    legend = ax.legend(loc='lower right', framealpha=0.95, edgecolor='black', fontsize=11)
    legend.get_frame().set_linewidth(1.5)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/plot_5_complete.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/plot_5_complete.pdf', bbox_inches='tight')
    plt.close()
    print("Generated Plot 5: Complete with all curves")

def generate_progressive_plots():
    """Generate all 5 progressive plots"""
    
    # Create output directory
    output_dir = 'progressive_plots'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")
    
    # Calculate all curve data
    num_instructions_range, vla_clip_nrmse_values, monkey_nrmse_values, random_nrmse_values = calculate_all_curves()
    
    # Generate all 5 plots
    print("\nGenerating progressive plots...")
    save_plot_1(output_dir, num_instructions_range)
    save_plot_2(output_dir, num_instructions_range)
    save_plot_3(output_dir, num_instructions_range, monkey_nrmse_values)
    save_plot_4(output_dir, num_instructions_range, monkey_nrmse_values, random_nrmse_values)
    save_plot_5(output_dir, num_instructions_range, vla_clip_nrmse_values, monkey_nrmse_values, random_nrmse_values)
    
    # Save numerical data
    results_df = pd.DataFrame({
        'num_instructions': num_instructions_range,
        'vla_clip_nrmse': vla_clip_nrmse_values,
        'monkey_verifier_nrmse': monkey_nrmse_values,
        'random_selection_nrmse': random_nrmse_values
    })
    results_df.to_csv(f'{output_dir}/progressive_plots_data.csv', index=False)
    
    print(f"\nAll progressive plots saved in: {output_dir}/")
    print("Files generated:")
    print("  - plot_1_original_only.png/pdf")
    print("  - plot_2_both_baselines.png/pdf") 
    print("  - plot_3_add_robomonkey.png/pdf")
    print("  - plot_4_add_random.png/pdf")
    print("  - plot_5_complete.png/pdf")
    print("  - progressive_plots_data.csv")
    
    return results_df

if __name__ == "__main__":
    # Check if required data files exist
    required_files = [
        '/root/bridge_dataset_extracted/bridge_vla_clip_scores_20250809_194343.json'
    ]
    
    # Check for monkey verifier files
    monkey_files = [f for f in os.listdir('/root/bridge_dataset_extracted/') 
                   if f.startswith('bridge_monkey_verifier_scores_') and f.endswith('.json')]
    
    missing_files = []
    for f in required_files:
        if not os.path.exists(f):
            missing_files.append(f)
    
    if not monkey_files:
        missing_files.append("bridge_monkey_verifier_scores_*.json")
    
    if missing_files:
        print("Error: Missing required files:")
        for f in missing_files:
            print(f"  - {f}")
        exit(1)
    
    # Generate the progressive plots
    results = generate_progressive_plots()
