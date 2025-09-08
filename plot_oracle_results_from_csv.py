#!/usr/bin/env python3
"""
Plot oracle verifier results from saved CSV data.

This script reads the oracle_rephrases_scaling_results.csv file and generates
the same plot showing oracle performance vs samples per rephrase for different
numbers of rephrases.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Set up the matplotlib parameters with larger fonts for a publication-quality figure
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 16,
    'axes.labelsize': 18,
    'axes.titlesize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 14,
    'figure.figsize': (12, 8),
    'text.usetex': False,
})

def plot_oracle_results_from_csv(csv_file_path):
    """
    Generate the oracle verifier plot from saved CSV results.
    
    Args:
        csv_file_path: Path to the CSV file containing oracle results
    """
    
    # Load the CSV data
    print(f"Loading oracle results from: {csv_file_path}")
    df = pd.read_csv(csv_file_path)
    
    print(f"Loaded {len(df)} data points")
    print(f"Number of rephrases: {sorted(df['num_rephrases'].unique())}")
    print(f"Samples per rephrase: {sorted(df['samples_per_rephrase'].unique())}")
    
    # Calculate baseline NRMSE (assuming ~16.65% based on previous results)
    # We can infer this from the improvement percentages
    sample_row = df.iloc[0]
    baseline_nrmse = sample_row['oracle_nrmse'] / (1 - sample_row['improvement_percent']/100)
    print(f"Inferred baseline NRMSE: {baseline_nrmse:.4f}")
    
    # Create the plot
    fig, ax = plt.subplots()
    
    # Colors for different rephrase counts
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    markers = ['o', 's', '^', 'D', 'v', '<', '>']
    
    # Get unique rephrase counts and sort them, excluding 1 rephrase
    rephrase_counts = sorted([r for r in df['num_rephrases'].unique() if r != 1])
    
    # Add repeated sampling curve first (so it appears in front)
    repeated_thresholds = [1, 2, 4, 8, 16, 32, 64, 128]
    repeated_vla_nrmse = [0.16, 0.1395698519403141, 0.12382259063373457, 0.10865227458014033,
                          0.09702733800476325, 0.09015774016003288, 0.08320682061341735, 0.07728755638551704]
    
    ax.plot(repeated_thresholds, repeated_vla_nrmse, 'o-', linewidth=3.0, markersize=10,
           color='red', marker='*', label='Repeated Sampling', markeredgewidth=1, 
           markeredgecolor='white', zorder=5)
    
    # Add instruction rephrasing (greedy) curve second (so it appears in front)
    greedy_thresholds = [1, 2, 4, 8, 16, 32, 64, 128]
    greedy_vla_nrmse = [0.16, 0.135, 0.12, 0.1014, 0.09, 0.08187, 0.074474, 0.07022]
    
    ax.plot(greedy_thresholds, greedy_vla_nrmse, 'o-', linewidth=3.0, markersize=10,
           color='orange', marker='P', label='Instruction Rephrasing', markeredgewidth=1, 
           markeredgecolor='white', zorder=4)
    
    # Plot each rephrase count as a separate curve (behind the main curves)
    for i, num_rephrases in enumerate(rephrase_counts):
        # Filter data for this number of rephrases
        rephrase_data = df[df['num_rephrases'] == num_rephrases].copy()
        rephrase_data = rephrase_data.sort_values('samples_per_rephrase')
        
        samples_x = rephrase_data['samples_per_rephrase'].values
        nrmse_y = rephrase_data['oracle_nrmse'].values
        
        # Plot the curve for this number of rephrases (with lower zorder so they appear behind)
        ax.plot(samples_x, nrmse_y, 'o-', linewidth=2.0, markersize=7, 
               color=colors[i % len(colors)], marker=markers[i % len(markers)], 
               label=f'{num_rephrases} Rephrases', markeredgewidth=1, markeredgecolor='white',
               zorder=3)
        
        print(f"\n{num_rephrases} Rephrases:")
        for _, row in rephrase_data.iterrows():
            augmentation_note = " (augmented)" if row['samples_per_rephrase'] > 4 else ""
            print(f"  {int(row['samples_per_rephrase']):3d} samples: {row['oracle_nrmse']:.4f} NRMSE ({row['improvement_percent']:.1f}% improvement){augmentation_note}")
    
    # Add baseline line for original instruction (behind other curves)
    ax.axhline(y=baseline_nrmse, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, 
               label='Original Instruction Baseline', zorder=1)
    
    # Axis labels and formatting
    ax.set_xlabel("Number of Generated Actions per Rephrase")
    ax.set_ylabel("Action Error (Average NRMSE)")
    ax.set_xscale('log', base=2)
    
    # Set x-axis ticks (combine oracle, repeated sampling, and greedy thresholds)
    samples_per_rephrase_list = sorted(df['samples_per_rephrase'].unique())
    all_thresholds = sorted(set(samples_per_rephrase_list + repeated_thresholds + greedy_thresholds))
    ax.set_xticks(all_thresholds)
    ax.set_xticklabels([str(t) for t in all_thresholds])
    
    # Set y-axis limits to better show the improvement (include repeated sampling and greedy data)
    all_nrmse_values = list(df['oracle_nrmse'].values) + repeated_vla_nrmse + greedy_vla_nrmse
    y_min = min(all_nrmse_values) * 0.9
    y_max = baseline_nrmse * 1.05
    ax.set_ylim(y_min, y_max)
    
    # Grid and border
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_color('black')
    
    # Get handles and labels for legend reordering
    handles, labels = ax.get_legend_handles_labels()
    
    # Find the repeated sampling and greedy entries and move them to the top
    repeated_idx = None
    greedy_idx = None
    
    for i, label in enumerate(labels):
        if 'Repeated Sampling' in label:
            repeated_idx = i
        elif 'Instruction Rephrasing (Greedy)' in label:
            greedy_idx = i
    
    # Reorder: Repeated Sampling first, then Greedy, then the rest
    new_handles = []
    new_labels = []
    
    # Add repeated sampling first
    if repeated_idx is not None:
        new_handles.append(handles[repeated_idx])
        new_labels.append(labels[repeated_idx])
    
    # Add greedy second
    if greedy_idx is not None:
        new_handles.append(handles[greedy_idx])
        new_labels.append(labels[greedy_idx])
    
    # Add remaining entries (excluding repeated sampling and greedy)
    for i, (handle, label) in enumerate(zip(handles, labels)):
        if i != repeated_idx and i != greedy_idx:
            new_handles.append(handle)
            new_labels.append(label)
    
    # Create legend with reordered entries
    legend = ax.legend(new_handles, new_labels, bbox_to_anchor=(1.05, 1), loc='upper left', framealpha=0.95, edgecolor='black', fontsize=10)
    legend.get_frame().set_linewidth(1.5)
    
    # Title
    ax.set_title('Oracle Verifier vs Repeated Sampling: Performance Comparison\nAcross Different Sample Counts (with Gaussian Augmentation)', 
                 fontsize=14, pad=20)
    
    # Save plot
    output_file = '/root/validate_clip_verifier/oracle_results_from_csv.png'
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as: {output_file}")
    
    # Also save as PDF
    plt.savefig('/root/validate_clip_verifier/oracle_results_from_csv.pdf', bbox_inches='tight')
    print(f"Plot also saved as: oracle_results_from_csv.pdf")
    
    plt.show()
    
    # Print summary statistics
    print(f"\n=== Oracle Results Summary ===")
    print(f"Baseline NRMSE: {baseline_nrmse:.4f}")
    
    # Find best overall result from oracle data
    best_result = df.loc[df['oracle_nrmse'].idxmin()]
    print(f"\nBest Oracle Performance:")
    print(f"  {int(best_result['num_rephrases'])} rephrases with {int(best_result['samples_per_rephrase'])} samples per rephrase")
    print(f"  NRMSE: {best_result['oracle_nrmse']:.4f} ({best_result['improvement_percent']:.1f}% improvement)")
    
    # Repeated sampling results
    print(f"\n=== Repeated Sampling Results ===")
    best_repeated_nrmse = min(repeated_vla_nrmse)
    best_repeated_threshold = repeated_thresholds[repeated_vla_nrmse.index(best_repeated_nrmse)]
    best_repeated_improvement = (1 - best_repeated_nrmse/baseline_nrmse) * 100
    
    print(f"Best Repeated Sampling Performance:")
    print(f"  {best_repeated_threshold} samples threshold")
    print(f"  NRMSE: {best_repeated_nrmse:.4f} ({best_repeated_improvement:.1f}% improvement)")
    
    print(f"\nRepeated Sampling Performance by Threshold:")
    for threshold, nrmse in zip(repeated_thresholds, repeated_vla_nrmse):
        improvement = (1 - nrmse/baseline_nrmse) * 100
        print(f"  {threshold:3d} samples: {nrmse:.4f} NRMSE ({improvement:.1f}% improvement)")
    
    # Instruction rephrasing (greedy) results
    print(f"\n=== Instruction Rephrasing (Greedy) Results ===")
    best_greedy_nrmse = min(greedy_vla_nrmse)
    best_greedy_threshold = greedy_thresholds[greedy_vla_nrmse.index(best_greedy_nrmse)]
    best_greedy_improvement = (1 - best_greedy_nrmse/baseline_nrmse) * 100
    
    print(f"Best Greedy Performance:")
    print(f"  {best_greedy_threshold} samples threshold")
    print(f"  NRMSE: {best_greedy_nrmse:.4f} ({best_greedy_improvement:.1f}% improvement)")
    
    print(f"\nGreedy Performance by Threshold:")
    for threshold, nrmse in zip(greedy_thresholds, greedy_vla_nrmse):
        improvement = (1 - nrmse/baseline_nrmse) * 100
        print(f"  {threshold:3d} samples: {nrmse:.4f} NRMSE ({improvement:.1f}% improvement)")
    
    # Analyze the effect of augmentation (excluding 1 rephrase)
    print(f"\nAugmentation Effect Analysis:")
    for num_rephrases in rephrase_counts:
        rephrase_data = df[df['num_rephrases'] == num_rephrases].copy()
        rephrase_data = rephrase_data.sort_values('samples_per_rephrase')
        
        # Find the jump from 4 to 8 samples (where augmentation starts)
        four_samples = rephrase_data[rephrase_data['samples_per_rephrase'] == 4]
        eight_samples = rephrase_data[rephrase_data['samples_per_rephrase'] == 8]
        
        if len(four_samples) > 0 and len(eight_samples) > 0:
            improvement_jump = eight_samples.iloc[0]['improvement_percent'] - four_samples.iloc[0]['improvement_percent']
            print(f"  {int(num_rephrases):2d} rephrases: +{improvement_jump:.1f}% improvement jump (4→8 samples)")
    
    # Analyze scaling with number of rephrases at high sample count
    print(f"\nScaling with Number of Rephrases (at 128 samples per rephrase):")
    high_sample_data = df[df['samples_per_rephrase'] == 128].copy()
    high_sample_data = high_sample_data.sort_values('num_rephrases')
    
    for _, row in high_sample_data.iterrows():
        print(f"  {int(row['num_rephrases']):2d} rephrases: {row['oracle_nrmse']:.4f} NRMSE ({row['improvement_percent']:.1f}% improvement)")
    
    return df

def analyze_diminishing_returns(df):
    """
    Analyze diminishing returns in the oracle results.
    
    Args:
        df: DataFrame containing oracle results
    """
    print(f"\n=== Diminishing Returns Analysis ===")
    
    # For each number of rephrases, analyze the marginal benefit of additional samples (excluding 1 rephrase)
    rephrase_counts = sorted([r for r in df['num_rephrases'].unique() if r != 1])
    
    for num_rephrases in rephrase_counts:
        print(f"\n{int(num_rephrases)} Rephrases - Marginal Benefit per Additional Sample:")
        rephrase_data = df[df['num_rephrases'] == num_rephrases].copy()
        rephrase_data = rephrase_data.sort_values('samples_per_rephrase')
        
        prev_improvement = None
        prev_samples = None
        
        for _, row in rephrase_data.iterrows():
            if prev_improvement is not None:
                marginal_benefit = row['improvement_percent'] - prev_improvement
                additional_samples = int(row['samples_per_rephrase']) - int(prev_samples)
                efficiency = marginal_benefit / additional_samples if additional_samples > 0 else 0
                
                print(f"  {int(prev_samples):3d} → {int(row['samples_per_rephrase']):3d} samples: +{marginal_benefit:5.1f}% ({efficiency:5.2f}% per sample)")
            
            prev_improvement = row['improvement_percent']
            prev_samples = row['samples_per_rephrase']

if __name__ == "__main__":
    # Check if the CSV file exists
    csv_file = '/root/validate_clip_verifier/oracle_rephrases_scaling_results.csv'
    
    if not os.path.exists(csv_file):
        print(f"Error: CSV file not found: {csv_file}")
        print("Please run generate_plot_actions_per_instruction.py first to generate the data.")
        exit(1)
    
    # Generate the plot from CSV data
    df = plot_oracle_results_from_csv(csv_file)
    
    # Perform additional analysis
    analyze_diminishing_returns(df)
    
    print(f"\nPlot generation complete!")
