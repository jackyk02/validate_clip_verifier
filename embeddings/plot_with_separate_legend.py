#!/usr/bin/env python3
"""
Enhanced plot showing average NRMSE vs number of rephrased instructions,
comparing VLA-CLIP, monkey verifier, random selection, and clustering methods.
This version creates the plot with a separate legend.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set up the matplotlib parameters with larger fonts for a publication-quality figure
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 16,
    'axes.labelsize': 18,
    'axes.titlesize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.figsize': (12, 8),
    'text.usetex': False,
})

def create_plot_with_separate_legend():
    """Create the main plot and a separate legend"""
    
    # Load the results
    df = pd.read_csv('/root/validate_clip_verifier/enhanced_threshold_results.csv')
    
    # Extract data
    num_instructions_range = df['num_instructions'].values
    vla_clip_nrmse_values = df['vla_clip_nrmse'].values
    monkey_nrmse_values = df['monkey_verifier_nrmse'].values
    random_nrmse_values = df['random_selection_nrmse'].values
    clustering_nrmse_values = df['clustering_vla_clip_nrmse'].values
    
    # Create figure with subplots: main plot and legend
    fig = plt.figure(figsize=(14, 8))
    
    # Main plot takes up most of the space
    ax_main = plt.subplot2grid((1, 4), (0, 0), colspan=3)
    
    # Legend subplot
    ax_legend = plt.subplot2grid((1, 4), (0, 3))
    ax_legend.axis('off')  # Hide the legend subplot axes
    
    # Colors and markers
    color_clip = '#1f77b4'
    color_monkey = '#d62728'
    color_random = '#ff7f0e'  # Orange
    color_clustering = '#2ca02c'  # Green
    marker_clip = 'o'
    marker_monkey = 's'
    marker_random = '^'  # Triangle
    marker_clustering = 'D'  # Diamond
    
    # Add baseline horizontal lines
    ax_main.axhline(y=0.1665, color='black', linestyle='-', linewidth=2.0, 
                   label='Original Instruction')
    ax_main.axhline(y=0.1793, color='gray', linestyle='-', linewidth=2.0, 
                   label='Rephrased Instruction')
    
    # Plot all curves
    line1 = ax_main.plot(num_instructions_range, vla_clip_nrmse_values, label='VLA-CLIP Verifier', 
            color=color_clip, marker=marker_clip, markersize=10, linestyle='-', linewidth=2.5)
    
    line2 = ax_main.plot(num_instructions_range, monkey_nrmse_values, label='RoboMonkey Verifier', 
            color=color_monkey, marker=marker_monkey, markersize=10, linestyle='--', linewidth=2.5)
    
    line3 = ax_main.plot(num_instructions_range, random_nrmse_values, label='Random Selection', 
            color=color_random, marker=marker_random, markersize=10, linestyle=':', linewidth=2.5)
    
    line4 = ax_main.plot(num_instructions_range, clustering_nrmse_values, label='Clustering + VLA-CLIP', 
            color=color_clustering, marker=marker_clustering, markersize=10, linestyle='-.', linewidth=2.5)
    
    # Axis labels and scale for main plot
    ax_main.set_xlabel("Number of Rephrases")
    ax_main.set_ylabel("Action Error (Average NRMSE)")
    ax_main.set_xscale('log', base=2)
    ax_main.set_xticks(num_instructions_range)
    ax_main.set_xticklabels([str(t) for t in num_instructions_range])
    
    # Grid and border for main plot
    ax_main.grid(True, linestyle='--', alpha=0.7)
    ax_main.set_axisbelow(True)
    for spine in ax_main.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_color('black')
    
    # Create separate legend
    # Get all the line objects and labels
    lines = []
    labels = []
    
    # Add baseline lines
    lines.extend([plt.Line2D([0], [0], color='black', linestyle='-', linewidth=2.0),
                  plt.Line2D([0], [0], color='gray', linestyle='-', linewidth=2.0)])
    labels.extend(['Original Instruction', 'Rephrased Instruction'])
    
    # Add method lines
    lines.extend([plt.Line2D([0], [0], color=color_clip, marker=marker_clip, 
                            markersize=10, linestyle='-', linewidth=2.5),
                  plt.Line2D([0], [0], color=color_monkey, marker=marker_monkey, 
                            markersize=10, linestyle='--', linewidth=2.5),
                  plt.Line2D([0], [0], color=color_random, marker=marker_random, 
                            markersize=10, linestyle=':', linewidth=2.5),
                  plt.Line2D([0], [0], color=color_clustering, marker=marker_clustering, 
                            markersize=10, linestyle='-.', linewidth=2.5)])
    labels.extend(['VLA-CLIP Verifier', 'RoboMonkey Verifier', 'Random Selection', 'Clustering + VLA-CLIP'])
    
    # Place legend in the separate subplot
    legend = ax_legend.legend(lines, labels, loc='center', framealpha=0.95, 
                             edgecolor='black', fontsize=14)
    legend.get_frame().set_linewidth(1.5)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save the plot
    output_file = '/root/validate_clip_verifier/enhanced_threshold_analysis_separate_legend.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {output_file}")
    
    # Also save as PDF
    plt.savefig('/root/validate_clip_verifier/enhanced_threshold_analysis_separate_legend.pdf', bbox_inches='tight')
    print(f"Plot also saved as PDF")
    
    plt.show()
    
    # Print some statistics
    print(f"\n=== Performance Summary ===")
    print(f"Best VLA-CLIP NRMSE: {min(vla_clip_nrmse_values):.4f}")
    print(f"Best Monkey NRMSE: {min(monkey_nrmse_values):.4f}")
    print(f"Best Random NRMSE: {min(random_nrmse_values):.4f}")
    print(f"Best Clustering NRMSE: {min(clustering_nrmse_values):.4f}")
    
    # Find the best overall method
    best_methods = [
        ("VLA-CLIP", min(vla_clip_nrmse_values)),
        ("Monkey", min(monkey_nrmse_values)),
        ("Random", min(random_nrmse_values)),
        ("Clustering", min(clustering_nrmse_values))
    ]
    best_method, best_score = min(best_methods, key=lambda x: x[1])
    print(f"Best performing method: {best_method} with NRMSE {best_score:.4f}")

def create_alternative_layout():
    """Create an alternative layout with legend below the plot"""
    
    # Load the results
    df = pd.read_csv('/root/validate_clip_verifier/enhanced_threshold_results.csv')
    
    # Extract data
    num_instructions_range = df['num_instructions'].values
    vla_clip_nrmse_values = df['vla_clip_nrmse'].values
    monkey_nrmse_values = df['monkey_verifier_nrmse'].values
    random_nrmse_values = df['random_selection_nrmse'].values
    clustering_nrmse_values = df['clustering_vla_clip_nrmse'].values
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Colors and markers
    color_clip = '#1f77b4'
    color_monkey = '#d62728'
    color_random = '#ff7f0e'  # Orange
    color_clustering = '#2ca02c'  # Green
    marker_clip = 'o'
    marker_monkey = 's'
    marker_random = '^'  # Triangle
    marker_clustering = 'D'  # Diamond
    
    # Add baseline horizontal lines
    ax.axhline(y=0.1665, color='black', linestyle='-', linewidth=2.0, 
               label='Original Instruction')
    ax.axhline(y=0.1793, color='gray', linestyle='-', linewidth=2.0, 
               label='Rephrased Instruction')
    
    # Plot all curves
    ax.plot(num_instructions_range, vla_clip_nrmse_values, label='VLA-CLIP Verifier', 
            color=color_clip, marker=marker_clip, markersize=10, linestyle='-', linewidth=2.5)
    
    ax.plot(num_instructions_range, monkey_nrmse_values, label='RoboMonkey Verifier', 
            color=color_monkey, marker=marker_monkey, markersize=10, linestyle='--', linewidth=2.5)
    
    ax.plot(num_instructions_range, random_nrmse_values, label='Random Selection', 
            color=color_random, marker=marker_random, markersize=10, linestyle=':', linewidth=2.5)
    
    ax.plot(num_instructions_range, clustering_nrmse_values, label='Clustering + VLA-CLIP', 
            color=color_clustering, marker=marker_clustering, markersize=10, linestyle='-.', linewidth=2.5)
    
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
    
    # Place legend below the plot
    legend = ax.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', 
                      ncol=3, framealpha=0.95, edgecolor='black', fontsize=14)
    legend.get_frame().set_linewidth(1.5)
    
    # Save the plot
    output_file = '/root/validate_clip_verifier/enhanced_threshold_analysis_legend_below.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Alternative plot saved as: {output_file}")
    
    plt.show()

if __name__ == "__main__":
    print("Creating plot with separate legend...")
    create_plot_with_separate_legend()
    
    print("\nCreating alternative plot with legend below...")
    create_alternative_layout()

