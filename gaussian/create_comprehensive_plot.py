#!/usr/bin/env python3
"""
Create comprehensive plot from Gaussian threshold results showing:
- VLA-CLIP Verifier performance
- Monkey Verifier performance  
- Random Selection baseline
- Comparison and improvement analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set up matplotlib for publication-quality figures
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

def load_data():
    """Load the threshold results data"""
    df = pd.read_csv('gaussian_threshold_results.csv')
    return df

def create_main_comparison_plot(df):
    """Create the main comparison plot"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Colors and markers
    color_vla_clip = '#1f77b4'  # Blue
    color_monkey = '#d62728'    # Red  
    color_random = '#ff7f0e'    # Orange
    
    # Plot all three methods
    ax.plot(df['num_samples'], df['vla_clip_nrmse'], 
            label='VLA-CLIP Verifier', color=color_vla_clip, 
            marker='o', markersize=8, linestyle='-', linewidth=2.5)
    
    ax.plot(df['num_samples'], df['monkey_verifier_nrmse'], 
            label='RoboMonkey Verifier', color=color_monkey, 
            marker='s', markersize=8, linestyle='--', linewidth=2.5)
    
    # ax.plot(df['num_samples'], df['random_selection_nrmse'], 
    #         label='Random Selection', color=color_random, 
    #         marker='^', markersize=8, linestyle=':', linewidth=2.5)
    
    # Add baseline horizontal line (using the 128-sample value as true baseline)
    baseline_nrmse = df.iloc[-1]['random_selection_nrmse']  # 128 samples = all data baseline
    ax.axhline(y=0.1665, color='black', linestyle='-', linewidth=2.0, alpha=0.7,
               label=f'Greedy Decoding')
    
    # Formatting
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("Action Error (Average NRMSE)")
    ax.set_xscale('log', base=2)
    ax.set_xticks(df['num_samples'])
    ax.set_xticklabels([str(int(x)) for x in df['num_samples']])
    
    # Grid and styling
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_color('black')
    
    # Legend
    legend = ax.legend(loc='upper right', framealpha=0.95, edgecolor='black')
    legend.get_frame().set_linewidth(1.5)
    
    # Title
    ax.set_title('Verifier Performance on Gaussian-Augmented Samples', fontsize=20, pad=20)
    
    # Set y-axis limits with some padding
    y_min = df[['vla_clip_nrmse', 'monkey_verifier_nrmse', 'random_selection_nrmse']].min().min()
    y_max = df[['vla_clip_nrmse', 'monkey_verifier_nrmse', 'random_selection_nrmse']].max().max()
    ax.set_ylim(y_min * 0.98, y_max * 1.02)
    
    plt.tight_layout()
    plt.savefig('gaussian_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig('gaussian_comprehensive_analysis.pdf', bbox_inches='tight')
    plt.show()
    
    return fig, ax

def create_improvement_analysis_plot(df):
    """Create improvement analysis plot"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Calculate improvements over random selection
    vla_clip_improvement = ((df['random_selection_nrmse'] - df['vla_clip_nrmse']) / df['random_selection_nrmse'] * 100)
    monkey_improvement = ((df['random_selection_nrmse'] - df['monkey_verifier_nrmse']) / df['random_selection_nrmse'] * 100)
    
    # Plot improvements
    ax.plot(df['num_samples'], vla_clip_improvement, 
            label='VLA-CLIP vs Random', color='#1f77b4', 
            marker='o', markersize=8, linestyle='-', linewidth=2.5)
    
    ax.plot(df['num_samples'], monkey_improvement, 
            label='RoboMonkey vs Random', color='#d62728', 
            marker='s', markersize=8, linestyle='--', linewidth=2.5)
    
    # Formatting
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("NRMSE Improvement (%)")
    ax.set_xscale('log', base=2)
    ax.set_xticks(df['num_samples'])
    ax.set_xticklabels([str(int(x)) for x in df['num_samples']])
    
    # Grid and styling
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_color('black')
    
    # Add horizontal line at 0%
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Legend
    legend = ax.legend(loc='best', framealpha=0.95, edgecolor='black')
    legend.get_frame().set_linewidth(1.5)
    
    # Title
    ax.set_title('Verifier Improvement over Random Selection', fontsize=18, pad=20)
    
    plt.tight_layout()
    plt.savefig('gaussian_improvement_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig('gaussian_improvement_analysis.pdf', bbox_inches='tight')
    plt.show()
    
    return fig, ax

def print_statistics(df):
    """Print detailed statistics"""
    print("\n" + "="*80)
    print("GAUSSIAN VERIFIER ANALYSIS RESULTS")
    print("="*80)
    
    # Overall statistics
    print(f"\nDataset: 12,800 Gaussian-augmented samples (128 per original instruction)")
    print(f"Baseline (all samples): {df.iloc[-1]['random_selection_nrmse']:.4f} NRMSE")
    
    # Best performance for each method
    best_vla_clip = df['vla_clip_nrmse'].min()
    best_vla_clip_samples = df.loc[df['vla_clip_nrmse'].idxmin(), 'num_samples']
    
    best_monkey = df['monkey_verifier_nrmse'].min()
    best_monkey_samples = df.loc[df['monkey_verifier_nrmse'].idxmin(), 'num_samples']
    
    best_random = df['random_selection_nrmse'].min()
    best_random_samples = df.loc[df['random_selection_nrmse'].idxmin(), 'num_samples']
    
    print(f"\n--- BEST PERFORMANCE ---")
    print(f"VLA-CLIP Verifier:    {best_vla_clip:.4f} NRMSE (at {int(best_vla_clip_samples)} samples)")
    print(f"RoboMonkey Verifier:  {best_monkey:.4f} NRMSE (at {int(best_monkey_samples)} samples)")
    print(f"Random Selection:     {best_random:.4f} NRMSE (at {int(best_random_samples)} samples)")
    
    # Improvements
    vla_clip_vs_random = ((best_random - best_vla_clip) / best_random * 100)
    monkey_vs_random = ((best_random - best_monkey) / best_random * 100)
    vla_clip_vs_monkey = ((best_monkey - best_vla_clip) / best_monkey * 100)
    
    print(f"\n--- IMPROVEMENTS ---")
    print(f"VLA-CLIP vs Random:     {vla_clip_vs_random:+.2f}%")
    print(f"RoboMonkey vs Random:   {monkey_vs_random:+.2f}%")
    print(f"VLA-CLIP vs RoboMonkey: {vla_clip_vs_monkey:+.2f}%")
    
    # Performance at different sample counts
    print(f"\n--- PERFORMANCE BY SAMPLE COUNT ---")
    for _, row in df.iterrows():
        n_samples = int(row['num_samples'])
        vla_clip = row['vla_clip_nrmse']
        monkey = row['monkey_verifier_nrmse']
        random = row['random_selection_nrmse']
        
        vla_improvement = ((random - vla_clip) / random * 100)
        monkey_improvement = ((random - monkey) / random * 100)
        
        print(f"{n_samples:3d} samples: VLA-CLIP={vla_clip:.4f} (+{vla_improvement:.1f}%), "
              f"Monkey={monkey:.4f} (+{monkey_improvement:.1f}%), Random={random:.4f}")
    
    # Winner analysis
    print(f"\n--- WINNER ANALYSIS ---")
    vla_wins = (df['vla_clip_nrmse'] < df['monkey_verifier_nrmse']).sum()
    monkey_wins = (df['monkey_verifier_nrmse'] < df['vla_clip_nrmse']).sum()
    ties = (df['vla_clip_nrmse'] == df['monkey_verifier_nrmse']).sum()
    
    print(f"VLA-CLIP wins: {vla_wins}/{len(df)} sample counts")
    print(f"RoboMonkey wins: {monkey_wins}/{len(df)} sample counts") 
    print(f"Ties: {ties}/{len(df)} sample counts")
    
    if best_vla_clip < best_monkey:
        winner = "VLA-CLIP"
        winner_score = best_vla_clip
    else:
        winner = "RoboMonkey"
        winner_score = best_monkey
    
    print(f"\nOverall best performer: {winner} with {winner_score:.4f} NRMSE")

def main():
    """Main function to generate all plots and analysis"""
    print("Loading Gaussian threshold results...")
    df = load_data()
    
    print("Creating comprehensive comparison plot...")
    create_main_comparison_plot(df)
    
    print("Creating improvement analysis plot...")
    create_improvement_analysis_plot(df)
    
    print("Generating detailed statistics...")
    print_statistics(df)
    
    print(f"\nPlots saved:")
    print("  - gaussian_comprehensive_analysis.png/pdf")
    print("  - gaussian_improvement_analysis.png/pdf")

if __name__ == "__main__":
    main()
