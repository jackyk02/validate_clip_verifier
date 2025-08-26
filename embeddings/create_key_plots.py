#!/usr/bin/env python3
"""
Script to create key plots from the instruction embedding analysis results.

This script reads the analysis results CSV and creates:
1. Scatter plot of embedding similarity vs action error (with trend line)
2. Binned analysis showing mean action error by similarity quartiles
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import warnings
warnings.filterwarnings('ignore')

def create_key_plots(csv_file, output_prefix="key_plots"):
    """Create the first and last plots from the analysis."""
    
    # Load the data
    print(f"Loading data from {csv_file}")
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} samples")
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with 2 subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Key Results: Instruction Embedding Similarity vs Action Error', fontsize=16)
    
    # Plot 1: Scatter plot with trend line
    axes[0].scatter(df['embedding_similarity'], df['openvla_nrmse'], alpha=0.6, s=20, color='steelblue')
    axes[0].set_xlabel('Embedding Similarity (Cosine)', fontsize=12)
    axes[0].set_ylabel('OpenVLA NRMSE (Action Error)', fontsize=12)
    axes[0].set_title('Embedding Similarity vs Action Error', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(df['embedding_similarity'], df['openvla_nrmse'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(df['embedding_similarity'].min(), df['embedding_similarity'].max(), 100)
    axes[0].plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label=f'Trend line (slope={z[0]:.4f})')
    axes[0].legend()
    
    # Calculate correlation for summary stats (but don't display on plot)
    from scipy.stats import pearsonr
    corr, p_val = pearsonr(df['embedding_similarity'], df['openvla_nrmse'])
    
    # Plot 2: Binned analysis
    # Create bins based on similarity quartiles
    df['similarity_bin'] = pd.qcut(df['embedding_similarity'], q=4, labels=['Q1\n(Lowest)', 'Q2', 'Q3', 'Q4\n(Highest)'])
    bin_stats = df.groupby('similarity_bin')['openvla_nrmse'].agg(['mean', 'std', 'count'])
    
    # Create bar plot
    bars = axes[1].bar(range(len(bin_stats)), bin_stats['mean'], 
                      alpha=0.7, color=['#ff7f7f', '#7fbf7f', '#7f7fff', '#bf7fff'])
    axes[1].set_xlabel('Embedding Similarity Quartiles', fontsize=12)
    axes[1].set_ylabel('Mean OpenVLA NRMSE (Action Error)', fontsize=12)
    axes[1].set_title('Mean Action Error by Similarity Quartile', fontsize=14)
    axes[1].set_xticks(range(len(bin_stats)))
    axes[1].set_xticklabels(bin_stats.index)
    axes[1].grid(True, alpha=0.3, axis='y')
    

    
    plt.tight_layout()
    
    # Save the plot
    output_file = f'{output_prefix}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Key plots saved as {output_file}")
    
    # Print summary statistics
    print(f"\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total samples: {len(df):,}")
    print(f"Embedding similarity: {df['embedding_similarity'].mean():.4f} ± {df['embedding_similarity'].std():.4f}")
    print(f"Action error (NRMSE): {df['openvla_nrmse'].mean():.4f} ± {df['openvla_nrmse'].std():.4f}")
    print(f"Correlation: {corr:.4f} (p-value: {p_val:.2e})")
    
    print(f"\nBinned Analysis:")
    print("-" * 50)
    for i, (quartile, stats) in enumerate(bin_stats.iterrows()):
        similarity_range = df[df['similarity_bin'] == quartile]['embedding_similarity']
        print(f"{quartile}: Similarity {similarity_range.min():.3f}-{similarity_range.max():.3f}, "
              f"Mean Error: {stats['mean']:.4f} ± {stats['std']:.4f} (n={stats['count']})")
    
    # Calculate effect size
    error_range = bin_stats['mean'].max() - bin_stats['mean'].min()
    print(f"\nEffect size: {error_range:.4f} NRMSE difference between highest and lowest similarity quartiles")
    print(f"Relative improvement: {(error_range/bin_stats['mean'].max())*100:.1f}% lower error for highest similarity")
    
    return fig, bin_stats

def main():
    parser = argparse.ArgumentParser(description='Create key plots from instruction embedding analysis results')
    parser.add_argument('csv_file', help='Path to the analysis results CSV file')
    parser.add_argument('--output', default='key_plots', help='Output file prefix (default: key_plots)')
    
    args = parser.parse_args()
    
    # Create the plots
    fig, bin_stats = create_key_plots(args.csv_file, args.output)
    
    print(f"\nPlots saved successfully!")
    print(f"File: {args.output}.png")

if __name__ == "__main__":
    main()
