#!/usr/bin/env python3
"""
Enhanced plot showing average NRMSE vs number of rephrased instructions,
comparing VLA-CLIP and monkey verifier selection methods.

X-axis: Number of rephrased instructions (1 to 128)
Y-axis: Average NRMSE across 100 datapoints
- Horizontal line at baseline for original instruction average NRMSE
- VLA-CLIP curve: select top N instructions by VLA-CLIP score
- Monkey Verifier curve: select top N instructions by monkey verifier score
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
import os
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

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

# Global BERT model and tokenizer for embedding computation
BERT_MODEL = None
BERT_TOKENIZER = None

def initialize_bert_model():
    """Initialize BERT model and tokenizer for embedding computation"""
    global BERT_MODEL, BERT_TOKENIZER
    if BERT_MODEL is None:
        print("Loading BERT model for embeddings...")
        model_name = 'bert-base-uncased'
        BERT_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
        BERT_MODEL = AutoModel.from_pretrained(model_name)
        BERT_MODEL.eval()
        print("BERT model loaded successfully")

def get_bert_embeddings(texts):
    """
    Compute BERT embeddings for a list of texts
    Returns: numpy array of shape (len(texts), embedding_dim)
    """
    initialize_bert_model()
    
    embeddings = []
    batch_size = 32  # Process in batches to avoid memory issues
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize the batch
            inputs = BERT_TOKENIZER(batch_texts, padding=True, truncation=True, 
                                  return_tensors='pt', max_length=512)
            
            # Get BERT outputs
            outputs = BERT_MODEL(**inputs)
            
            # Use [CLS] token embeddings (first token)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(batch_embeddings)
    
    return np.vstack(embeddings)

def load_vla_clip_data():
    """Load VLA-CLIP scores and organize by sample"""
    print("Loading VLA-CLIP scores...")
    with open('/root/validate_clip_verifier/bridge_vla_clip_scores_20250809_194343.json', 'r') as f:
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
    monkey_files = [f for f in os.listdir('/root/validate_clip_verifier/') 
                   if f.startswith('bridge_monkey_verifier_scores_') and f.endswith('.json')]
    if not monkey_files:
        raise FileNotFoundError("No monkey verifier scores file found!")
    monkey_file = sorted(monkey_files)[-1]  # Get the most recent one
    
    with open(f'/root/validate_clip_verifier/{monkey_file}', 'r') as f:
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

def calculate_threshold_nrmse_monkey(samples, num_instructions):
    """
    For each sample, select the top num_instructions by monkey verifier score
    and return the average NRMSE across all samples.
    """
    sample_nrmse_values = []
    
    for sample_id, results in samples.items():
        # Separate original and rephrased instructions
        original = [r for r in results if r['is_original']]
        rephrased = [r for r in results if not r['is_original'] and r['monkey_verifier_score'] is not None]
        
        if not original:
            continue
            
        if num_instructions == 1:
            # Only use original instruction
            selected = original
        else:
            # Select top (num_instructions - 1) rephrased instructions by monkey verifier score
            # Plus the original instruction
            rephrased_sorted = sorted(rephrased, key=lambda x: x['monkey_verifier_score'], reverse=True)
            top_rephrased = rephrased_sorted[:num_instructions-1]
            selected = original + top_rephrased
        
        # Calculate average NRMSE for this sample's selected instructions
        if selected:
            nrmse_values = [r['openvla_nrmse'] for r in selected]
            sample_avg_nrmse = np.mean(nrmse_values)
            sample_nrmse_values.append(sample_avg_nrmse)
    
    # Return average NRMSE across all samples
    return np.mean(sample_nrmse_values) if sample_nrmse_values else None

def calculate_threshold_nrmse_random(samples, num_instructions, random_seed=42):
    """
    For each sample, randomly select num_instructions (including original)
    and return the average NRMSE across all samples.
    """
    np.random.seed(random_seed)  # For reproducible results
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
            # Randomly select (num_instructions - 1) rephrased instructions
            # Plus the original instruction
            if len(rephrased) >= num_instructions - 1:
                random_rephrased = np.random.choice(rephrased, size=num_instructions-1, replace=False).tolist()
            else:
                random_rephrased = rephrased  # Use all available if not enough
            selected = original + random_rephrased
        
        # Calculate average NRMSE for this sample's selected instructions
        if selected:
            nrmse_values = [r['openvla_nrmse'] for r in selected]
            sample_avg_nrmse = np.mean(nrmse_values)
            sample_nrmse_values.append(sample_avg_nrmse)
    
    # Return average NRMSE across all samples
    return np.mean(sample_nrmse_values) if sample_nrmse_values else None

def calculate_threshold_nrmse_clustering(samples, num_instructions):
    """
    For each sample, use clustering-based selection:
    1. Start with n (threshold) rephrased instructions
    2. Compute BERT embeddings of rephrased language instructions
    3. Cluster with KMeans (cluster_num depends on the n threshold)
    4. Compute the average vla_clip_similarity_score of each cluster
    5. Only keep the highest scoring cluster
    6. Select the highest scoring instruction within the cluster
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
            # Step 1: Start with top n rephrased instructions by VLA-CLIP score
            rephrased_sorted = sorted(rephrased, key=lambda x: x['vla_clip_similarity_score'], reverse=True)
            top_rephrased = rephrased_sorted[:num_instructions-1]  # -1 to account for original
            
            if len(top_rephrased) < 2:
                # Not enough instructions to cluster, fall back to simple selection
                selected = original + top_rephrased
            else:
                # Step 2: Compute BERT embeddings
                instructions_text = [r['instruction'] for r in top_rephrased]
                embeddings = get_bert_embeddings(instructions_text)
                
                # Step 3: Determine number of clusters (adaptive based on threshold)
                # Use fewer clusters for smaller thresholds, more for larger
                if num_instructions <= 4:
                    n_clusters = 2
                elif num_instructions <= 16:h
                    n_clusters = min(3, len(top_rephrased))
                elif num_instructions <= 64:
                    n_clusters = min(5, len(top_rephrased))
                else:
                    n_clusters = min(8, len(top_rephrased))
                
                # Ensure we don't have more clusters than data points
                n_clusters = min(n_clusters, len(top_rephrased))
                
                if n_clusters >= 2:
                    # Perform KMeans clustering
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    cluster_labels = kmeans.fit_predict(embeddings)
                    
                    # Step 4: Compute average VLA-CLIP score for each cluster
                    cluster_scores = {}
                    cluster_instructions = {}
                    
                    for i, (instruction_data, cluster_id) in enumerate(zip(top_rephrased, cluster_labels)):
                        if cluster_id not in cluster_scores:
                            cluster_scores[cluster_id] = []
                            cluster_instructions[cluster_id] = []
                        
                        cluster_scores[cluster_id].append(instruction_data['vla_clip_similarity_score'])
                        cluster_instructions[cluster_id].append(instruction_data)
                    
                    # Calculate average scores for each cluster
                    cluster_avg_scores = {
                        cluster_id: np.mean(scores) 
                        for cluster_id, scores in cluster_scores.items()
                    }
                    
                    # Step 5: Select the highest scoring cluster
                    best_cluster_id = max(cluster_avg_scores.keys(), key=lambda x: cluster_avg_scores[x])
                    best_cluster_instructions = cluster_instructions[best_cluster_id]
                    
                    # Step 6: Select the highest scoring instruction within the cluster
                    best_instruction = max(best_cluster_instructions, 
                                         key=lambda x: x['vla_clip_similarity_score'])
                    
                    selected = original + [best_instruction]
                else:
                    # Fall back to simple selection if clustering not possible
                    selected = original + top_rephrased[:1]
        
        # Calculate average NRMSE for this sample's selected instructions
        if selected:
            nrmse_values = [r['openvla_nrmse'] for r in selected]
            sample_avg_nrmse = np.mean(nrmse_values)
            sample_nrmse_values.append(sample_avg_nrmse)
    
    # Return average NRMSE across all samples
    return np.mean(sample_nrmse_values) if sample_nrmse_values else None

def generate_enhanced_plot():
    """Generate the enhanced threshold analysis plot with both methods"""
    
    # Load data
    vla_clip_samples, original_avg_nrmse_clip = load_vla_clip_data()
    monkey_samples, original_avg_nrmse_monkey = load_monkey_verifier_data()
    
    # Use the original NRMSE from VLA-CLIP data (should be the same)
    original_avg_nrmse = original_avg_nrmse_clip
    
    # Calculate NRMSE for specific instruction thresholds
    num_instructions_range = [1, 2, 4, 8, 16, 32, 64, 128]
    vla_clip_nrmse_values = []
    monkey_nrmse_values = []
    random_nrmse_values = []
    clustering_nrmse_values = []
    
    print("Calculating average NRMSE for different instruction thresholds...")
    
    for num_instructions in num_instructions_range:
        # VLA-CLIP method
        vla_clip_nrmse = calculate_threshold_nrmse_vla_clip(vla_clip_samples, num_instructions)
        vla_clip_nrmse_values.append(vla_clip_nrmse)
        
        # Monkey verifier method
        monkey_nrmse = calculate_threshold_nrmse_monkey(monkey_samples, num_instructions)
        monkey_nrmse_values.append(monkey_nrmse)
        
        # Random selection method (use VLA-CLIP samples for data availability)
        random_nrmse = calculate_threshold_nrmse_random(vla_clip_samples, num_instructions)
        random_nrmse_values.append(random_nrmse)
        
        # Clustering-based method
        clustering_nrmse = calculate_threshold_nrmse_clustering(vla_clip_samples, num_instructions)
        clustering_nrmse_values.append(clustering_nrmse)
        
        print(f"  {num_instructions} instructions - VLA-CLIP: {vla_clip_nrmse:.4f}, Monkey: {monkey_nrmse:.4f}, Random: {random_nrmse:.4f}, Clustering: {clustering_nrmse:.4f}")
    
    # Create the plot with publication-quality styling
    fig, ax = plt.subplots()
    
    # Colors and markers following the example
    color_clip = '#1f77b4'
    color_monkey = '#d62728'
    color_random = '#ff7f0e'  # Orange
    color_clustering = '#2ca02c'  # Green
    marker_clip = 'o'
    marker_monkey = 's'
    marker_random = '^'  # Triangle
    marker_clustering = 'D'  # Diamond
    
    # Add greedy baselines
    ax.axhline(y=0.1665, color='black', linestyle='-', linewidth=2.0, 
               label='Original Instruction')
    ax.axhline(y=0.1793, color='gray', linestyle='-', linewidth=2.0, 
               label='Rephrased Instruction')
    
    # Plot VLA-CLIP curve
    ax.plot(num_instructions_range, vla_clip_nrmse_values, label='VLA-CLIP Verifier', 
            color=color_clip, marker=marker_clip, markersize=10, linestyle='-', linewidth=2.5)
    
    # Plot Monkey Verifier curve
    ax.plot(num_instructions_range, monkey_nrmse_values, label='RoboMonkey Verifier', 
            color=color_monkey, marker=marker_monkey, markersize=10, linestyle='--', linewidth=2.5)
    
    # Plot Random Selection curve
    ax.plot(num_instructions_range, random_nrmse_values, label='Random Selection', 
            color=color_random, marker=marker_random, markersize=10, linestyle=':', linewidth=2.5)
    
    # Plot Clustering-based curve
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
    
    # Legend
    legend = ax.legend(loc='lower right', framealpha=0.95, edgecolor='black', fontsize=11)
    legend.get_frame().set_linewidth(1.5)
    
    # Save plot
    output_file = 'enhanced_threshold_analysis.png'
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as: {output_file}")
    
    # Also save as PDF
    plt.savefig('enhanced_threshold_analysis.pdf', bbox_inches='tight')
    print(f"Plot also saved as: enhanced_threshold_analysis.pdf")
    
    plt.show()
    
    # Calculate statistics for reporting
    best_clip_nrmse = min([v for v in vla_clip_nrmse_values if v is not None])
    best_clip_idx = [v for v in vla_clip_nrmse_values if v is not None].index(best_clip_nrmse)
    best_clip_num = num_instructions_range[best_clip_idx]
    
    best_monkey_nrmse = min([v for v in monkey_nrmse_values if v is not None])
    best_monkey_idx = [v for v in monkey_nrmse_values if v is not None].index(best_monkey_nrmse)
    best_monkey_num = num_instructions_range[best_monkey_idx]
    
    best_random_nrmse = min([v for v in random_nrmse_values if v is not None])
    best_random_idx = [v for v in random_nrmse_values if v is not None].index(best_random_nrmse)
    best_random_num = num_instructions_range[best_random_idx]
    
    best_clustering_nrmse = min([v for v in clustering_nrmse_values if v is not None])
    best_clustering_idx = [v for v in clustering_nrmse_values if v is not None].index(best_clustering_nrmse)
    best_clustering_num = num_instructions_range[best_clustering_idx]
    
    # Print detailed comparison
    print(f"\n=== Detailed Comparison ===")
    print(f"Original instructions average NRMSE: {original_avg_nrmse:.4f}")
    print(f"\nVLA-CLIP Method:")
    print(f"  Best NRMSE: {best_clip_nrmse:.4f} (with {best_clip_num} instructions)")
    print(f"  Improvement: {((original_avg_nrmse - best_clip_nrmse) / original_avg_nrmse * 100):.1f}%")
    print(f"  NRMSE at 128 instructions: {vla_clip_nrmse_values[-1] if vla_clip_nrmse_values[-1] is not None else 'N/A'}")
    
    print(f"\nMonkey Verifier Method:")
    print(f"  Best NRMSE: {best_monkey_nrmse:.4f} (with {best_monkey_num} instructions)")
    print(f"  Improvement: {((original_avg_nrmse - best_monkey_nrmse) / original_avg_nrmse * 100):.1f}%")
    print(f"  NRMSE at 128 instructions: {monkey_nrmse_values[-1] if monkey_nrmse_values[-1] is not None else 'N/A'}")
    
    print(f"\nRandom Selection Method:")
    print(f"  Best NRMSE: {best_random_nrmse:.4f} (with {best_random_num} instructions)")
    print(f"  Improvement: {((original_avg_nrmse - best_random_nrmse) / original_avg_nrmse * 100):.1f}%")
    print(f"  NRMSE at 128 instructions: {random_nrmse_values[-1] if random_nrmse_values[-1] is not None else 'N/A'}")
    
    print(f"\nClustering + VLA-CLIP Method:")
    print(f"  Best NRMSE: {best_clustering_nrmse:.4f} (with {best_clustering_num} instructions)")
    print(f"  Improvement: {((original_avg_nrmse - best_clustering_nrmse) / original_avg_nrmse * 100):.1f}%")
    print(f"  NRMSE at 128 instructions: {clustering_nrmse_values[-1] if clustering_nrmse_values[-1] is not None else 'N/A'}")
    
    # Determine which method is best
    all_best_scores = [
        ("VLA-CLIP", best_clip_nrmse),
        ("Monkey Verifier", best_monkey_nrmse),
        ("Random Selection", best_random_nrmse),
        ("Clustering + VLA-CLIP", best_clustering_nrmse)
    ]
    winner_name, winner_score = min(all_best_scores, key=lambda x: x[1])
    
    print(f"\nBest performing method: {winner_name} with NRMSE {winner_score:.4f}")
    
    # Compare verifier methods against random
    clip_vs_random = ((best_random_nrmse - best_clip_nrmse) / best_random_nrmse * 100)
    monkey_vs_random = ((best_random_nrmse - best_monkey_nrmse) / best_random_nrmse * 100)
    clustering_vs_random = ((best_random_nrmse - best_clustering_nrmse) / best_random_nrmse * 100)
    
    print(f"VLA-CLIP vs Random: {clip_vs_random:.1f}% better")
    print(f"Monkey vs Random: {monkey_vs_random:.1f}% better")
    print(f"Clustering vs Random: {clustering_vs_random:.1f}% better")
    
    # Compare clustering against other methods
    clustering_vs_clip = ((best_clip_nrmse - best_clustering_nrmse) / best_clip_nrmse * 100)
    clustering_vs_monkey = ((best_monkey_nrmse - best_clustering_nrmse) / best_monkey_nrmse * 100)
    
    print(f"Clustering vs VLA-CLIP: {clustering_vs_clip:.1f}% better")
    print(f"Clustering vs Monkey: {clustering_vs_monkey:.1f}% better")
    
    # Save numerical results
    results_df = pd.DataFrame({
        'num_instructions': num_instructions_range,
        'vla_clip_nrmse': vla_clip_nrmse_values,
        'monkey_verifier_nrmse': monkey_nrmse_values,
        'random_selection_nrmse': random_nrmse_values,
        'clustering_vla_clip_nrmse': clustering_nrmse_values
    })
    results_df.to_csv('enhanced_threshold_results.csv', index=False)
    print(f"Numerical results saved as: enhanced_threshold_results.csv")
    
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
        '/root/validate_clip_verifier/bridge_vla_clip_scores_20250809_194343.json'
    ]
    
    # Check for monkey verifier files
    monkey_files = [f for f in os.listdir('/root/validate_clip_verifier/') 
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
    
    # Generate the enhanced plot
    results = generate_enhanced_plot()
