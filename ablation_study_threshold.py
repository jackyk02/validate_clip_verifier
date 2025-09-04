#!/usr/bin/env python3
"""
Threshold-based ablation study comparing different approaches for selecting rephrases.

This script compares different selection methods across various thresholds (1, 2, 4, 8, 16, 32)
and shows the oracle action error (best possible performance) for each method at each threshold.

Methods compared:
1. K-means clustering based on generated actions from OpenVLA
2. K-means clustering based on semantic embeddings from BERT
3. Closest semantic embeddings to original instruction
4. First N rephrases (baseline approach)
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set up matplotlib for publication-quality figures
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

class ThresholdAblationStudy:
    def __init__(self, bert_model='bert-base-uncased'):
        """Initialize the ablation study with BERT model for embeddings."""
        print(f"Loading BERT model: {bert_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)
        self.model = AutoModel.from_pretrained(bert_model)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded on device: {self.device}")
        
    def load_data(self, json_file):
        """Load OpenVLA actions data and organize by sample."""
        print(f"Loading data from {json_file}")
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        print(f"Loaded {len(data['results'])} OpenVLA action results")
        
        # Organize data by sample_id
        samples = defaultdict(list)
        for result in data['results']:
            sample_id = result['sample_id']
            samples[sample_id].append(result)
        
        print(f"Found {len(samples)} unique samples")
        return samples
    
    def get_bert_embeddings(self, texts, batch_size=32):
        """Get BERT embeddings for a list of texts."""
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Computing BERT embeddings", leave=False):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(batch_texts, return_tensors='pt', truncation=True,
                                   padding=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use [CLS] token embeddings
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.extend(batch_embeddings)
        
        return np.array(embeddings)
    
    def select_rephrases_action_clustering(self, sample_results, n_rephrases):
        """Method 1: K-means clustering based on generated actions from OpenVLA."""
        # Separate original and rephrased instructions
        original = [r for r in sample_results if r['is_original']]
        rephrased = [r for r in sample_results if not r['is_original']]
        
        if len(rephrased) <= n_rephrases:
            return original + rephrased
        
        if n_rephrases == 0:
            return original
        
        # Extract generated actions (7-dimensional vectors)
        actions = np.array([r['generated_action'] for r in rephrased])
        
        # Perform k-means clustering
        kmeans = KMeans(n_clusters=n_rephrases, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(actions)
        
        # Select one representative from each cluster (closest to centroid)
        selected_rephrases = []
        for cluster_id in range(n_rephrases):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            if len(cluster_indices) > 0:
                # Find the point closest to the centroid
                centroid = kmeans.cluster_centers_[cluster_id]
                distances = np.linalg.norm(actions[cluster_indices] - centroid, axis=1)
                closest_idx = cluster_indices[np.argmin(distances)]
                selected_rephrases.append(rephrased[closest_idx])
        
        return original + selected_rephrases
    
    def select_rephrases_embedding_clustering(self, sample_results, n_rephrases):
        """Method 2: K-means clustering based on BERT semantic embeddings."""
        # Separate original and rephrased instructions
        original = [r for r in sample_results if r['is_original']]
        rephrased = [r for r in sample_results if not r['is_original']]
        
        if len(rephrased) <= n_rephrases:
            return original + rephrased
        
        if n_rephrases == 0:
            return original
        
        # Get BERT embeddings for rephrased instructions
        instructions = [r['instruction'] for r in rephrased]
        embeddings = self.get_bert_embeddings(instructions)
        
        # Perform k-means clustering
        kmeans = KMeans(n_clusters=n_rephrases, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Select one representative from each cluster (closest to centroid)
        selected_rephrases = []
        for cluster_id in range(n_rephrases):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            if len(cluster_indices) > 0:
                # Find the point closest to the centroid
                centroid = kmeans.cluster_centers_[cluster_id]
                distances = np.linalg.norm(embeddings[cluster_indices] - centroid, axis=1)
                closest_idx = cluster_indices[np.argmin(distances)]
                selected_rephrases.append(rephrased[closest_idx])
        
        return original + selected_rephrases
    
    def select_rephrases_closest_semantic(self, sample_results, n_rephrases):
        """Method 3: Select N rephrases closest to original instruction in semantic embedding space."""
        # Separate original and rephrased instructions
        original = [r for r in sample_results if r['is_original']]
        rephrased = [r for r in sample_results if not r['is_original']]
        
        if len(rephrased) <= n_rephrases:
            return original + rephrased
        
        if n_rephrases == 0:
            return original
        
        # Get original instruction embedding
        original_instruction = original[0]['original_instruction']
        original_embedding = self.get_bert_embeddings([original_instruction])[0]
        
        # Get embeddings for all rephrased instructions
        rephrased_instructions = [r['instruction'] for r in rephrased]
        rephrased_embeddings = self.get_bert_embeddings(rephrased_instructions)
        
        # Calculate cosine similarities to original
        similarities = cosine_similarity([original_embedding], rephrased_embeddings)[0]
        
        # Select top n_rephrases most similar
        top_indices = np.argsort(similarities)[-n_rephrases:]
        selected_rephrases = [rephrased[i] for i in top_indices]
        
        return original + selected_rephrases
    
    def select_rephrases_first_n(self, sample_results, n_rephrases):
        """Method 4: Select first N rephrases (baseline approach)."""
        # Separate original and rephrased instructions
        original = [r for r in sample_results if r['is_original']]
        rephrased = [r for r in sample_results if not r['is_original']]
        
        # Select first n_rephrases
        selected_rephrases = rephrased[:n_rephrases]
        
        return original + selected_rephrases
    
    def calculate_oracle_nrmse(self, selected_results):
        """Calculate oracle NRMSE (best possible) for selected results."""
        if not selected_results:
            return None
        
        nrmse_values = [r['nrmse'] for r in selected_results]
        return np.min(nrmse_values)
    
    def run_threshold_ablation_study(self, samples, thresholds=[1, 2, 4, 8, 16, 32], n_samples=None):
        """Run the threshold-based ablation study."""
        print("Running threshold-based ablation study...")
        
        # Limit samples for faster computation if specified
        if n_samples:
            sample_ids = list(samples.keys())[:n_samples]
            samples = {k: samples[k] for k in sample_ids}
            print(f"Using {len(samples)} samples for analysis")
        
        methods = {
            'action_clustering': self.select_rephrases_action_clustering,
            'embedding_clustering': self.select_rephrases_embedding_clustering,
            'closest_semantic': self.select_rephrases_closest_semantic,
            'first_n': self.select_rephrases_first_n
        }
        
        results = {}
        for method_name in methods.keys():
            results[method_name] = {threshold: [] for threshold in thresholds}
        
        for sample_id, sample_results in tqdm(samples.items(), desc="Processing samples"):
            for threshold in thresholds:
                n_rephrases = threshold - 1 if threshold > 1 else 0  # -1 because we always include original
                
                for method_name, method_func in methods.items():
                    selected = method_func(sample_results, n_rephrases)
                    oracle_nrmse = self.calculate_oracle_nrmse(selected)
                    results[method_name][threshold].append(oracle_nrmse)
        
        return results, thresholds
    
    def analyze_threshold_results(self, results, thresholds):
        """Analyze and print threshold results."""
        print("\n" + "="*80)
        print("THRESHOLD-BASED ABLATION STUDY RESULTS")
        print("="*80)
        
        method_names = {
            'action_clustering': 'Action K-means Clustering',
            'embedding_clustering': 'BERT Embedding K-means Clustering', 
            'closest_semantic': 'Closest Semantic Embeddings',
            'first_n': 'First N Rephrases (Baseline)'
        }
        
        # Calculate summary statistics
        summary_stats = {}
        for method, name in method_names.items():
            summary_stats[method] = {
                'name': name,
                'thresholds': thresholds,
                'oracle_means': [],
                'oracle_stds': []
            }
            
            print(f"\n{name}:")
            for threshold in thresholds:
                oracle_values = np.array([x for x in results[method][threshold] if x is not None])
                mean_oracle = np.mean(oracle_values)
                std_oracle = np.std(oracle_values)
                
                summary_stats[method]['oracle_means'].append(mean_oracle)
                summary_stats[method]['oracle_stds'].append(std_oracle)
                
                print(f"  {threshold:2d} instructions: Oracle NRMSE = {mean_oracle:.4f} ± {std_oracle:.4f}")
        
        return summary_stats
    
    def create_threshold_visualization(self, summary_stats, thresholds):
        """Create visualization of the threshold-based ablation study results."""
        print("\nCreating threshold visualization...")
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Colors and markers for each method
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        markers = ['o', 's', '^', 'D']
        linestyles = ['-', '--', '-.', ':']
        
        method_order = ['action_clustering', 'embedding_clustering', 'closest_semantic', 'first_n']
        
        for i, method in enumerate(method_order):
            stats = summary_stats[method]
            oracle_means = stats['oracle_means']
            oracle_stds = stats['oracle_stds']
            
            # Plot the line without error bars
            ax.plot(thresholds, oracle_means,
                   label=stats['name'], color=colors[i], marker=markers[i],
                   markersize=10, linestyle=linestyles[i], linewidth=3.0)
        
        # Customize the plot
        ax.set_xlabel('Number of Instructions (Threshold)')
        ax.set_ylabel('Oracle Action Error (Best Possible NRMSE)')
        ax.set_title('Ablation Study: Oracle Performance vs Selection Method\nAcross Different Instruction Thresholds')
        
        # Set x-axis to log scale and customize ticks
        ax.set_xscale('log', base=2)
        ax.set_xticks(thresholds)
        ax.set_xticklabels([str(t) for t in thresholds])
        
        # Grid and styling
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
        
        # Legend
        ax.legend(loc='upper right', framealpha=0.95, edgecolor='black')
        
        # Set y-axis limits to better show differences
        all_means = []
        for method in method_order:
            all_means.extend(summary_stats[method]['oracle_means'])
        y_min = min(all_means) * 0.9
        y_max = max(all_means) * 1.1
        ax.set_ylim(y_min, y_max)
        
        plt.tight_layout()
        plt.savefig('threshold_ablation_study.png', dpi=300, bbox_inches='tight')
        plt.savefig('threshold_ablation_study.pdf', bbox_inches='tight')
        print("Threshold visualization saved as threshold_ablation_study.png and .pdf")
        
        return fig
    
    def save_threshold_results(self, results, summary_stats, thresholds):
        """Save detailed threshold results to files."""
        # Save summary statistics
        summary_data = []
        for method, stats in summary_stats.items():
            for i, threshold in enumerate(thresholds):
                summary_data.append({
                    'method': method,
                    'method_name': stats['name'],
                    'threshold': threshold,
                    'oracle_mean': stats['oracle_means'][i],
                    'oracle_std': stats['oracle_stds'][i]
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv('threshold_ablation_summary.csv', index=False)
        print("Summary statistics saved to threshold_ablation_summary.csv")
        
        # Save detailed results
        detailed_data = []
        methods = list(results.keys())
        n_samples = len(results[methods[0]][thresholds[0]])
        
        for sample_idx in range(n_samples):
            row = {'sample_id': sample_idx}
            for method in methods:
                for threshold in thresholds:
                    col_name = f'{method}_threshold_{threshold}'
                    row[col_name] = results[method][threshold][sample_idx]
            detailed_data.append(row)
        
        detailed_df = pd.DataFrame(detailed_data)
        detailed_df.to_csv('threshold_ablation_detailed.csv', index=False)
        print("Detailed results saved to threshold_ablation_detailed.csv")
        
        # Print best performing method at each threshold
        print(f"\nBest performing method at each threshold:")
        for threshold in thresholds:
            best_method = None
            best_score = float('inf')
            for method, stats in summary_stats.items():
                threshold_idx = thresholds.index(threshold)
                score = stats['oracle_means'][threshold_idx]
                if score < best_score:
                    best_score = score
                    best_method = stats['name']
            print(f"  {threshold:2d} instructions: {best_method} (Oracle NRMSE: {best_score:.4f})")

def main():
    # Initialize threshold ablation study
    study = ThresholdAblationStudy()
    
    # Load data
    required_file = '/root/validate_clip_verifier/bridge_openvla_actions_20250809_192948.json'
    samples = study.load_data(required_file)
    
    # Define thresholds to test
    thresholds = [1, 2, 4, 8, 16, 32]
    
    # Run threshold ablation study (use subset for faster computation during development)
    results, thresholds = study.run_threshold_ablation_study(samples, thresholds, n_samples=100)
    
    # Analyze results
    summary_stats = study.analyze_threshold_results(results, thresholds)
    
    # Create visualizations
    study.create_threshold_visualization(summary_stats, thresholds)
    
    # Save results
    study.save_threshold_results(results, summary_stats, thresholds)
    
    print(f"\nThreshold ablation study complete! Generated files:")
    print("- threshold_ablation_study.png")
    print("- threshold_ablation_study.pdf") 
    print("- threshold_ablation_summary.csv")
    print("- threshold_ablation_detailed.csv")

if __name__ == "__main__":
    main()
