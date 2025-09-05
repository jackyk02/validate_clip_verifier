#!/usr/bin/env python3
"""
Threshold-based ablation study comparing different approaches for selecting samples from 128 available samples.

This script compares different selection methods across various thresholds (1, 2, 4, 8, 16, 32)
representing the number of samples to select from the total 128 available samples.
It shows the oracle action error (best possible performance) for each method at each threshold.

Methods compared:
1. K-means clustering based on Qwen3 semantic embeddings
2. Closest semantic embeddings to original instruction
3. First N samples (baseline approach)
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import warnings
import pickle
import os
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
    def __init__(self, model_name='Qwen/Qwen3-Embedding-4B'):
        """Initialize the ablation study with Qwen3 model for embeddings."""
        print(f"Loading Qwen3 embedding model: {model_name}")
        # Try to load on GPU first, fallback to CPU if out of memory
        try:
            # Load model on GPU without flash attention
            print("Attempting to load on GPU...")
            self.model = SentenceTransformer(model_name, device='cuda')
            print(f"Qwen3 model loaded on GPU")
        except Exception as e:
            print(f"Failed to load on GPU: {e}")
            try:
                # Try loading on CPU
                print("Attempting to load on CPU...")
                self.model = SentenceTransformer(model_name, device='cpu')
                print(f"Qwen3 model loaded on CPU")
            except Exception as e2:
                print(f"Failed to load Qwen3 model: {e2}")
                print("Falling back to a smaller embedding model...")
                # Fallback to a smaller model that should work
                self.model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
                model_name = 'all-MiniLM-L6-v2'
                print(f"Loaded fallback model: {model_name}")
        
        self.model_name = model_name
        print(f"Final model loaded: {self.model_name}")
        
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
    
    def get_qwen3_embeddings(self, texts, batch_size=100):
        """Get Qwen3 embeddings for a list of texts."""
        # Use sentence-transformers to encode texts
        # For documents, we don't need a special prompt
        embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=True)
        return embeddings
    
    def precompute_all_embeddings(self, samples):
        """Precompute embeddings for all instructions in all samples."""
        print("Precomputing embeddings for all instructions...")
        
        # Collect all unique instructions
        all_instructions = []
        instruction_to_idx = {}
        
        for sample_id, sample_results in tqdm(samples.items(), desc="Collecting instructions"):
            for result in sample_results:
                instruction = result['instruction']
                if instruction not in instruction_to_idx:
                    instruction_to_idx[instruction] = len(all_instructions)
                    all_instructions.append(instruction)
                    
                # Also add original instruction if it exists and is different
                if 'original_instruction' in result and result['original_instruction'] not in instruction_to_idx:
                    orig_instruction = result['original_instruction']
                    instruction_to_idx[orig_instruction] = len(all_instructions)
                    all_instructions.append(orig_instruction)
        
        print(f"Found {len(all_instructions)} unique instructions")
        
        # Compute embeddings for all instructions
        print("Computing embeddings...")
        all_embeddings = self.get_qwen3_embeddings(all_instructions, batch_size=100)
        
        # Create embedding cache
        self.embedding_cache = {}
        for instruction, idx in instruction_to_idx.items():
            self.embedding_cache[instruction] = all_embeddings[idx]
        
        print(f"Precomputed embeddings for {len(self.embedding_cache)} instructions")
        return self.embedding_cache
    
    def save_embeddings_cache(self, cache_file='embeddings_cache.pkl'):
        """Save precomputed embeddings to disk."""
        if not hasattr(self, 'embedding_cache'):
            raise ValueError("No embeddings cache to save. Call precompute_all_embeddings first.")
        
        print(f"Saving embeddings cache to {cache_file}...")
        with open(cache_file, 'wb') as f:
            pickle.dump(self.embedding_cache, f)
        print(f"Saved {len(self.embedding_cache)} embeddings to cache file")
    
    def load_embeddings_cache(self, cache_file='embeddings_cache.pkl'):
        """Load precomputed embeddings from disk."""
        if os.path.exists(cache_file):
            print(f"Loading embeddings cache from {cache_file}...")
            with open(cache_file, 'rb') as f:
                self.embedding_cache = pickle.load(f)
            print(f"Loaded {len(self.embedding_cache)} embeddings from cache file")
            return True
        else:
            print(f"Cache file {cache_file} not found")
            return False
    
    def get_cached_embedding(self, instruction):
        """Get precomputed embedding for an instruction."""
        if not hasattr(self, 'embedding_cache'):
            raise ValueError("Embeddings not precomputed. Call precompute_all_embeddings first.")
        return self.embedding_cache[instruction]
    
    def get_cached_embeddings(self, instructions):
        """Get precomputed embeddings for a list of instructions."""
        if not hasattr(self, 'embedding_cache'):
            raise ValueError("Embeddings not precomputed. Call precompute_all_embeddings first.")
        return np.array([self.embedding_cache[instruction] for instruction in instructions])
    
    
    def select_rephrases_embedding_clustering(self, sample_results, n_total_samples):
        """Method 2: K-means clustering based on Qwen3 semantic embeddings."""
        # Separate original and rephrased instructions
        original = [r for r in sample_results if r['is_original']]
        rephrased = [r for r in sample_results if not r['is_original']]
        
        # If we want fewer samples than available, always include original first
        if n_total_samples <= 1:
            return original[:1]  # Just return the original
        
        # Calculate how many rephrases to select (total - 1 for original)
        n_rephrases = n_total_samples - 1
        
        if len(rephrased) <= n_rephrases:
            return original + rephrased
        
        if n_rephrases == 0:
            return original
        
        # Get precomputed embeddings for rephrased instructions
        instructions = [r['instruction'] for r in rephrased]
        embeddings = self.get_cached_embeddings(instructions)
        
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
    
    def select_rephrases_closest_semantic(self, sample_results, n_total_samples):
        """Method 3: Select N rephrases closest to original instruction in semantic embedding space."""
        # Separate original and rephrased instructions
        original = [r for r in sample_results if r['is_original']]
        rephrased = [r for r in sample_results if not r['is_original']]
        
        # If we want fewer samples than available, always include original first
        if n_total_samples <= 1:
            return original[:1]  # Just return the original
        
        # Calculate how many rephrases to select (total - 1 for original)
        n_rephrases = n_total_samples - 1
        
        if len(rephrased) <= n_rephrases:
            return original + rephrased
        
        if n_rephrases == 0:
            return original
        
        # Get precomputed original instruction embedding
        original_instruction = original[0]['original_instruction']
        original_embedding = self.get_cached_embedding(original_instruction)
        
        # Get precomputed embeddings for all rephrased instructions
        rephrased_instructions = [r['instruction'] for r in rephrased]
        rephrased_embeddings = self.get_cached_embeddings(rephrased_instructions)
        
        # Calculate cosine similarities to original using sentence-transformers similarity
        similarities = self.model.similarity([original_embedding], rephrased_embeddings)[0].cpu().numpy()
        
        # Select top n_rephrases most similar
        top_indices = np.argsort(similarities)[-n_rephrases:]
        selected_rephrases = [rephrased[i] for i in top_indices]
        
        return original + selected_rephrases
    
    def select_rephrases_first_n(self, sample_results, n_total_samples):
        """Method 4: Select first N rephrases (baseline approach)."""
        # Separate original and rephrased instructions
        original = [r for r in sample_results if r['is_original']]
        rephrased = [r for r in sample_results if not r['is_original']]
        
        # If we want fewer samples than available, always include original first
        if n_total_samples <= 1:
            return original[:1]  # Just return the original
        
        # Calculate how many rephrases to select (total - 1 for original)
        n_rephrases = n_total_samples - 1
        
        # Select first n_rephrases
        selected_rephrases = rephrased[:n_rephrases]
        
        return original + selected_rephrases
    
    def calculate_oracle_nrmse(self, selected_results):
        """Calculate oracle NRMSE (best possible) for selected results."""
        if not selected_results:
            return None
        
        nrmse_values = [r['nrmse'] for r in selected_results]
        return np.min(nrmse_values)
    
    def run_threshold_ablation_study(self, samples, thresholds=[1, 2, 4, 8, 16, 32], n_samples=None, print_selections=False):
        """Run the threshold-based ablation study."""
        print("Running threshold-based ablation study...")
        
        # Limit samples for faster computation if specified
        if n_samples:
            sample_ids = list(samples.keys())[:n_samples]
            samples = {k: samples[k] for k in sample_ids}
            print(f"Using {len(samples)} samples for analysis")
        
        methods = {
            'embedding_clustering': self.select_rephrases_embedding_clustering,
            'closest_semantic': self.select_rephrases_closest_semantic,
            'first_n': self.select_rephrases_first_n
        }
        
        method_names = {
            'embedding_clustering': 'Qwen3 Embedding K-means Clustering',
            'closest_semantic': 'Closest Semantic Embeddings',
            'first_n': 'First N Rephrases (Baseline)'
        }
        
        results = {}
        for method_name in methods.keys():
            results[method_name] = {threshold: [] for threshold in thresholds}
        
        # Print selections for first few samples if requested
        samples_to_print = 3 if print_selections else 0
        sample_count = 0
        
        for sample_id, sample_results in tqdm(samples.items(), desc="Processing samples"):
            for threshold in thresholds:
                # threshold represents total number of samples to select (including original)
                n_total_samples = threshold
                
                for method_name, method_func in methods.items():
                    selected = method_func(sample_results, n_total_samples)
                    oracle_nrmse = self.calculate_oracle_nrmse(selected)
                    results[method_name][threshold].append(oracle_nrmse)
                    
                    # Print selections for first few samples
                    if print_selections and sample_count < samples_to_print:
                        if method_name == 'embedding_clustering':  # Print once per threshold
                            print(f"\n--- Sample {sample_id}, Threshold {threshold} ---")
                        
                        print(f"\n{method_names[method_name]}:")
                        for i, result in enumerate(selected):
                            marker = "[ORIGINAL]" if result['is_original'] else "[REPHRASE]"
                            print(f"  {i+1}. {marker} {result['instruction'][:80]}{'...' if len(result['instruction']) > 80 else ''} (NRMSE: {result['nrmse']:.4f})")
                        print(f"  → Oracle NRMSE: {oracle_nrmse:.4f}")
            
            sample_count += 1
        
        return results, thresholds
    
    def print_selection_examples(self, samples, thresholds=[1, 2, 4, 8, 16, 32], n_examples=3):
        """Print selection examples for the first few samples to understand method behavior."""
        print("="*80)
        print("SELECTION EXAMPLES")
        print("="*80)
        
        methods = {
            'embedding_clustering': self.select_rephrases_embedding_clustering,
            'closest_semantic': self.select_rephrases_closest_semantic,
            'first_n': self.select_rephrases_first_n
        }
        
        method_names = {
            'embedding_clustering': 'Qwen3 Embedding K-means Clustering',
            'closest_semantic': 'Closest Semantic Embeddings',
            'first_n': 'First N Rephrases (Baseline)'
        }
        
        sample_ids = list(samples.keys())[:n_examples]
        
        for sample_id in sample_ids:
            sample_results = samples[sample_id]
            print(f"\n{'='*60}")
            print(f"SAMPLE: {sample_id}")
            print(f"{'='*60}")
            print(f"Total available instructions: {len(sample_results)}")
            
            # Show original instruction
            original = [r for r in sample_results if r['is_original']][0]
            print(f"Original: {original['instruction']} (NRMSE: {original['nrmse']:.4f})")
            
            for threshold in thresholds:
                print(f"\n--- THRESHOLD {threshold} (Select {threshold} total samples) ---")
                
                for method_name, method_func in methods.items():
                    selected = method_func(sample_results, threshold)
                    oracle_nrmse = self.calculate_oracle_nrmse(selected)
                    
                    print(f"\n{method_names[method_name]}:")
                    for i, result in enumerate(selected):
                        marker = "[ORIGINAL]" if result['is_original'] else "[REPHRASE]"
                        print(f"  {i+1}. {marker} {result['instruction'][:100]}{'...' if len(result['instruction']) > 100 else ''}")
                        print(f"      NRMSE: {result['nrmse']:.4f}")
                    print(f"  → Oracle NRMSE: {oracle_nrmse:.4f} (best among selected)")
            
            print("\n" + "-"*60)
    
    def analyze_threshold_results(self, results, thresholds):
        """Analyze and print threshold results."""
        print("\n" + "="*80)
        print("THRESHOLD-BASED ABLATION STUDY RESULTS")
        print("="*80)
        
        method_names = {
            'embedding_clustering': 'Qwen3 Embedding K-means Clustering', 
            'closest_semantic': 'Top-N Closest Semantic Embeddings',
            'first_n': 'First-N GPT-4o Rephrases'
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
                
                print(f"  {threshold:2d} samples: Oracle NRMSE = {mean_oracle:.4f} ± {std_oracle:.4f}")
        
        return summary_stats
    
    def create_threshold_visualization(self, summary_stats, thresholds):
        """Create visualization of the threshold-based ablation study results."""
        print("\nCreating threshold visualization...")
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Colors and markers for each method
        colors = ['#ff7f0e', '#2ca02c', '#d62728']
        markers = ['s', '^', 'D']
        linestyles = ['--', '-.', ':']
        
        method_order = ['embedding_clustering', 'closest_semantic', 'first_n']
        
        for i, method in enumerate(method_order):
            stats = summary_stats[method]
            oracle_means = stats['oracle_means']
            oracle_stds = stats['oracle_stds']
            
            # Plot the line without error bars
            ax.plot(thresholds, oracle_means,
                   label=stats['name'], color=colors[i], marker=markers[i],
                   markersize=10, linestyle=linestyles[i], linewidth=3.0)
        
        # Customize the plot
        ax.set_xlabel('Number of Selected Samples')
        ax.set_ylabel('Oracle Action Error (Average NRMSE)')
        # ax.set_title('Ablation Study: Oracle Performance vs Selection Method\nAcross Different Instruction Thresholds')
        
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
            print(f"  {threshold:2d} samples: {best_method} (Oracle NRMSE: {best_score:.4f})")

def main():
    # Initialize threshold ablation study
    study = ThresholdAblationStudy()
    
    # Load data
    required_file = '/root/validate_clip_verifier/bridge_openvla_actions_20250809_192948.json'
    samples = study.load_data(required_file)
    
    # Try to load precomputed embeddings, otherwise compute them
    cache_file = 'qwen3_embeddings_cache.pkl'
    if not study.load_embeddings_cache(cache_file):
        print("Computing embeddings for the first time...")
        study.precompute_all_embeddings(samples)
        study.save_embeddings_cache(cache_file)
    else:
        print("Using cached embeddings")
    
    # Define thresholds to test
    thresholds = [1, 2, 4, 8, 16, 32]
    
    # Run threshold ablation study (use subset for faster computation during development)
    results, thresholds = study.run_threshold_ablation_study(samples, thresholds, n_samples=100, print_selections=True)
    
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
