#!/usr/bin/env python3
"""
Script to analyze the relationship between instruction embedding similarity and action errors.

This script:
1. Loads the JSON data containing instructions and openvla_nrmse scores
2. Uses BERT to embed both "instruction" and "original_instruction" 
3. Calculates cosine similarity between embeddings
4. Analyzes correlation between embedding similarity and openvla_nrmse
5. Generates visualizations and statistical analysis
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import argparse
import warnings
warnings.filterwarnings('ignore')

class InstructionEmbeddingAnalyzer:
    def __init__(self, model_name='bert-base-uncased'):
        """Initialize the analyzer with BERT model."""
        print(f"Loading BERT model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded on device: {self.device}")
    
    def get_bert_embedding(self, text):
        """Get BERT embedding for a single text."""
        # Tokenize and encode
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, 
                               padding=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use [CLS] token embedding (first token)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return embedding.flatten()
    
    def get_batch_embeddings(self, texts, batch_size=32):
        """Get BERT embeddings for a batch of texts."""
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Computing embeddings"):
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
    
    def load_data(self, json_file):
        """Load and process the JSON data."""
        print(f"Loading data from {json_file}")
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        results = data['results']
        print(f"Loaded {len(results)} samples")
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(results)
        
        # Filter out samples where instruction == original_instruction (no rephrasing)
        df_rephrased = df[df['instruction'] != df['original_instruction']].copy()
        print(f"Found {len(df_rephrased)} rephrased samples")
        
        return df, df_rephrased
    
    def compute_similarities(self, df, nrmse_threshold=0.55):
        """Compute cosine similarities between instruction embeddings."""
        print("Computing BERT embeddings...")
        
        # Remove outliers based on NRMSE threshold
        initial_count = len(df)
        df_filtered = df[df['openvla_nrmse'] <= nrmse_threshold].copy()
        removed_count = initial_count - len(df_filtered)
        print(f"Removed {removed_count} outliers with NRMSE > {nrmse_threshold}")
        print(f"Remaining samples: {len(df_filtered)}")
        
        if len(df_filtered) == 0:
            raise ValueError(f"No samples remaining after filtering with threshold {nrmse_threshold}")
        
        # Get unique instructions to avoid recomputing
        unique_instructions = list(set(df_filtered['instruction'].tolist() + df_filtered['original_instruction'].tolist()))
        print(f"Computing embeddings for {len(unique_instructions)} unique instructions")
        
        # Compute embeddings
        embeddings = self.get_batch_embeddings(unique_instructions)
        
        # Create mapping from instruction to embedding
        instruction_to_embedding = dict(zip(unique_instructions, embeddings))
        
        # Get embeddings for each row
        instruction_embeddings = np.array([instruction_to_embedding[inst] for inst in df_filtered['instruction']])
        original_embeddings = np.array([instruction_to_embedding[inst] for inst in df_filtered['original_instruction']])
        
        # Compute cosine similarities
        print("Computing cosine similarities...")
        similarities = []
        for i in range(len(instruction_embeddings)):
            sim = cosine_similarity([instruction_embeddings[i]], [original_embeddings[i]])[0, 0]
            similarities.append(sim)
        
        df_filtered['embedding_similarity'] = similarities
        return df_filtered
    
    def analyze_correlation(self, df):
        """Analyze correlation between embedding similarity and openvla_nrmse."""
        print("\n" + "="*50)
        print("CORRELATION ANALYSIS")
        print("="*50)
        
        # Basic statistics
        print(f"Number of samples: {len(df)}")
        print(f"Embedding similarity - Mean: {df['embedding_similarity'].mean():.4f}, Std: {df['embedding_similarity'].std():.4f}")
        print(f"OpenVLA NRMSE - Mean: {df['openvla_nrmse'].mean():.4f}, Std: {df['openvla_nrmse'].std():.4f}")
        
        # Correlation analysis
        pearson_corr, pearson_p = pearsonr(df['embedding_similarity'], df['openvla_nrmse'])
        spearman_corr, spearman_p = spearmanr(df['embedding_similarity'], df['openvla_nrmse'])
        
        print(f"\nCorrelation between embedding similarity and OpenVLA NRMSE:")
        print(f"Pearson correlation: {pearson_corr:.4f} (p-value: {pearson_p:.4e})")
        print(f"Spearman correlation: {spearman_corr:.4f} (p-value: {spearman_p:.4e})")
        
        # Interpretation
        if pearson_p < 0.05:
            direction = "negative" if pearson_corr < 0 else "positive"
            strength = "weak" if abs(pearson_corr) < 0.3 else "moderate" if abs(pearson_corr) < 0.7 else "strong"
            print(f"\nResult: There is a statistically significant {strength} {direction} correlation.")
            if pearson_corr < 0:
                print("This suggests that when instructions are more similar in embedding space,")
                print("the action error (openvla_nrmse) tends to be LOWER.")
            else:
                print("This suggests that when instructions are more similar in embedding space,")
                print("the action error (openvla_nrmse) tends to be HIGHER.")
        else:
            print(f"\nResult: No statistically significant correlation found (p > 0.05).")
        
        return pearson_corr, pearson_p, spearman_corr, spearman_p
    
    def create_visualizations(self, df, output_prefix="analysis"):
        """Create visualizations of the analysis."""
        print("\nCreating visualizations...")
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Instruction Embedding Similarity vs Action Error Analysis', fontsize=16)
        
        # 1. Scatter plot
        axes[0, 0].scatter(df['embedding_similarity'], df['openvla_nrmse'], alpha=0.6, s=20)
        axes[0, 0].set_xlabel('Embedding Similarity (Cosine)')
        axes[0, 0].set_ylabel('OpenVLA NRMSE')
        axes[0, 0].set_title('Embedding Similarity vs Action Error')
        
        # Add trend line
        z = np.polyfit(df['embedding_similarity'], df['openvla_nrmse'], 1)
        p = np.poly1d(z)
        axes[0, 0].plot(df['embedding_similarity'], p(df['embedding_similarity']), "r--", alpha=0.8)
        
        # 2. Distribution of embedding similarities
        axes[0, 1].hist(df['embedding_similarity'], bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Embedding Similarity')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Embedding Similarities')
        axes[0, 1].axvline(df['embedding_similarity'].mean(), color='red', linestyle='--', 
                          label=f'Mean: {df["embedding_similarity"].mean():.3f}')
        axes[0, 1].legend()
        
        # 3. Distribution of NRMSE
        axes[1, 0].hist(df['openvla_nrmse'], bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('OpenVLA NRMSE')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Action Errors')
        axes[1, 0].axvline(df['openvla_nrmse'].mean(), color='red', linestyle='--',
                          label=f'Mean: {df["openvla_nrmse"].mean():.3f}')
        axes[1, 0].legend()
        
        # 4. Binned analysis
        # Create bins based on similarity quartiles
        df['similarity_bin'] = pd.qcut(df['embedding_similarity'], q=4, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])
        bin_stats = df.groupby('similarity_bin')['openvla_nrmse'].agg(['mean', 'std', 'count'])
        
        axes[1, 1].bar(range(len(bin_stats)), bin_stats['mean'], yerr=bin_stats['std'], 
                      capsize=5, alpha=0.7)
        axes[1, 1].set_xlabel('Embedding Similarity Quartiles')
        axes[1, 1].set_ylabel('Mean OpenVLA NRMSE')
        axes[1, 1].set_title('Mean Action Error by Similarity Quartile')
        axes[1, 1].set_xticks(range(len(bin_stats)))
        axes[1, 1].set_xticklabels(bin_stats.index, rotation=45)
        
        # Add sample counts
        for i, (idx, row) in enumerate(bin_stats.iterrows()):
            axes[1, 1].text(i, row['mean'] + row['std'] + 0.001, f'n={row["count"]}', 
                           ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{output_prefix}_visualization.png', dpi=300, bbox_inches='tight')
        print(f"Visualization saved as {output_prefix}_visualization.png")
        
        # Print binned statistics
        print(f"\nBinned Analysis (by similarity quartiles):")
        print(bin_stats)
        
        return fig
    
    def save_results(self, df, correlations, output_prefix="analysis"):
        """Save detailed results to files."""
        # Save processed data
        df.to_csv(f'{output_prefix}_results.csv', index=False)
        print(f"Detailed results saved to {output_prefix}_results.csv")
        
        # Save summary statistics
        pearson_corr, pearson_p, spearman_corr, spearman_p = correlations
        
        summary = {
            'total_samples_after_filtering': int(len(df)),
            'nrmse_max_value': float(df['openvla_nrmse'].max()),
            'embedding_similarity_mean': float(df['embedding_similarity'].mean()),
            'embedding_similarity_std': float(df['embedding_similarity'].std()),
            'openvla_nrmse_mean': float(df['openvla_nrmse'].mean()),
            'openvla_nrmse_std': float(df['openvla_nrmse'].std()),
            'pearson_correlation': float(pearson_corr),
            'pearson_p_value': float(pearson_p),
            'spearman_correlation': float(spearman_corr),
            'spearman_p_value': float(spearman_p),
            'significant_correlation': bool(pearson_p < 0.05)
        }
        
        with open(f'{output_prefix}_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary statistics saved to {output_prefix}_summary.json")

def main():
    parser = argparse.ArgumentParser(description='Analyze instruction embedding similarity vs action errors')
    parser.add_argument('json_file', help='Path to the JSON file containing the data')
    parser.add_argument('--model', default='bert-base-uncased', help='BERT model to use')
    parser.add_argument('--output', default='analysis', help='Output file prefix')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for embedding computation')
    parser.add_argument('--nrmse-threshold', type=float, default=0.55, help='Remove outliers with NRMSE higher than this threshold')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = InstructionEmbeddingAnalyzer(model_name=args.model)
    
    # Load data
    df_all, df_rephrased = analyzer.load_data(args.json_file)
    
    # Compute similarities for rephrased samples (with outlier removal)
    df_analyzed = analyzer.compute_similarities(df_rephrased, nrmse_threshold=args.nrmse_threshold)
    
    # Analyze correlations
    correlations = analyzer.analyze_correlation(df_analyzed)
    
    # Create visualizations
    analyzer.create_visualizations(df_analyzed, args.output)
    
    # Save results
    analyzer.save_results(df_analyzed, correlations, args.output)
    
    print(f"\nAnalysis complete! Check the generated files:")
    print(f"- {args.output}_visualization.png")
    print(f"- {args.output}_results.csv") 
    print(f"- {args.output}_summary.json")

if __name__ == "__main__":
    main()
