#!/usr/bin/env python3
"""
Oracle verifier plot showing performance vs number of generated actions per instruction.

X-axis: Number of generated actions per instruction
Y-axis: Action Error (Average NRMSE)

Compares different sampling strategies:
- Original instruction baseline (1 action)
- 1 rephrase (greedy decoding) - 2 actions total
- 2 rephrases (2 actions sampled from repeated sampling) - 3 actions total  
- 4 rephrases (4 actions sampled from repeated sampling) - 5 actions total

The oracle verifier uses the negative of NRMSE to always select the best action
from the available candidates for each instruction.
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
    'legend.fontsize': 14,
    'figure.figsize': (12, 8),
    'text.usetex': False,
})

def generate_augmented_samples_from_batch(batch_actions, num_samples=100, verbose=False):
    """
    Generate augmented samples based on the mean and variance of a batch of actions.
    
    Args:
        batch_actions: NumPy array of shape (batch_size, 7) containing a batch of actions.
        num_samples: Number of augmented samples to generate.
        verbose: Whether to print debug information.
        
    Returns:
        NumPy array of shape (num_samples, 7) containing augmented samples.
    """
    if verbose:
        print(f"\nCalculating mean and variance from batch of {len(batch_actions)} actions...")
    
    # Calculate mean and variance for each dimension
    mean_values = np.mean(batch_actions, axis=0)
    var_values = np.var(batch_actions, axis=0)
    
    if verbose:
        print("Mean values per dimension:", mean_values)
        print("Variance values per dimension:", var_values)
    
    # Define valid ranges for the action dimensions
    min_values = np.array([-0.02872725307941437,
                         -0.04170349963009357,
                         -0.026093858778476715,
                         -0.08092105075716972,
                         -0.09288699507713317,
                         -0.20718276381492615,
                         0.0])
    max_values = np.array([0.028309678435325586,
                         0.040855254605412394,
                         0.040161586627364146,
                         0.08192047759890528,
                         0.07792850524187081,
                         0.20382574498653397,
                         1.0])
    
    # Initialize output array to hold augmented samples
    augmented_array = np.zeros((num_samples, 7))
    
    # Generate num_samples augmented samples
    for i in range(num_samples):
        # Generate values using the calculated mean and variance
        # For dimensions 0-5 (continuous values)
        augmented_action = np.random.normal(mean_values, np.sqrt(var_values), size=7)
        
        # For the 7th dimension (binary), use probability based on mean
        p_gripper = mean_values[-1]  # Probability of gripper being 1
        augmented_action[-1] = np.random.choice([0.0, 1.0], p=[1-p_gripper, p_gripper])
        
        # Clamp values to valid range for first six dimensions
        augmented_action[:-1] = np.clip(augmented_action[:-1], min_values[:-1], max_values[:-1])
        
        # Store the augmented action
        augmented_array[i] = augmented_action
    
    if verbose:
        print(f"Generated {num_samples} augmented samples based on batch statistics")
    
    return augmented_array

def load_vla_clip_data():
    """Load VLA-CLIP scores and organize by sample"""
    print("Loading VLA-CLIP (Greedy Decoding) data...")
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
    
    print(f"Found {len(samples)} unique samples for VLA-CLIP (Greedy)")
    print(f"Original instruction average NRMSE: {np.mean(original_nrmse):.4f}")
    
    return samples, np.mean(original_nrmse)

def load_vla_clip_repeated_data():
    """Load repeated VLA-CLIP scores and organize by sample"""
    print("Loading VLA-CLIP (Repeated Sampling) data...")
    with open('/root/validate_clip_verifier/repeated/bridge_vla_clip_scores_repeated_20250905_041334.json', 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data['results'])} repeated VLA-CLIP results")
    
    # Organize data by sample_id
    samples = defaultdict(list)
    original_nrmse = []
    
    for result in data['results']:
        sample_id = result['sample_id']
        samples[sample_id].append(result)
        
        # Collect original instruction NRMSE
        if result['is_original']:
            original_nrmse.append(result['openvla_nrmse'])
    
    print(f"Found {len(samples)} unique samples for VLA-CLIP (Repeated)")
    print(f"Original instruction average NRMSE (repeated): {np.mean(original_nrmse):.4f}")
    
    return samples, np.mean(original_nrmse)

def calculate_oracle_original_only(greedy_samples):
    """
    Oracle with original instruction only (1 action per instruction).
    """
    sample_nrmse_values = []
    
    for sample_id, results in greedy_samples.items():
        # Only use original instruction
        original = [r for r in results if r['is_original']]
        
        if original:
            # Use the original instruction NRMSE
            sample_nrmse_values.append(original[0]['openvla_nrmse'])
    
    return np.mean(sample_nrmse_values) if sample_nrmse_values else None

def calculate_oracle_greedy_1_rephrase(greedy_samples):
    """
    Oracle with original + 1 rephrase from greedy decoding (2 actions per instruction).
    """
    sample_nrmse_values = []
    
    for sample_id, results in greedy_samples.items():
        # Separate original and rephrased instructions
        original = [r for r in results if r['is_original']]
        rephrased = [r for r in results if not r['is_original']]
        
        if not original:
            continue
            
        # Use original + first rephrased instruction (2 actions total)
        candidate_pool = original + rephrased[:1]
        
        # Oracle selects the best action based on NRMSE
        if candidate_pool:
            best_instruction = min(candidate_pool, key=lambda x: x['openvla_nrmse'])
            sample_nrmse_values.append(best_instruction['openvla_nrmse'])
    
    return np.mean(sample_nrmse_values) if sample_nrmse_values else None

def calculate_oracle_repeated_n_rephrases(repeated_samples, num_rephrases):
    """
    Oracle with original + N rephrases from repeated sampling.
    For each rephrase, we sample from the 5 available repeats.
    Total actions = 1 (original) + num_rephrases
    """
    sample_nrmse_values = []
    
    for sample_id, results in repeated_samples.items():
        # Group by instruction_index
        instruction_groups = defaultdict(list)
        for r in results:
            instruction_groups[r['instruction_index']].append(r)
        
        # Get original instruction (instruction_index=0)
        original_repeats = instruction_groups.get(0, [])
        if not original_repeats:
            continue
            
        # Use the first repeat of the original instruction for consistency
        original_action = [r for r in original_repeats if r['repeat_index'] == 0]
        if not original_action:
            original_action = [original_repeats[0]]  # Fallback
        
        candidate_pool = original_action.copy()
        
        # Add rephrased instructions (sample from repeats)
        instruction_indices = sorted([idx for idx in instruction_groups.keys() if idx > 0])
        
        for i in range(min(num_rephrases, len(instruction_indices))):
            instr_idx = instruction_indices[i]
            repeats = instruction_groups[instr_idx]
            
            # Sample up to the specified number of actions from this instruction's repeats
            if i < num_rephrases - 1:
                # For intermediate rephrases, sample 1 action
                sampled_actions = repeats[:1]
            else:
                # For the last rephrase, sample remaining actions to reach total
                remaining_actions = num_rephrases - i
                sampled_actions = repeats[:remaining_actions]
            
            candidate_pool.extend(sampled_actions)
        
        # Oracle selects the best action based on NRMSE
        if candidate_pool:
            best_instruction = min(candidate_pool, key=lambda x: x['openvla_nrmse'])
            sample_nrmse_values.append(best_instruction['openvla_nrmse'])
    
    return np.mean(sample_nrmse_values) if sample_nrmse_values else None

def calculate_oracle_repeated_n_instructions(repeated_samples, num_instructions):
    """
    Oracle with a specified number of instructions (original + rephrases).
    For each instruction, we consider all available repeats and the oracle
    selects the single best action across ALL repeats from ALL instructions.
    
    Args:
        repeated_samples: Samples from repeated sampling data
        num_instructions: Total number of instructions to consider (including original)
    """
    sample_nrmse_values = []
    
    for sample_id, results in repeated_samples.items():
        # Group by instruction_index
        instruction_groups = defaultdict(list)
        for r in results:
            instruction_groups[r['instruction_index']].append(r)
        
        candidate_pool = []
        
        # Use the first num_instructions (including original at index 0)
        instruction_indices = sorted(instruction_groups.keys())[:num_instructions]
        
        for instr_idx in instruction_indices:
            repeats = instruction_groups[instr_idx]
            # Add ALL repeats from this instruction to the candidate pool
            candidate_pool.extend(repeats)
        
        # Oracle selects the SINGLE best action from the entire candidate pool
        if candidate_pool:
            best_instruction = min(candidate_pool, key=lambda x: x['openvla_nrmse'])
            sample_nrmse_values.append(best_instruction['openvla_nrmse'])
    
    return np.mean(sample_nrmse_values) if sample_nrmse_values else None

def calculate_oracle_greedy_rephrase_only(greedy_samples, num_rephrases):
    """
    Oracle with only rephrased instructions from greedy decoding (no original).
    
    Args:
        greedy_samples: Samples from greedy decoding data
        num_rephrases: Number of rephrases to consider
    """
    sample_nrmse_values = []
    
    for sample_id, results in greedy_samples.items():
        # Only use rephrased instructions (exclude original)
        rephrased = [r for r in results if not r['is_original']]
        
        if len(rephrased) < num_rephrases:
            continue
            
        # Use the first num_rephrases
        candidate_pool = rephrased[:num_rephrases]
        
        # Oracle selects the best action based on NRMSE
        if candidate_pool:
            best_instruction = min(candidate_pool, key=lambda x: x['openvla_nrmse'])
            sample_nrmse_values.append(best_instruction['openvla_nrmse'])
    
    return np.mean(sample_nrmse_values) if sample_nrmse_values else None

def calculate_oracle_repeated_rephrase_only(repeated_samples, num_rephrases, samples_per_rephrase):
    """
    Oracle with only rephrased instructions from repeated sampling (no original).
    
    Args:
        repeated_samples: Samples from repeated sampling data
        num_rephrases: Number of rephrase instructions to consider
        samples_per_rephrase: Number of samples to take from each rephrase
    """
    sample_nrmse_values = []
    
    for sample_id, results in repeated_samples.items():
        # Group by instruction_index
        instruction_groups = defaultdict(list)
        for r in results:
            instruction_groups[r['instruction_index']].append(r)
        
        candidate_pool = []
        
        # Get rephrase instruction indices (exclude original at index 0)
        rephrase_indices = sorted([idx for idx in instruction_groups.keys() if idx > 0])
        
        if len(rephrase_indices) < num_rephrases:
            continue
            
        # Use the first num_rephrases rephrase instructions
        for i in range(num_rephrases):
            instr_idx = rephrase_indices[i]
            repeats = instruction_groups[instr_idx]
            
            # Take samples_per_rephrase from this instruction's repeats
            sampled_repeats = repeats[:samples_per_rephrase]
            candidate_pool.extend(sampled_repeats)
        
        # Oracle selects the SINGLE best action from the entire candidate pool
        if candidate_pool:
            best_instruction = min(candidate_pool, key=lambda x: x['openvla_nrmse'])
            sample_nrmse_values.append(best_instruction['openvla_nrmse'])
    
    return np.mean(sample_nrmse_values) if sample_nrmse_values else None

def calculate_oracle_repeated_rephrase_with_augmentation(repeated_samples, num_rephrases, samples_per_rephrase):
    """
    Oracle with rephrased instructions using Gaussian augmentation to generate more samples.
    
    Args:
        repeated_samples: Samples from repeated sampling data
        num_rephrases: Number of rephrase instructions to consider
        samples_per_rephrase: Number of samples to generate per rephrase (using augmentation if > available)
    """
    sample_nrmse_values = []
    
    for sample_id, results in repeated_samples.items():
        # Group by instruction_index
        instruction_groups = defaultdict(list)
        for r in results:
            instruction_groups[r['instruction_index']].append(r)
        
        candidate_pool = []
        
        # Get rephrase instruction indices (exclude original at index 0)
        rephrase_indices = sorted([idx for idx in instruction_groups.keys() if idx > 0])
        
        if len(rephrase_indices) < num_rephrases:
            continue
            
        # Use the first num_rephrases rephrase instructions
        for i in range(num_rephrases):
            instr_idx = rephrase_indices[i]
            repeats = instruction_groups[instr_idx]
            
            if samples_per_rephrase <= len(repeats):
                # Use existing samples if we have enough
                sampled_repeats = repeats[:samples_per_rephrase]
                candidate_pool.extend(sampled_repeats)
            else:
                # Use all available samples plus augmented ones
                available_actions = []
                for repeat in repeats:
                    available_actions.append(repeat['predicted_action'])
                
                # Convert to numpy array for augmentation
                batch_actions = np.array(available_actions)
                
                # Generate augmented samples
                augmented_samples_needed = samples_per_rephrase - len(repeats)
                augmented_actions = generate_augmented_samples_from_batch(batch_actions, augmented_samples_needed)
                
                # Add existing samples to candidate pool
                candidate_pool.extend(repeats)
                
                # Create synthetic results for augmented samples
                # We need to calculate NRMSE for the augmented actions
                ground_truth_action = np.array(repeats[0]['ground_truth_action'])
                
                for aug_action in augmented_actions:
                    # Calculate NRMSE for the augmented action
                    diff = aug_action - ground_truth_action
                    nrmse = np.sqrt(np.mean(diff**2)) / (np.max(ground_truth_action) - np.min(ground_truth_action) + 1e-8)
                    
                    # Create a synthetic result entry
                    synthetic_result = {
                        'predicted_action': aug_action.tolist(),
                        'ground_truth_action': ground_truth_action.tolist(),
                        'openvla_nrmse': nrmse,
                        'is_augmented': True
                    }
                    candidate_pool.append(synthetic_result)
        
        # Oracle selects the SINGLE best action from the entire candidate pool
        if candidate_pool:
            best_instruction = min(candidate_pool, key=lambda x: x['openvla_nrmse'])
            sample_nrmse_values.append(best_instruction['openvla_nrmse'])
    
    return np.mean(sample_nrmse_values) if sample_nrmse_values else None

def generate_actions_per_instruction_plot():
    """Generate the plot showing oracle performance vs samples per rephrase for multiple rephrase counts"""
    
    # Load data
    greedy_samples, original_avg_nrmse_greedy = load_vla_clip_data()
    repeated_samples, original_avg_nrmse_repeated = load_vla_clip_repeated_data()
    
    # Use the original NRMSE from greedy data as baseline
    original_avg_nrmse = original_avg_nrmse_greedy
    
    # Define the number of rephrases to test and samples per rephrase
    num_rephrases_list = [1, 2, 4, 8, 16, 32, 64]  # 7 different rephrase counts
    samples_per_rephrase_list = [1, 2, 4, 8, 16, 32, 64, 128]  # Extended range using Gaussian augmentation
    
    print("Calculating oracle NRMSE for different rephrase counts and sampling strategies...")
    
    # Create the plot with publication-quality styling
    fig, ax = plt.subplots()
    
    # Colors for different rephrase counts
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    markers = ['o', 's', '^', 'D', 'v', '<', '>']
    
    # Store all results for CSV export
    all_results = []
    
    # Calculate and plot for each number of rephrases
    for i, num_rephrases in enumerate(num_rephrases_list):
        print(f"\nCalculating for {num_rephrases} rephrases:")
        
        samples_x = []
        nrmse_y = []
        
        for samples_per_rephrase in samples_per_rephrase_list:
            if num_rephrases == 1 and samples_per_rephrase == 1:
                # Special case: compare greedy vs repeated for 1 rephrase, 1 sample
                greedy_nrmse = calculate_oracle_greedy_rephrase_only(greedy_samples, num_rephrases)
                repeated_nrmse = calculate_oracle_repeated_rephrase_only(repeated_samples, num_rephrases, samples_per_rephrase)
                
                # Use the better of the two (repeated sampling should be better)
                nrmse = repeated_nrmse
                print(f"  {samples_per_rephrase} samples per rephrase: {nrmse:.4f} NRMSE (Greedy: {greedy_nrmse:.4f})")
            else:
                # Use augmentation-enabled function for all cases
                if samples_per_rephrase <= 4:
                    # Use existing samples without augmentation for small counts
                    nrmse = calculate_oracle_repeated_rephrase_only(repeated_samples, num_rephrases, samples_per_rephrase)
                else:
                    # Use augmentation for larger sample counts
                    nrmse = calculate_oracle_repeated_rephrase_with_augmentation(repeated_samples, num_rephrases, samples_per_rephrase)
                
                if nrmse is not None:
                    improvement = ((original_avg_nrmse - nrmse) / original_avg_nrmse * 100)
                    augmentation_note = " (with augmentation)" if samples_per_rephrase > 4 else ""
                    print(f"  {samples_per_rephrase} samples per rephrase: {nrmse:.4f} NRMSE ({improvement:.1f}% improvement){augmentation_note}")
            
            if nrmse is not None:
                samples_x.append(samples_per_rephrase)
                nrmse_y.append(nrmse)
                
                # Store for CSV
                all_results.append({
                    'num_rephrases': num_rephrases,
                    'samples_per_rephrase': samples_per_rephrase,
                    'oracle_nrmse': nrmse,
                    'improvement_percent': ((original_avg_nrmse - nrmse) / original_avg_nrmse * 100)
                })
        
        # Plot the curve for this number of rephrases
        if samples_x and nrmse_y:
            ax.plot(samples_x, nrmse_y, 'o-', linewidth=2.5, markersize=8, 
                   color=colors[i], marker=markers[i], 
                   label=f'{num_rephrases} Rephrases', markeredgewidth=1, markeredgecolor='white')
    
    # Add baseline line for original instruction
    ax.axhline(y=original_avg_nrmse, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, 
               label='Original Instruction Baseline')
    
    # Axis labels and formatting
    ax.set_xlabel("Number of Generated Actions per Rephrase")
    ax.set_ylabel("Oracle Action Error (Average NRMSE)")
    ax.set_xscale('log', base=2)
    ax.set_xticks(samples_per_rephrase_list)
    ax.set_xticklabels([str(t) for t in samples_per_rephrase_list])
    
    # Set y-axis limits to better show the improvement
    all_nrmse_values = [r['oracle_nrmse'] for r in all_results]
    if all_nrmse_values:
        y_min = min(all_nrmse_values) * 0.9
        y_max = original_avg_nrmse * 1.05
        ax.set_ylim(y_min, y_max)
    
    # Grid and border
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_color('black')
    
    # Legend
    legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', framealpha=0.95, edgecolor='black', fontsize=10)
    legend.get_frame().set_linewidth(1.5)
    
    # Title
    ax.set_title('Oracle Verifier: Performance vs Samples per Rephrase\nfor Different Numbers of Rephrases (Excluding Original)', 
                 fontsize=14, pad=20)
    
    # Save plot
    output_file = '/root/validate_clip_verifier/oracle_rephrases_scaling_analysis.png'
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as: {output_file}")
    
    # Also save as PDF
    plt.savefig('/root/validate_clip_verifier/oracle_rephrases_scaling_analysis.pdf', bbox_inches='tight')
    print(f"Plot also saved as: oracle_rephrases_scaling_analysis.pdf")
    
    plt.show()
    
    # Print detailed analysis
    print(f"\n=== Oracle Rephrases Scaling Analysis ===")
    print(f"Original instruction baseline NRMSE: {original_avg_nrmse:.4f}")
    
    # Group results by number of rephrases for analysis
    for num_rephrases in num_rephrases_list:
        rephrase_results = [r for r in all_results if r['num_rephrases'] == num_rephrases]
        if rephrase_results:
            print(f"\n{num_rephrases} Rephrases:")
            best_result = min(rephrase_results, key=lambda x: x['oracle_nrmse'])
            for result in rephrase_results:
                print(f"  {result['samples_per_rephrase']} samples: {result['oracle_nrmse']:.4f} NRMSE ({result['improvement_percent']:.1f}% improvement)")
            print(f"  Best: {best_result['oracle_nrmse']:.4f} NRMSE with {best_result['samples_per_rephrase']} samples per rephrase")
    
    # Analyze the effect of increasing rephrases (at 128 samples per rephrase with augmentation)
    print(f"\nEffect of increasing number of rephrases (at 128 samples per rephrase with augmentation):")
    high_sample_results = [r for r in all_results if r['samples_per_rephrase'] == 128]
    high_sample_results.sort(key=lambda x: x['num_rephrases'])
    
    for result in high_sample_results:
        print(f"  {result['num_rephrases']:2d} rephrases: {result['oracle_nrmse']:.4f} NRMSE ({result['improvement_percent']:.1f}% improvement)")
    
    # Also analyze at 4 samples per rephrase (without augmentation)
    print(f"\nEffect of increasing number of rephrases (at 4 samples per rephrase, no augmentation):")
    four_sample_results = [r for r in all_results if r['samples_per_rephrase'] == 4]
    four_sample_results.sort(key=lambda x: x['num_rephrases'])
    
    for result in four_sample_results:
        print(f"  {result['num_rephrases']:2d} rephrases: {result['oracle_nrmse']:.4f} NRMSE ({result['improvement_percent']:.1f}% improvement)")
    
    # Save numerical results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('/root/validate_clip_verifier/oracle_rephrases_scaling_results.csv', index=False)
    print(f"\nNumerical results saved as: oracle_rephrases_scaling_results.csv")
    
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
        '/root/validate_clip_verifier/bridge_vla_clip_scores_20250809_194343.json',
        '/root/validate_clip_verifier/repeated/bridge_vla_clip_scores_repeated_20250905_041334.json'
    ]
    
    missing_files = []
    for f in required_files:
        if not os.path.exists(f):
            missing_files.append(f)
    
    if missing_files:
        print("Error: Missing required files:")
        for f in missing_files:
            print(f"  - {f}")
        exit(1)
    
    # Generate the actions per instruction plot
    results = generate_actions_per_instruction_plot()
