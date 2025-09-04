#!/usr/bin/env python3
"""
Generate OpenVLA actions using Gaussian sampling approach.
For each original instruction, generate a small batch to estimate mean/variance,
then generate 128 augmented samples using Gaussian distribution.
Total expected datapoints: 100 samples * 128 augmented = 12800 datapoints.
"""

import requests
import json
import numpy as np
import os
from tqdm import tqdm
from datetime import datetime
from token2action import TokenActionConverter

# Define the ranges for NRMSE calculation (from collect.py)
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
ranges = max_values - min_values

def calculate_nrmse(action0, action1):
    """
    Calculate normalized root mean squared error between two actions
    """
    # Normalize the difference by the range
    normalized_diff = (action0 - action1) / ranges
    nrmse = np.sqrt(np.mean(normalized_diff**2))
    return nrmse

def get_batch_actions(instruction, image_path, num_samples, temperature=0.5):
    """
    Get batch actions for a single instruction using OpenVLA.
    
    Args:
        instruction: Single instruction string
        image_path: Path to the image file
        num_samples: Number of samples to generate
        temperature: Temperature for sampling
    
    Returns:
        Tuple of (output_ids, actions) as numpy arrays
    """
    image_path = os.path.abspath(image_path)
    
    # Create list of identical instructions for batch processing
    instructions = [instruction] * num_samples
    
    payload = {
        "instructions": instructions,
        "image_path": image_path,
        "temperature": temperature
    }
    
    res = requests.post(
        "http://localhost:3200/batch",
        data=json.dumps(payload),
        headers={'Content-Type': 'application/json'}
    )
    res.raise_for_status()
    result = json.loads(res.text)
    return np.array(result["output_ids"]), np.array(result["actions"])

def generate_augmented_samples_from_batch(batch_actions, num_samples=128):
    """
    Generate augmented samples based on the mean and variance of a batch of actions.
    
    Args:
        batch_actions: NumPy array of shape (batch_size, 7) containing a batch of actions.
        num_samples: Number of augmented samples to generate.
        
    Returns:
        NumPy array of shape (num_samples, 7) containing augmented samples.
    """
    # Calculate mean and variance for each dimension
    mean_values = np.mean(batch_actions, axis=0)
    var_values = np.var(batch_actions, axis=0)
    
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
    
    return augmented_array

def load_bridge_data():
    """Load bridge samples"""
    with open('../bridge_samples.json', 'r') as f:
        bridge_data = json.load(f)
    return bridge_data

def main():
    print("Loading bridge dataset...")
    bridge_data = load_bridge_data()
    
    samples = bridge_data['samples']
    print(f"Found {len(samples)} bridge samples")
    
    # Initialize token-action converter
    print("Initializing token-action converter...")
    token_converter = TokenActionConverter(n_action_bins=256, unnorm_key="bridge_orig")
    
    # Configuration
    batch_size = 10  # Number of samples to generate for mean/variance estimation
    num_augmented = 128  # Number of augmented samples per original instruction
    temperature = 1.0  # Temperature for initial batch generation
    
    expected_total = len(samples) * num_augmented
    print(f"Expected total datapoints: {expected_total}")
    
    # Results list to store all datapoints
    results = []
    
    # Process each sample
    with tqdm(total=expected_total, desc="Generating Gaussian actions") as pbar:
        for sample_idx, sample in enumerate(samples):
            sample_id = sample['sample_id']
            original_instruction = sample['original_instruction']
            ground_truth_action = np.array(sample['current_ground_truth_action'])
            
            # Get OpenVLA processed image path
            image_filename = sample['state']['agent_view_image_file']
            base_name = image_filename.split('_')[0]  # Extract number from "N_clip.jpg"
            openvla_image_path = f"../processed_images/openvla/{base_name}_openvla.jpg"
            
            if not os.path.exists(openvla_image_path):
                print(f"Warning: OpenVLA image not found: {openvla_image_path}")
                # Skip this sample and update progress bar
                pbar.update(num_augmented)
                continue
            
            try:
                # Generate a small batch of actions to estimate mean and variance
                batch_output_ids, batch_actions = get_batch_actions(
                    instruction=original_instruction,
                    image_path=openvla_image_path,
                    num_samples=batch_size,
                    temperature=temperature
                )
                
                # Calculate batch statistics
                batch_mean = np.mean(batch_actions, axis=0)
                batch_variance = np.var(batch_actions, axis=0)
                
                # Generate augmented samples based on the mean and variance of the batch
                augmented_actions = generate_augmented_samples_from_batch(
                    batch_actions=batch_actions, 
                    num_samples=num_augmented
                )
                
                # Create result entries for each augmented action
                for aug_idx, augmented_action in enumerate(augmented_actions):
                    # Calculate NRMSE between ground truth and augmented action
                    nrmse = calculate_nrmse(ground_truth_action, augmented_action)
                    
                    # Convert augmented action back to output_ids using token converter
                    try:
                        output_ids = token_converter.action_to_token(augmented_action)
                        output_ids_list = output_ids.tolist()
                    except Exception as e:
                        print(f"Warning: Could not convert action to tokens for sample {sample_id}, aug {aug_idx}: {e}")
                        output_ids_list = []
                    
                    # Create result entry (following same format as generate_vla_actions.py)
                    result_entry = {
                        'sample_id': sample_id,
                        'sample_index': sample_idx,
                        'instruction_index': aug_idx,  # Use aug_idx as instruction_index for consistency
                        'is_original': False,  # All are augmented, not original
                        'instruction': original_instruction,
                        'original_instruction': original_instruction,
                        'ground_truth_action': ground_truth_action.tolist(),
                        'generated_action': augmented_action.tolist(),
                        'output_ids': output_ids_list,  # Convert actions to tokens
                        'nrmse': float(nrmse),
                        'image_path': openvla_image_path,
                        'episode_id': sample['state']['episode_id'],
                        'timestep': sample['state']['timestep'],
                        # Additional Gaussian-specific metadata
                        'batch_mean': batch_mean.tolist(),
                        'batch_variance': batch_variance.tolist(),
                        'batch_size': batch_size,
                        'generation_method': 'gaussian_augmented'
                    }
                    
                    results.append(result_entry)
                    pbar.update(1)
                    
            except Exception as e:
                print(f"Error processing sample {sample_id}: {e}")
                # Skip this sample and update progress bar
                pbar.update(num_augmented)
                continue
    
    print(f"\nGenerated {len(results)} datapoints")
    
    # Save results to JSON file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"bridge_openvla_actions_gaussian_{timestamp}.json"
    
    # Prepare final data structure (following same format as generate_vla_actions.py)
    final_data = {
        'metadata': {
            'timestamp': timestamp,
            'total_datapoints': len(results),
            'expected_datapoints': expected_total,
            'source_bridge_file': 'bridge_samples.json',
            'source_rephrase_file': 'gaussian_augmented',  # Indicate this uses Gaussian method
            'model': 'OpenVLA',
            'temperature': temperature,
            'api_endpoint': 'http://localhost:3200',
            'method': 'gaussian_sampling',
            'batch_size': batch_size,
            'num_augmented_per_sample': num_augmented,
            'nrmse_ranges': {
                'min_values': min_values.tolist(),
                'max_values': max_values.tolist(),
                'ranges': ranges.tolist()
            }
        },
        'results': results
    }
    
    print(f"Saving results to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(final_data, f, indent=2)
    
    print(f"Results saved successfully!")
    
    # Print summary statistics (following same format as generate_vla_actions.py)
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    nrmse_values = [r['nrmse'] for r in results]
    
    print(f"Total datapoints: {len(results)}")
    print(f"Samples processed: {len(set(r['sample_id'] for r in results))}")
    print(f"Augmented samples per original: {num_augmented}")
    print(f"Overall NRMSE - Mean: {np.mean(nrmse_values):.4f}, Std: {np.std(nrmse_values):.4f}")
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    main()
