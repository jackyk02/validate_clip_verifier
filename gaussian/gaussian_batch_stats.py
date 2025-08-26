# Example: python scripts/gaussian_batch_stats.py --policy openvla --batch_size 10 --temperature 1.0

import numpy as np
import pandas as pd
import json
import os
import glob
import time
import pickle
from tqdm import tqdm
import openpyxl
import requests
import argparse

# Policy to port mapping
POLICY_API = {
    "cogact": 2100,
    "octo": 2200,
    "openvla": 2300,
    "spatialvla": 2400
}

def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='Run experiment with specific policy and parameters')
    parser.add_argument('--policy', type=str, required=True, 
                      choices=['cogact', 'octo', 'openvla', 'spatialvla'],
                      help='Policy to use for the experiment')
    parser.add_argument('--batch_size', type=int, default=10,
                      help='Number of actions to sample for calculating mean and variance')
    parser.add_argument('--num_augmented', type=int, default=100,
                      help='Number of augmented samples to generate')
    parser.add_argument('--temperature', type=float, default=0.5,
                      help='Temperature parameter for action generation (higher values increase randomness)')
    return parser.parse_args()

def calculate_nrmse(action0, action1):
    """
    Calculate normalized root mean squared error between two actions
    """
    # Define the ranges for NRMSE calculation
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
    
    # Normalize the difference by the range
    normalized_diff = (action0 - action1) / ranges
    nrmse = np.sqrt(np.mean(normalized_diff**2))
    return nrmse

def generate_actions(image_path, instruction, number_samples, temperature, api_url):
    """
    Generate actions using the API endpoint.
    """
    requests.post(
        f"{api_url}/reset",
        json={"instruction": instruction},
        headers={"Content-Type": "application/json"}
    )

    payload = {
        "image_path": image_path,
        "instruction": instruction,
        "num_samples": number_samples,
        "temperature": temperature,
    }
    
    response = requests.post(
        f"{api_url}/batch",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    result = response.json()
    return np.array(result['actions'])

def generate_augmented_samples_from_batch(batch_actions, num_samples=100):
    """
    Generate augmented samples based on the mean and variance of a batch of actions.
    
    Args:
        batch_actions: NumPy array of shape (batch_size, 7) containing a batch of actions.
        num_samples: Number of augmented samples to generate.
        
    Returns:
        NumPy array of shape (num_samples, 7) containing augmented samples.
    """
    print(f"\nCalculating mean and variance from batch of {len(batch_actions)} actions...")
    
    # Calculate mean and variance for each dimension
    mean_values = np.mean(batch_actions, axis=0)
    var_values = np.var(batch_actions, axis=0)
    
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
    
    print(f"Generated {num_samples} augmented samples based on batch statistics")
    
    return augmented_array

def save_to_csv(data_buffer, base_filename="dataset_results", file_number=0):
    """
    Save data buffer to csv file with sequential numbering in the log directory.
    """
    if not data_buffer:
        return
        
    # Create log directory if it doesn't exist
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
        
    df = pd.DataFrame(data_buffer)
    filename = os.path.join(log_dir, f"{base_filename}_{file_number}.csv")
    df.to_csv(filename, index=False)
    print(f"Saved file: {filename}")

def run_experiment(policy, batch_size=10, num_augmented=100, temperature=0.5):
    """
    Run the experiment by sampling a small batch from openvla, 
    then fit both the mean and variance from this batch and generate augmented samples.
    Loop through all datapoints (1999900, 2000000).
    """
    # Get API port based on policy
    port = POLICY_API.get(policy)
    if not port:
        raise ValueError(f"Invalid policy: {policy}")
    
    api_url = f"http://localhost:{port}"
    print(f"Running experiment with policy: {policy}, port: {port}")
    print(f"Batch size for mean/variance estimation: {batch_size}")
    print(f"Number of augmented samples to generate: {num_augmented}")
    print(f"Temperature for action generation: {temperature}")
    
    # Load necessary data
    baseline_actions = pickle.load(open("data/full_bridge/action_full.pkl", "rb"))
    instructions = pickle.load(open("data/full_bridge/instruction_full.pkl", "rb"))
    
    # Buffer to accumulate data before writing to csv
    data_buffer = []
    points_per_file = 10000
    current_file_number = 0
    total_points = 0
    
    # Process each datapoint in the range
    for idx in range(1999900, 2000000):
        print(f"Processing index: {idx}")
        
        # Generate a small batch of actions to estimate mean and variance
        batch_actions = generate_actions(
            image_path=f"/root/vla-api/data/full_bridge/images/{idx}.jpg",
            instruction=instructions[idx],
            number_samples=batch_size,
            temperature=temperature,  # Using user-specified temperature
            api_url=api_url
        )
        
        # Calculate mean and variance from the batch
        batch_mean = np.mean(batch_actions, axis=0)
        batch_variance = np.var(batch_actions, axis=0)
        
        # Generate augmented samples based on the mean and variance of the batch
        augmented_actions = generate_augmented_samples_from_batch(
            batch_actions=batch_actions, 
            num_samples=num_augmented
        )
        
        # Create data points comparing baseline action with each augmented action
        for pair_idx, augmented_action in enumerate(augmented_actions):
            # Calculate NRMSE between baseline and augmented action
            nrmse = calculate_nrmse(baseline_actions[idx], augmented_action)
            
            data_dict = {
                'index': idx,
                'pair_index': pair_idx,
                'action0': baseline_actions[idx],
                'action1': augmented_action,
                'nrmse': nrmse
            }
            data_buffer.append(data_dict)
            total_points += 1
            
            # Check if we've reached the points threshold for a new file
            if total_points % points_per_file == 0:
                save_to_csv(data_buffer, 
                           base_filename=f"gaussian_full_bridge_{policy}_bs{batch_size}_n{num_augmented}_temp{temperature}",  
                           file_number=current_file_number)
                data_buffer = []  # Clear buffer after writing
                current_file_number += 1
        
        # Save batch statistics to a separate file (one row per index)
        stats_dict = {
            'index': idx,
            'batch_size': batch_size,
            'temperature': temperature,
            'batch_mean': batch_mean.tolist(),
            'batch_variance': batch_variance.tolist()
        }
        
        # Append to statistics CSV
        stats_df = pd.DataFrame([stats_dict])
        stats_path = os.path.join("log", f"batch_statistics_{policy}_bs{batch_size}_temp{temperature}.csv")
        
        # If file doesn't exist, create it with header, otherwise append without header
        if not os.path.exists(stats_path):
            stats_df.to_csv(stats_path, index=False)
        else:
            stats_df.to_csv(stats_path, mode='a', header=False, index=False)
    
    # Write any remaining data in buffer to a final file
    if data_buffer:
        save_to_csv(data_buffer, 
                   base_filename=f"gaussian_full_bridge_{policy}_bs{batch_size}_n{num_augmented}_temp{temperature}", 
                   file_number=current_file_number)
    
    print("Experiment completed successfully.")

if __name__ == '__main__':
    args = parse_arguments()
    run_experiment(
        policy=args.policy,
        batch_size=args.batch_size,
        num_augmented=args.num_augmented,
        temperature=args.temperature
    )