#!/usr/bin/env python3
"""
Generate OpenVLA actions for red team instructions and their 128 rephrased versions,
then calculate NRMSE between sampled actions and ground truth actions.
Uses red team instructions from q4_random_instructions.csv instead of original instructions.
"""

import requests
import json
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from datetime import datetime

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

def get_batch_actions(instructions, image_path, temperature=0.0):
    """
    Get batch actions for multiple instructions using OpenVLA.
    
    Args:
        instructions: List of instruction strings or a single instruction string
        image_path: Path to the image file
        temperature: Temperature for sampling
    
    Returns:
        Tuple of (output_ids, actions) as numpy arrays
    """
    image_path = os.path.abspath(image_path)
    
    # Handle both single instruction and list of instructions
    if isinstance(instructions, str):
        instructions = [instructions]
    
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

def load_red_team_mapping():
    """Load CSV file and create mapping from sample_id to red_team_instruction"""
    csv_file = 'q4_random_instructions.csv'
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    
    df = pd.read_csv(csv_file)
    sample_to_red_team = {}
    
    for _, row in df.iterrows():
        sample_id = int(row['sample_id'])
        red_team_instruction = str(row['red_team_instruction'])
        sample_to_red_team[sample_id] = red_team_instruction
    
    print(f"Loaded {len(sample_to_red_team)} sample-to-red-team mappings from CSV")
    return sample_to_red_team

def load_bridge_data():
    """Load bridge samples, red team instruction mapping, and instruction rephrases"""
    # Load bridge samples
    with open('../bridge_samples.json', 'r') as f:
        bridge_data = json.load(f)
    
    # Load red team instruction mapping
    sample_to_red_team = load_red_team_mapping()
    
    # Load instruction rephrases
    rephrase_files = [f for f in os.listdir('.') if f.startswith('red_team_instruction_rephrases_') and f.endswith('.json')]
    if not rephrase_files:
        raise FileNotFoundError("No rephrase file found!")
    
    rephrase_file = rephrase_files[0]  # Take the most recent one
    print(f"Loading rephrases from: {rephrase_file}")
    
    with open(rephrase_file, 'r') as f:
        rephrase_data = json.load(f)
    
    return bridge_data, sample_to_red_team, rephrase_data

def main():
    print("Loading bridge dataset, red team mappings, and instruction rephrases...")
    bridge_data, sample_to_red_team, rephrase_data = load_bridge_data()
    
    samples = bridge_data['samples']
    rephrase_instructions = rephrase_data['instructions']
    
    print(f"Found {len(samples)} bridge samples")
    print(f"Found {len(sample_to_red_team)} red team instruction mappings")
    print(f"Found {len(rephrase_instructions)} unique instructions with rephrases")
    
    # Results list to store all datapoints
    results = []
    total_expected = 0
    
    # Count expected datapoints
    for sample in samples:
        sample_id = sample['sample_id']
        if sample_id in sample_to_red_team:
            red_team_instruction = sample_to_red_team[sample_id]
            if red_team_instruction in rephrase_instructions:
                rephrases = rephrase_instructions[red_team_instruction]['rephrases']
                total_expected += 1 + len(rephrases)  # Red team instruction + rephrases
    
    print(f"Expected total datapoints: {total_expected}")
    
    # Process each sample
    with tqdm(total=total_expected, desc="Generating actions") as pbar:
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
                continue
            
            # Check if we have a red team instruction for this sample
            if sample_id not in sample_to_red_team:
                print(f"Warning: No red team instruction found for sample_id: {sample_id}")
                continue
            
            red_team_instruction = sample_to_red_team[sample_id]
            
            # Check if we have rephrases for this red team instruction
            if red_team_instruction not in rephrase_instructions:
                print(f"Warning: No rephrases found for red team instruction: '{red_team_instruction}'")
                continue
            
            instruction_data = rephrase_instructions[red_team_instruction]
            rephrases = instruction_data['rephrases']
            
            # Prepare all instructions (red team instruction + rephrases)
            all_instructions = [red_team_instruction] + rephrases
            
            try:
                # Get actions for all instructions at once
                output_ids, generated_actions = get_batch_actions(
                    instructions=all_instructions,
                    image_path=openvla_image_path,
                    temperature=0.0
                )
                
                # Process each generated action
                for instr_idx, (instruction, generated_action, output_id) in enumerate(zip(all_instructions, generated_actions, output_ids)):
                    # Calculate NRMSE
                    nrmse = calculate_nrmse(ground_truth_action, generated_action)
                    
                    # Create result entry
                    result_entry = {
                        'sample_id': sample_id,
                        'sample_index': sample_idx,
                        'instruction_index': instr_idx,
                        'is_red_team_original': instr_idx == 0,
                        'instruction': instruction,
                        'original_instruction': original_instruction,
                        'red_team_instruction': red_team_instruction,
                        'ground_truth_action': ground_truth_action.tolist(),
                        'generated_action': generated_action.tolist(),
                        'output_ids': output_id.tolist(),
                        'nrmse': float(nrmse),
                        'image_path': openvla_image_path,
                        'episode_id': sample['episode_id'],
                        'timestep': sample['timestep']
                    }
                    
                    results.append(result_entry)
                    pbar.update(1)
                    
            except Exception as e:
                print(f"Error processing sample {sample_id}: {e}")
                # Skip this sample and update progress bar for all expected instructions
                expected_instructions = 1 + len(rephrases)
                pbar.update(expected_instructions)
                continue
    
    print(f"\nGenerated {len(results)} datapoints")
    
    # Save results to JSON file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"red_team_openvla_actions_{timestamp}.json"
    
    # Prepare final data structure
    final_data = {
        'metadata': {
            'timestamp': timestamp,
            'total_datapoints': len(results),
            'expected_datapoints': total_expected,
            'source_bridge_file': 'bridge_samples.json',
            'source_red_team_csv': 'q4_random_instructions.csv',
            'source_rephrase_file': rephrase_data.get('timestamp', 'unknown'),
            'model': 'OpenVLA',
            'temperature': 0,
            'api_endpoint': 'http://localhost:3200',
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
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    nrmse_values = [r['nrmse'] for r in results]
    red_team_original_nrmse = [r['nrmse'] for r in results if r['is_red_team_original']]
    rephrase_nrmse = [r['nrmse'] for r in results if not r['is_red_team_original']]
    
    print(f"Total datapoints: {len(results)}")
    print(f"Red team instruction datapoints: {len(red_team_original_nrmse)}")
    print(f"Rephrased instruction datapoints: {len(rephrase_nrmse)}")
    print(f"Overall NRMSE - Mean: {np.mean(nrmse_values):.4f}, Std: {np.std(nrmse_values):.4f}")
    if red_team_original_nrmse:
        print(f"Red Team NRMSE - Mean: {np.mean(red_team_original_nrmse):.4f}, Std: {np.std(red_team_original_nrmse):.4f}")
    if rephrase_nrmse:
        print(f"Rephrase NRMSE - Mean: {np.mean(rephrase_nrmse):.4f}, Std: {np.std(rephrase_nrmse):.4f}")
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    main()
