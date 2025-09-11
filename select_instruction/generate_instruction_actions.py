#!/usr/bin/env python3
"""
Generate OpenVLA actions for each sample's instruction and rephrases,
recording their NRMSE values as a list for each sample.
The first value in the list is for the original instruction, followed by rephrases.
"""

import requests
import json
import numpy as np
import os
import argparse
from tqdm import tqdm
from datetime import datetime

# Define the ranges for NRMSE calculation (from generate_vla_actions.py)
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

def load_data():
    """Load augmented instructions and ground truth actions"""
    # Load augmented instructions (with rephrases)
    with open('augmented_instructions.json', 'r') as f:
        augmented_data = json.load(f)
    
    # Load ground truth actions
    print("Loading ground truth actions (this may take a moment due to large file size)...")
    with open('unique_simple_groundtruth.json', 'r') as f:
        groundtruth_data = json.load(f)
    
    return augmented_data, groundtruth_data

def process_batch(batch_samples, groundtruth_sample_map):
    """
    Process a batch of samples together for efficiency.
    
    Args:
        batch_samples: List of augmented samples to process
        groundtruth_sample_map: Mapping from sample_id to ground truth data
    
    Returns:
        List of result entries for the batch
    """
    batch_results = []
    
    # Process each sample in the batch
    batch_metadata = []
    
    for aug_sample in batch_samples:
        sample_id = aug_sample['sample_id']
        original_instruction = aug_sample['instruction']
        rephrases = aug_sample['rephrases']
        
        # Find corresponding ground truth sample
        if sample_id not in groundtruth_sample_map:
            print(f"Warning: No ground truth sample found for sample_id {sample_id}")
            continue
        
        groundtruth_sample = groundtruth_sample_map[sample_id]
        ground_truth_action = np.array(groundtruth_sample['current_ground_truth_action'])
        
        # Get image path
        image_filename = f"{sample_id}.jpg"
        image_path = f"bridge_images_openvla/{image_filename}"
        
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue
        
        # Store metadata for processing
        batch_metadata.append({
            'sample_id': sample_id,
            'original_instruction': original_instruction,
            'rephrases': rephrases,
            'ground_truth_action': ground_truth_action,
            'image_path': image_path
        })
    
    if not batch_metadata:
        return batch_results
    
    # Process each sample with batch API calls
    for meta in batch_metadata:
        try:
            sample_instructions = [meta['original_instruction']] + meta['rephrases']
            
            # Get actions for all instructions of this sample at once
            output_ids, generated_actions = get_batch_actions(
                instructions=sample_instructions,
                image_path=meta['image_path'],
                temperature=0.0
            )
            
            # Calculate NRMSE for each instruction
            nrmse_list = []
            action_details = []
            
            for instr_idx, (instruction, generated_action, output_id) in enumerate(zip(sample_instructions, generated_actions, output_ids)):
                # Calculate NRMSE
                nrmse = calculate_nrmse(meta['ground_truth_action'], generated_action)
                nrmse_list.append(float(nrmse))
                
                # Store detailed action information
                action_details.append({
                    'instruction': instruction,
                    'is_original': instr_idx == 0,
                    'generated_action': generated_action.tolist(),
                    'output_ids': output_id.tolist(),
                    'nrmse': float(nrmse)
                })
            
            # Create result entry for this sample
            result_entry = {
                'sample_id': meta['sample_id'],
                'original_instruction': meta['original_instruction'],
                'ground_truth_action': meta['ground_truth_action'].tolist(),
                'nrmse_list': nrmse_list,  # First is original, rest are rephrases
                'num_rephrases': len(meta['rephrases']),
                'action_details': action_details,
                'image_path': meta['image_path']
            }
            
            batch_results.append(result_entry)
            
        except Exception as e:
            print(f"Error processing sample {meta['sample_id']}: {e}")
            continue
    
    return batch_results

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate OpenVLA actions for instructions and rephrases')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Number of samples to process together (default: 4)')
    args = parser.parse_args()
    
    print("Loading augmented instructions and ground truth dataset...")
    augmented_data, groundtruth_data = load_data()
    
    groundtruth_samples = groundtruth_data['extracted_samples']
    
    # Create a mapping from sample_id to ground truth sample for quick lookup
    groundtruth_sample_map = {sample['sample_id']: sample for sample in groundtruth_samples}
    
    print(f"Found {len(augmented_data)} samples with instructions and rephrases")
    print(f"Found {len(groundtruth_samples)} ground truth samples with actions")
    print(f"Processing {args.batch_size} samples at a time")
    
    # Results list to store all sample results
    results = []
    total_expected = sum(1 + len(item['rephrases']) for item in augmented_data)
    
    print(f"Expected total datapoints: {total_expected}")
    
    # Process samples in batches
    num_batches = (len(augmented_data) + args.batch_size - 1) // args.batch_size
    with tqdm(total=num_batches, desc="Processing sample batches") as pbar:
        for batch_start in range(0, len(augmented_data), args.batch_size):
            batch_end = min(batch_start + args.batch_size, len(augmented_data))
            batch_samples = augmented_data[batch_start:batch_end]
            
            # Process this batch of samples
            batch_results = process_batch(batch_samples, groundtruth_sample_map)
            results.extend(batch_results)
            
            pbar.update(1)
    
    print(f"\nProcessed {len(results)} samples successfully")
    
    # Save results to JSON file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"instruction_actions_nrmse_{timestamp}.json"
    
    # Calculate summary statistics
    all_original_nrmse = [r['nrmse_list'][0] for r in results]
    all_rephrase_nrmse = []
    for r in results:
        all_rephrase_nrmse.extend(r['nrmse_list'][1:])  # Skip first (original)
    
    # Prepare final data structure
    final_data = {
        'metadata': {
            'timestamp': timestamp,
            'total_samples': len(results),
            'total_datapoints': sum(len(r['nrmse_list']) for r in results),
            'expected_datapoints': total_expected,
            'source_augmented_file': 'augmented_instructions.json',
            'source_groundtruth_file': 'unique_simple_groundtruth.json',
            'model': 'OpenVLA',
            'temperature': 0,
            'api_endpoint': 'http://localhost:3200',
            'batch_size': args.batch_size,
            'nrmse_ranges': {
                'min_values': min_values.tolist(),
                'max_values': max_values.tolist(),
                'ranges': ranges.tolist()
            },
            'summary_stats': {
                'original_nrmse': {
                    'mean': float(np.mean(all_original_nrmse)),
                    'std': float(np.std(all_original_nrmse)),
                    'count': len(all_original_nrmse)
                },
                'rephrase_nrmse': {
                    'mean': float(np.mean(all_rephrase_nrmse)),
                    'std': float(np.std(all_rephrase_nrmse)),
                    'count': len(all_rephrase_nrmse)
                }
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
    
    print(f"Total samples processed: {len(results)}")
    print(f"Total datapoints: {sum(len(r['nrmse_list']) for r in results)}")
    print(f"Original instruction datapoints: {len(all_original_nrmse)}")
    print(f"Rephrased instruction datapoints: {len(all_rephrase_nrmse)}")
    print(f"Original NRMSE - Mean: {np.mean(all_original_nrmse):.4f}, Std: {np.std(all_original_nrmse):.4f}")
    print(f"Rephrase NRMSE - Mean: {np.mean(all_rephrase_nrmse):.4f}, Std: {np.std(all_rephrase_nrmse):.4f}")
    print(f"Results saved to: {output_file}")
    
    # Show example of first few samples
    print(f"\nExample results (first 3 samples):")
    for i, result in enumerate(results[:3]):
        print(f"Sample {result['sample_id']}: NRMSE list = {[f'{x:.4f}' for x in result['nrmse_list']]}")
        print(f"  Original: '{result['original_instruction']}'")
        print(f"  {result['num_rephrases']} rephrases")

if __name__ == "__main__":
    main()
