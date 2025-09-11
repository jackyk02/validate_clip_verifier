#!/usr/bin/env python3
"""
Generate OpenVLA output_ids for each sample's instruction and rephrases,
recording their output_ids as a list for each sample.
The first value in the list is for the original instruction, followed by rephrases.
"""

import requests
import json
import numpy as np
import os
import argparse
from tqdm import tqdm
from datetime import datetime

def get_batch_actions(instructions, image_paths, temperature=0.0):
    """
    Get batch output_ids for multiple instructions using OpenVLA.
    
    Args:
        instructions: List of instruction strings
        image_paths: List of image file paths (one per instruction)
        temperature: Temperature for sampling
    
    Returns:
        Tuple of (output_ids, actions) as numpy arrays
    """
    # Convert to absolute paths
    image_paths = [os.path.abspath(path) for path in image_paths]
    
    payload = {
        "instructions": instructions,
        "image_paths": image_paths,
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
    """Load augmented instructions"""
    # Load augmented instructions (with rephrases)
    with open('augmented_instructions.json', 'r') as f:
        augmented_data = json.load(f)
    
    return augmented_data

def process_batch(batch_samples):
    """
    Process a batch of samples together for efficiency by making a single API call.
    
    Args:
        batch_samples: List of augmented samples to process
    
    Returns:
        List of result entries for the batch
    """
    batch_results = []
    
    # Collect metadata for valid samples
    valid_samples_metadata = []
    
    for aug_sample in batch_samples:
        sample_id = aug_sample['sample_id']
        original_instruction = aug_sample['instruction']
        rephrases = aug_sample['rephrases']
        
        # Get image path
        image_filename = f"{sample_id}.jpg"
        image_path = f"bridge_images_openvla/{image_filename}"
        
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue
        
        # Store metadata for processing
        valid_samples_metadata.append({
            'sample_id': sample_id,
            'original_instruction': original_instruction,
            'rephrases': rephrases,
            'image_path': image_path
        })
    
    if not valid_samples_metadata:
        return batch_results
    
    # Collect all instructions and corresponding image paths for single batch API call
    all_instructions = []
    all_image_paths = []
    instruction_to_sample_mapping = []  # Maps instruction index to (sample_idx, instruction_idx_within_sample)
    
    for sample_idx, meta in enumerate(valid_samples_metadata):
        sample_instructions = [meta['original_instruction']] + meta['rephrases']
        
        for instr_idx, instruction in enumerate(sample_instructions):
            all_instructions.append(instruction)
            all_image_paths.append(meta['image_path'])
            instruction_to_sample_mapping.append((sample_idx, instr_idx))
    
    try:
        # Make single batch API call for all instructions across all samples
        output_ids, generated_actions = get_batch_actions(
            instructions=all_instructions,
            image_paths=all_image_paths,
            temperature=0.0
        )
        
        # Group results back by sample
        sample_results = {}
        for result_idx, (sample_idx, instr_idx) in enumerate(instruction_to_sample_mapping):
            if sample_idx not in sample_results:
                sample_results[sample_idx] = {
                    'instruction_indices': []
                }
            
            sample_results[sample_idx]['instruction_indices'].append(instr_idx)
        
        # Process results for each sample
        for sample_idx, meta in enumerate(valid_samples_metadata):
            if sample_idx not in sample_results:
                print(f"Warning: No results found for sample {meta['sample_id']}")
                continue
            
            sample_result = sample_results[sample_idx]
            
            # Sort by instruction index to maintain order (original first, then rephrases)
            sorted_indices = sorted(range(len(sample_result['instruction_indices'])), 
                                  key=lambda i: sample_result['instruction_indices'][i])
            
            # Collect output_ids in order
            output_ids_list = []
            
            for sort_idx in sorted_indices:
                result_idx = None
                # Find the corresponding result index for this sorted index
                for r_idx, (s_idx, i_idx) in enumerate(instruction_to_sample_mapping):
                    if s_idx == sample_idx and sample_result['instruction_indices'][sort_idx] == i_idx:
                        result_idx = r_idx
                        break
                
                if result_idx is not None:
                    output_ids_list.append(output_ids[result_idx].tolist())
            
            # Create result entry for this sample
            result_entry = {
                'sample_id': meta['sample_id'],
                'original_instruction': meta['original_instruction'],
                'output_ids_list': output_ids_list  # First is original, rest are rephrases
            }
            
            batch_results.append(result_entry)
            
    except Exception as e:
        print(f"Error processing batch: {e}")
        return batch_results
    
    return batch_results

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate OpenVLA actions for instructions and rephrases')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Number of samples to process together in a single API call (default: 100)')
    parser.add_argument('--gpu-id', type=int, required=True,
                        help='GPU ID (0-7) to determine which data partition to process')
    args = parser.parse_args()
    
    print("Loading augmented instructions dataset...")
    augmented_data = load_data()
    
    print(f"Found {len(augmented_data)} samples with instructions and rephrases")
    
    # Validate GPU ID
    if args.gpu_id < 0 or args.gpu_id > 7:
        raise ValueError("GPU ID must be between 0 and 7")
    
    # Calculate data partitioning for this GPU
    total_samples = len(augmented_data)
    samples_per_gpu = 180000
    start_sample = args.gpu_id * samples_per_gpu
    end_sample = min(start_sample + samples_per_gpu, total_samples)
    
    print(f"Total samples in dataset: {total_samples}")
    print(f"GPU {args.gpu_id} processing samples {start_sample} to {end_sample-1} ({end_sample - start_sample} samples)")
    
    # Partition the data for this GPU - simple slice by sample index
    gpu_samples = augmented_data[start_sample:end_sample]
    
    print(f"GPU {args.gpu_id} will process {len(gpu_samples)} samples")
    print(f"Processing {args.batch_size} samples at a time")
    
    # Results list to store all sample results
    results = []
    gpu_expected_datapoints = sum(1 + len(item['rephrases']) for item in gpu_samples)
    
    print(f"Expected datapoints for GPU {args.gpu_id}: {gpu_expected_datapoints}")
    
    # Process samples in batches
    num_batches = (len(gpu_samples) + args.batch_size - 1) // args.batch_size
    with tqdm(total=num_batches, desc=f"Processing sample batches on GPU {args.gpu_id}") as pbar:
        for batch_start in range(0, len(gpu_samples), args.batch_size):
            batch_end = min(batch_start + args.batch_size, len(gpu_samples))
            batch_samples = gpu_samples[batch_start:batch_end]
            
            # Process this batch of samples
            batch_results = process_batch(batch_samples)
            results.extend(batch_results)
            
            pbar.update(1)
    
    print(f"\nProcessed {len(results)} samples successfully")
    
    # Save results to JSON file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"instruction_output_ids_gpu{args.gpu_id}_{timestamp}.json"
    
    # Calculate summary statistics
    all_original_output_ids = [r['output_ids_list'][0] for r in results]
    all_rephrase_output_ids = []
    for r in results:
        all_rephrase_output_ids.extend(r['output_ids_list'][1:])  # Skip first (original)
    
    # Prepare final data structure
    final_data = {
        'metadata': {
            'timestamp': timestamp,
            'gpu_id': args.gpu_id,
            'total_samples': len(results),
            'total_datapoints': sum(len(r['output_ids_list']) for r in results),
            'expected_datapoints': gpu_expected_datapoints,
            'sample_range': {
                'start': start_sample,
                'end': end_sample - 1,
                'count': end_sample - start_sample
            },
            'source_augmented_file': 'augmented_instructions.json',
            'model': 'OpenVLA',
            'temperature': 0,
            'batch_size': args.batch_size,
            'summary_stats': {
                'original_output_ids': {
                    'count': len(all_original_output_ids)
                },
                'rephrase_output_ids': {
                    'count': len(all_rephrase_output_ids)
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
    print(f"SUMMARY - GPU {args.gpu_id}")
    print("="*60)
    
    print(f"GPU ID: {args.gpu_id}")
    print(f"Sample range: {start_sample} to {end_sample-1} ({end_sample - start_sample} samples)")
    print(f"Total samples processed: {len(results)}")
    print(f"Total datapoints: {sum(len(r['output_ids_list']) for r in results)}")
    print(f"Original instruction datapoints: {len(all_original_output_ids)}")
    print(f"Rephrased instruction datapoints: {len(all_rephrase_output_ids)}")
    print(f"Results saved to: {output_file}")
    
    # Show example of first few samples
    print(f"\nExample results (first 3 samples):")
    for i, result in enumerate(results[:3]):
        print(f"Sample {result['sample_id']}: {len(result['output_ids_list'])} output_ids lists")
        print(f"  Original: '{result['original_instruction']}'")
        print(f"  {len(result['output_ids_list']) - 1} rephrases")
        print(f"  First output_ids shape: {len(result['output_ids_list'][0]) if result['output_ids_list'] else 'N/A'}")

if __name__ == "__main__":
    main()
