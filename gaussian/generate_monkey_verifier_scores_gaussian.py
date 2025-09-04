#!/usr/bin/env python3
"""
Generate monkey-verifier scores for 12,800 Gaussian-augmented datapoints.
Uses the monkey-verifier API to score the combination of:
- Bridge dataset images (_robomonkey.jpg format)
- Original instructions (no rephrases, only Gaussian-augmented actions)
- Gaussian-augmented OpenVLA predicted actions
"""

import requests
import numpy as np
import json
import os
from tqdm import tqdm
from datetime import datetime

def load_data_files():
    """Load all required data files"""
    print("Loading data files...")
    
    # Load bridge samples
    with open('../bridge_samples.json', 'r') as f:
        bridge_data = json.load(f)
    print(f"Loaded {len(bridge_data['samples'])} bridge samples")
    
    # Load Gaussian-augmented OpenVLA actions
    with open('bridge_openvla_actions_gaussian_20250826_171110.json', 'r') as f:
        gaussian_data = json.load(f)
    print(f"Loaded {len(gaussian_data['results'])} Gaussian-augmented OpenVLA action predictions")
    
    return bridge_data, gaussian_data

def get_monkey_verifier_score(instruction, image_path, action_tokens):
    """
    Get score from monkey-verifier API
    
    Args:
        instruction (str): The instruction text
        image_path (str): Path to the preprocessed robomonkey image
        action_tokens (list): Tokenized action from OpenVLA output_ids
    
    Returns:
        float: Reward score from the verifier
    """
    url = "http://127.0.0.1:3100/process"
    
    payload = {
        "instruction": instruction,
        "image_path": image_path,
        "action": action_tokens
    }
    
    try:
        response = requests.post(url, data=json.dumps(payload), 
                               headers={'Content-Type': 'application/json'})
        response.raise_for_status()
        result = json.loads(response.text)
        return result['rewards']
    except Exception as e:
        print(f"Error calling monkey-verifier API: {e}")
        return None

def generate_monkey_verifier_scores_gaussian():
    """Generate monkey-verifier scores for all Gaussian-augmented datapoints"""
    
    # Load data
    bridge_data, gaussian_data = load_data_files()
    
    # Create mapping from bridge samples for fast lookup
    bridge_lookup = {}
    for sample in bridge_data['samples']:
        bridge_lookup[sample['sample_id']] = sample
    
    print(f"Created lookup table for {len(bridge_lookup)} bridge samples")
    
    # Prepare results
    results = []
    total_expected = len(gaussian_data['results'])  # 12,800 Gaussian-augmented datapoints
    
    print(f"Expected total datapoints: {total_expected}")
    print(f"Processing {total_expected} Gaussian-augmented samples with monkey-verifier...")
    
    # Process each Gaussian result
    for gaussian_result in tqdm(gaussian_data['results'], desc="Processing Gaussian datapoints"):
        sample_id = gaussian_result['sample_id']
        instruction = gaussian_result['instruction']
        predicted_action = gaussian_result['generated_action']
        
        # Get corresponding bridge sample
        if sample_id not in bridge_lookup:
            print(f"Warning: Bridge sample {sample_id} not found")
            continue
        
        bridge_sample = bridge_lookup[sample_id]
        
        # Get RoboMonkey image path (already preprocessed to 224x224)
        image_filename = bridge_sample['state']['agent_view_image_file']
        base_name = image_filename.split('_')[0]  # Extract number from "N_clip.jpg"
        robomonkey_image_path = os.path.abspath(f"../processed_images/robomonkey/{base_name}_robomonkey.jpg")
        
        if not os.path.exists(robomonkey_image_path):
            print(f"Warning: RoboMonkey image not found: {robomonkey_image_path}")
            continue
        
        # For Gaussian samples, we should have output_ids from the token conversion
        # If not available, we can skip this sample or use the action directly
        output_ids = gaussian_result.get('output_ids', [])
        if not output_ids:
            print(f"Warning: No output_ids found for Gaussian sample {sample_id}, instruction_index {gaussian_result['instruction_index']}")
            print("This sample was likely generated without token conversion. Skipping...")
            continue
        
        # Use the output_ids as tokenized action (wrap in list as expected by API)
        action_input = [output_ids]
        
        # Get monkey-verifier score
        try:
            verifier_score = get_monkey_verifier_score(
                instruction=instruction,
                image_path=robomonkey_image_path,
                action_tokens=action_input
            )
            
            # Extract single score if it's a list
            if isinstance(verifier_score, list) and len(verifier_score) > 0:
                verifier_score = verifier_score[0]
            
        except Exception as e:
            print(f"Error computing verifier score for sample {sample_id}, instruction_index {gaussian_result['instruction_index']}: {e}")
            verifier_score = None
        
        # Store result (following same format as original but with Gaussian-specific fields)
        result = {
            'sample_id': sample_id,
            'sample_index': gaussian_result['sample_index'],
            'instruction_index': gaussian_result['instruction_index'],
            'is_original': gaussian_result['is_original'],  # Always False for Gaussian
            'instruction': instruction,
            'original_instruction': gaussian_result['original_instruction'],
            'monkey_verifier_score': verifier_score,
            'predicted_action': predicted_action,
            'output_ids': output_ids,
            'ground_truth_action': gaussian_result['ground_truth_action'],
            'openvla_nrmse': gaussian_result['nrmse'],
            'image_filename': image_filename,
            'robomonkey_image_path': robomonkey_image_path,
            'episode_id': gaussian_result['episode_id'],
            'timestep': gaussian_result['timestep'],
            # Gaussian-specific metadata
            'batch_mean': gaussian_result['batch_mean'],
            'batch_variance': gaussian_result['batch_variance'],
            'batch_size': gaussian_result['batch_size'],
            'generation_method': gaussian_result['generation_method']
        }
        
        results.append(result)
    
    print(f"Generated {len(results)} monkey-verifier scores")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'bridge_monkey_verifier_scores_gaussian_{timestamp}.json'
    
    output_data = {
        'metadata': {
            'timestamp': timestamp,
            'total_scores_generated': len(results),
            'expected_total': total_expected,
            'source_bridge_file': 'bridge_samples.json',
            'source_gaussian_file': 'bridge_openvla_actions_gaussian_20250826_171110.json',
            'verifier_api_endpoint': 'http://127.0.0.1:3100/process',
            'image_format': '_robomonkey.jpg',
            'image_size': [224, 224],
            'action_tokenization': 'gaussian_augmented_actions_as_tokens',
            'method': 'gaussian_sampling',
            'gaussian_metadata': gaussian_data['metadata'],
            'description': 'Monkey-verifier scores for bridge dataset with Gaussian-augmented OpenVLA predictions'
        },
        'results': results
    }
    
    print(f"Saving results to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print("Monkey-verifier scoring completed successfully!")
    
    # Print summary statistics
    if results:
        scores = [r['monkey_verifier_score'] for r in results if r['monkey_verifier_score'] is not None]
        nrmse_values = [r['openvla_nrmse'] for r in results if r['openvla_nrmse'] is not None]
        
        if scores:
            print(f"\nSummary Statistics:")
            print(f"Valid scores: {len(scores)}/{len(results)}")
            print(f"Mean monkey-verifier score: {np.mean(scores):.4f}")
            print(f"Std monkey-verifier score: {np.std(scores):.4f}")
            print(f"Min monkey-verifier score: {np.min(scores):.4f}")
            print(f"Max monkey-verifier score: {np.max(scores):.4f}")
        else:
            print("No valid monkey-verifier scores generated!")
        
        if nrmse_values:
            print(f"Mean NRMSE: {np.mean(nrmse_values):.4f}")
            print(f"Std NRMSE: {np.std(nrmse_values):.4f}")
            print(f"Min NRMSE: {np.min(nrmse_values):.4f}")
            print(f"Max NRMSE: {np.max(nrmse_values):.4f}")
    
    return output_data

if __name__ == "__main__":
    # Check if required data files exist
    required_dirs = [
        '../bridge_images',
        '../processed_images/robomonkey'
    ]
    
    required_files = [
        '../bridge_samples.json',
        'bridge_openvla_actions_gaussian_20250826_171110.json'
    ]
    
    missing_items = []
    for item in required_dirs + required_files:
        if not os.path.exists(item):
            missing_items.append(item)
    
    if missing_items:
        print("Error: Missing required files/directories:")
        for item in missing_items:
            print(f"  - {item}")
        exit(1)
    
    print("All required files and directories found.")
    
    # Run scoring
    generate_monkey_verifier_scores_gaussian()
