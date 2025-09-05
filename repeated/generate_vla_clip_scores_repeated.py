#!/usr/bin/env python3
"""
Generate VLA-CLIP similarity scores for repeated OpenVLA actions data.
Uses the VLA-CLIP-bridge verifier to score the combination of:
- Bridge dataset images (_clip.jpg format)
- Rephrased instructions 
- Last 9 history actions + predicted OpenVLA action (with repeats)
"""

import torch
import numpy as np
import json
import os
from tqdm import tqdm
from datetime import datetime
from PIL import Image
import sys

# Add the VLA-CLIP inference path
sys.path.append('/root/vla-clip/bridge_verifier')
from vla_clip_inference_bridge import VLA_CLIP_Bridge_Inference

def load_data_files():
    """Load all required data files"""
    print("Loading data files...")
    
    # Load bridge samples
    with open('/root/validate_clip_verifier/bridge_samples.json', 'r') as f:
        bridge_data = json.load(f)
    print(f"Loaded {len(bridge_data['samples'])} bridge samples")
    
    # Load repeated OpenVLA actions
    with open('/root/validate_clip_verifier/bridge_openvla_actions_repeated_20250905_004406.json', 'r') as f:
        openvla_data = json.load(f)
    print(f"Loaded {len(openvla_data['results'])} repeated OpenVLA action predictions")
    
    return bridge_data, openvla_data

def create_action_history_with_prediction(last_9_history, predicted_action):
    """
    Combine last 9 history actions with predicted action to create input for verifier.
    The verifier expects a history of actions, so we use the 9 history actions + the predicted next action.
    """
    # Convert to numpy arrays
    history = np.array(last_9_history)
    prediction = np.array(predicted_action)
    
    # Create the full action sequence (last 9 + predicted next action)
    # Shape: (10, 7) for Bridge dataset (10 timesteps, 7 action dimensions)
    full_sequence = np.vstack([history, prediction.reshape(1, -1)])
    
    return full_sequence

def generate_vla_clip_scores_repeated():
    """Generate VLA-CLIP scores for all repeated action datapoints"""
    
    # Load data
    bridge_data, openvla_data = load_data_files()
    
    # Initialize VLA-CLIP model
    model_path = '/root/vla-clip/bridge_verifier/bridge_rephrases_epoch_20.pt'
    if not os.path.exists(model_path):
        print(f"Error: VLA-CLIP model not found at {model_path}")
        print("Please ensure the model file exists or update the path.")
        return
    
    print("Initializing VLA-CLIP model...")
    inference_model = VLA_CLIP_Bridge_Inference(
        model_path=model_path,
        history_length=10,  # 9 history + 1 predicted action
        use_transformer=True,  # Model was trained with transformer architecture
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    print("VLA-CLIP model loaded successfully")
    
    # Create mapping from bridge samples for fast lookup
    bridge_lookup = {}
    for sample in bridge_data['samples']:
        bridge_lookup[sample['sample_id']] = sample
    
    # Prepare results
    results = []
    openvla_results = openvla_data['results']
    total_expected = len(openvla_results)
    
    print(f"Generating VLA-CLIP scores for {total_expected} repeated action datapoints...")
    
    # Process each OpenVLA result
    for openvla_result in tqdm(openvla_results, desc="Processing repeated actions"):
        sample_id = openvla_result['sample_id']
        instruction = openvla_result['instruction']
        predicted_action = openvla_result['generated_action']
        
        # Get corresponding bridge sample
        if sample_id not in bridge_lookup:
            print(f"Warning: No bridge sample found for sample_id: {sample_id}")
            continue
        
        bridge_sample = bridge_lookup[sample_id]
        last_9_history = bridge_sample['last_9_history_actions']
        
        # Get image path (use _clip.jpg format)
        image_filename = bridge_sample['state']['agent_view_image_file']
        image_path = os.path.join('/root/validate_clip_verifier/bridge_images', image_filename)
        
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            continue
        
        # Create action history input for verifier (9 history + 1 predicted)
        action_history_input = create_action_history_with_prediction(
            last_9_history, predicted_action
        )
        
        # Get VLA-CLIP similarity score
        try:
            similarity_score = inference_model.get_history_score(
                image=image,
                instruction=instruction,
                action_history=action_history_input
            )
            
            # Convert tensor to float if needed
            if torch.is_tensor(similarity_score):
                similarity_score = similarity_score.item()
            
        except Exception as e:
            print(f"Error computing similarity for sample {sample_id}, instruction '{instruction}': {e}")
            similarity_score = None
        
        # Store result - include all fields from original OpenVLA result plus VLA-CLIP score
        result = {
            'sample_id': sample_id,
            'sample_index': openvla_result['sample_index'],
            'instruction_index': openvla_result['instruction_index'],
            'repeat_index': openvla_result['repeat_index'],
            'is_original': openvla_result['is_original'],
            'instruction': instruction,
            'original_instruction': openvla_result['original_instruction'],
            'vla_clip_similarity_score': similarity_score,
            'predicted_action': predicted_action,
            'ground_truth_action': openvla_result['ground_truth_action'],
            'openvla_nrmse': openvla_result['nrmse'],
            'output_ids': openvla_result['output_ids'],
            'image_filename': image_filename,
            'episode_id': bridge_sample['episode_id'],
            'timestep': bridge_sample['timestep']
        }
        
        results.append(result)
    
    print(f"Generated {len(results)} VLA-CLIP scores out of {total_expected} expected")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'bridge_vla_clip_scores_repeated_{timestamp}.json'
    
    output_data = {
        'metadata': {
            'timestamp': timestamp,
            'total_scores_generated': len(results),
            'expected_total': total_expected,
            'num_repeats_per_instruction': openvla_data['metadata']['num_repeats_per_instruction'],
            'source_bridge_file': 'bridge_samples.json',
            'source_openvla_file': 'bridge_openvla_actions_repeated_20250905_004406.json',
            'source_openvla_timestamp': openvla_data['metadata']['timestamp'],
            'vla_clip_model_path': model_path,
            'use_transformer': True,
            'history_length': 10,
            'action_dimensions': 7,
            'image_format': '_clip.jpg',
            'openvla_temperature': openvla_data['metadata']['temperature'],
            'description': 'VLA-CLIP similarity scores for bridge dataset with repeated OpenVLA predictions (temperature=1.0)'
        },
        'results': results
    }
    
    print(f"Saving results to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print("VLA-CLIP scoring completed successfully!")
    
    # Print summary statistics
    if results:
        scores = [r['vla_clip_similarity_score'] for r in results if r['vla_clip_similarity_score'] is not None]
        if scores:
            print(f"\nSummary Statistics:")
            print(f"Valid scores: {len(scores)}/{len(results)}")
            print(f"Mean similarity score: {np.mean(scores):.4f}")
            print(f"Std similarity score: {np.std(scores):.4f}")
            print(f"Min similarity score: {np.min(scores):.4f}")
            print(f"Max similarity score: {np.max(scores):.4f}")
            
            # Statistics by original vs rephrase
            original_scores = [r['vla_clip_similarity_score'] for r in results 
                             if r['vla_clip_similarity_score'] is not None and r['is_original']]
            rephrase_scores = [r['vla_clip_similarity_score'] for r in results 
                             if r['vla_clip_similarity_score'] is not None and not r['is_original']]
            
            if original_scores:
                print(f"Original instruction scores - Mean: {np.mean(original_scores):.4f}, Std: {np.std(original_scores):.4f}")
            if rephrase_scores:
                print(f"Rephrase instruction scores - Mean: {np.mean(rephrase_scores):.4f}, Std: {np.std(rephrase_scores):.4f}")
    
    return output_data

if __name__ == "__main__":
    # Check if required files exist
    required_files = [
        '/root/validate_clip_verifier/bridge_samples.json',
        '/root/validate_clip_verifier/bridge_openvla_actions_repeated_20250905_004406.json'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print("Error: Missing required files:")
        for f in missing_files:
            print(f"  - {f}")
        exit(1)
    
    # Run scoring
    generate_vla_clip_scores_repeated()
