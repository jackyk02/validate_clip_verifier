#!/usr/bin/env python3
"""
Generate VLA-CLIP similarity scores for 12,800 Gaussian-augmented datapoints.
Uses the VLA-CLIP-bridge verifier to score the combination of:
- Bridge dataset images (_clip.jpg format)
- Original instructions (no rephrases, only Gaussian-augmented actions)
- Last 9 history actions + predicted Gaussian-augmented OpenVLA action
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
    with open('../bridge_samples.json', 'r') as f:
        bridge_data = json.load(f)
    print(f"Loaded {len(bridge_data['samples'])} bridge samples")
    
    # Load Gaussian-augmented OpenVLA actions
    with open('bridge_openvla_actions_gaussian_20250826_171110.json', 'r') as f:
        gaussian_data = json.load(f)
    print(f"Loaded {len(gaussian_data['results'])} Gaussian-augmented OpenVLA action predictions")
    
    return bridge_data, gaussian_data

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

def generate_vla_clip_scores():
    """Generate VLA-CLIP scores for all 12,800 Gaussian-augmented datapoints"""
    
    # Load data
    bridge_data, gaussian_data = load_data_files()
    
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
    total_expected = len(gaussian_data['results'])  # 12,800 Gaussian-augmented datapoints
    
    print(f"Generating VLA-CLIP scores for {total_expected} Gaussian-augmented datapoints...")
    
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
        last_9_history = bridge_sample['last_9_history_actions']
        
        # Get image path (use _clip.jpg format)
        image_filename = bridge_sample['state']['agent_view_image_file']
        image_path = os.path.join('../bridge_images', image_filename)
        
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
            print(f"Error computing similarity for sample {sample_id}, instruction_index {gaussian_result['instruction_index']}: {e}")
            similarity_score = None
        
        # Store result (following same format as original but with Gaussian-specific fields)
        result = {
            'sample_id': sample_id,
            'sample_index': gaussian_result['sample_index'],
            'instruction_index': gaussian_result['instruction_index'],
            'is_original': gaussian_result['is_original'],  # Always False for Gaussian
            'instruction': instruction,
            'original_instruction': gaussian_result['original_instruction'],
            'vla_clip_similarity_score': similarity_score,
            'predicted_action': predicted_action,
            'ground_truth_action': gaussian_result['ground_truth_action'],
            'openvla_nrmse': gaussian_result['nrmse'],
            'image_filename': image_filename,
            'image_path': gaussian_result['image_path'],
            'episode_id': gaussian_result['episode_id'],
            'timestep': gaussian_result['timestep'],
            # Gaussian-specific metadata
            'batch_mean': gaussian_result['batch_mean'],
            'batch_variance': gaussian_result['batch_variance'],
            'batch_size': gaussian_result['batch_size'],
            'generation_method': gaussian_result['generation_method']
        }
        
        results.append(result)
    
    print(f"Generated {len(results)} VLA-CLIP scores out of {total_expected} expected")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'bridge_vla_clip_scores_gaussian_{timestamp}.json'
    
    output_data = {
        'metadata': {
            'timestamp': timestamp,
            'total_scores_generated': len(results),
            'expected_total': total_expected,
            'source_bridge_file': 'bridge_samples.json',
            'source_gaussian_file': 'bridge_openvla_actions_gaussian_20250826_171110.json',
            'vla_clip_model_path': model_path,
            'use_transformer': True,
            'history_length': 10,
            'action_dimensions': 7,
            'image_format': '_clip.jpg',
            'method': 'gaussian_sampling',
            'gaussian_metadata': gaussian_data['metadata'],
            'description': 'VLA-CLIP similarity scores for bridge dataset with Gaussian-augmented OpenVLA predictions'
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
        nrmse_values = [r['openvla_nrmse'] for r in results if r['openvla_nrmse'] is not None]
        
        if scores:
            print(f"\nSummary Statistics:")
            print(f"Valid scores: {len(scores)}/{len(results)}")
            print(f"Mean VLA-CLIP similarity score: {np.mean(scores):.4f}")
            print(f"Std VLA-CLIP similarity score: {np.std(scores):.4f}")
            print(f"Min VLA-CLIP similarity score: {np.min(scores):.4f}")
            print(f"Max VLA-CLIP similarity score: {np.max(scores):.4f}")
        
        if nrmse_values:
            print(f"Mean NRMSE: {np.mean(nrmse_values):.4f}")
            print(f"Std NRMSE: {np.std(nrmse_values):.4f}")
            print(f"Min NRMSE: {np.min(nrmse_values):.4f}")
            print(f"Max NRMSE: {np.max(nrmse_values):.4f}")
    
    return output_data

if __name__ == "__main__":
    # Check if required files exist
    required_files = [
        '../bridge_samples.json',
        'bridge_openvla_actions_gaussian_20250826_171110.json'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print("Error: Missing required files:")
        for f in missing_files:
            print(f"  - {f}")
        exit(1)
    
    # Check if VLA-CLIP model exists
    model_path = '/root/vla-clip/bridge_verifier/bridge_rephrases_epoch_20.pt'
    if not os.path.exists(model_path):
        print(f"Error: VLA-CLIP model not found at {model_path}")
        print("Please ensure the model file exists or update the path.")
        exit(1)
    
    # Run scoring
    generate_vla_clip_scores()
