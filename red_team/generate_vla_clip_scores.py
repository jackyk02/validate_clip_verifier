#!/usr/bin/env python3
"""
Generate VLA-CLIP similarity scores for red team instruction datapoints (128 rephrased instructions × 100 datapoints).
Uses the VLA-CLIP-bridge verifier to score the combination of:
- Bridge dataset images (_clip.jpg format)
- Red team rephrased instructions 
- Last 9 history actions + predicted OpenVLA action
"""

import torch
import numpy as np
import json
import os
import pandas as pd
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
    
    # Load red team rephrased instructions
    with open('red_team_instruction_rephrases_20250909_033446.json', 'r') as f:
        rephrase_data = json.load(f)
    print(f"Loaded rephrased instructions for {len(rephrase_data['instructions'])} red team instructions")
    
    # Load red team OpenVLA actions
    with open('red_team_openvla_actions_20250909_052140.json', 'r') as f:
        openvla_data = json.load(f)
    print(f"Loaded {len(openvla_data['results'])} red team OpenVLA action predictions")
    
    # Load red team instruction mapping
    csv_file = 'q4_random_instructions.csv'
    df = pd.read_csv(csv_file)
    sample_to_red_team = {}
    
    for _, row in df.iterrows():
        sample_id = int(row['sample_id'])
        red_team_instruction = str(row['red_team_instruction'])
        sample_to_red_team[sample_id] = red_team_instruction
    
    print(f"Loaded {len(sample_to_red_team)} sample-to-red-team mappings from CSV")
    
    return bridge_data, rephrase_data, openvla_data, sample_to_red_team

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
    """Generate VLA-CLIP scores for all red team datapoints"""
    
    # Load data
    bridge_data, rephrase_data, openvla_data, sample_to_red_team = load_data_files()
    
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
    
    # Create mapping from OpenVLA results for fast lookup
    openvla_lookup = {}
    for result in openvla_data['results']:
        key = (result['sample_id'], result['instruction_index'])
        openvla_lookup[key] = result
    
    # Prepare results
    results = []
    # Calculate expected total based on available data
    total_expected = len(openvla_data['results'])
    
    print(f"Generating VLA-CLIP scores for {total_expected} datapoints...")
    
    # Process each OpenVLA result directly (since they already contain the right combinations)
    for openvla_result in tqdm(openvla_data['results'], desc="Processing OpenVLA results"):
        sample_id = openvla_result['sample_id']
        instruction_idx = openvla_result['instruction_index']
        instruction = openvla_result['instruction']
        original_instruction = openvla_result['original_instruction']
        red_team_instruction = openvla_result['red_team_instruction']
        predicted_action = openvla_result['generated_action']
        
        # Find corresponding bridge sample
        bridge_sample = None
        for sample in bridge_data['samples']:
            if sample['sample_id'] == sample_id:
                bridge_sample = sample
                break
        
        if bridge_sample is None:
            print(f"Warning: No bridge sample found for sample_id {sample_id}")
            continue
        
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
            print(f"Error computing similarity for sample {sample_id}, instruction {instruction_idx}: {e}")
            similarity_score = None
        
        # Store result
        result = {
            'sample_id': sample_id,
            'instruction_index': instruction_idx,
            'is_red_team_original': openvla_result['is_red_team_original'],
            'instruction': instruction,
            'original_instruction': original_instruction,
            'red_team_instruction': red_team_instruction,
            'vla_clip_similarity_score': similarity_score,
            'predicted_action': predicted_action,
            'ground_truth_action': bridge_sample['current_ground_truth_action'],
            'openvla_nrmse': openvla_result['nrmse'],
            'image_filename': image_filename,
            'episode_id': bridge_sample['episode_id'],
            'timestep': bridge_sample['timestep']
        }
        
        results.append(result)
    
    print(f"Generated {len(results)} VLA-CLIP scores out of {total_expected} expected")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'red_team_vla_clip_scores_{timestamp}.json'
    
    output_data = {
        'metadata': {
            'timestamp': timestamp,
            'total_scores_generated': len(results),
            'expected_total': total_expected,
            'source_bridge_file': 'bridge_samples.json',
            'source_red_team_csv': 'q4_random_instructions.csv',
            'source_rephrase_file': 'red_team_instruction_rephrases_20250909_033446.json',
            'source_openvla_file': 'red_team_openvla_actions_20250909_052140',
            'vla_clip_model_path': model_path,
            'use_transformer': True,
            'history_length': 10,
            'action_dimensions': 7,
            'image_format': '_clip.jpg',
            'description': 'VLA-CLIP similarity scores for bridge dataset with red team rephrased instructions and OpenVLA predictions'
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
    
    return output_data

if __name__ == "__main__":
    # Check if required files exist
    required_files = [
        '../bridge_samples.json',
        'q4_random_instructions.csv',
        'red_team_instruction_rephrases_20250909_033446.json',
        'red_team_openvla_actions_20250909_052140.json'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print("Error: Missing required files:")
        for f in missing_files:
            print(f"  - {f}")
        exit(1)
    
    # Run scoring
    generate_vla_clip_scores()
