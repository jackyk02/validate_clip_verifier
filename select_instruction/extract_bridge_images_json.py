import argparse
import os
import numpy as np
from tqdm import tqdm
import json
import tensorflow as tf
import tensorflow_datasets as tfds
from preprocessing_utils import process_image_openvla
from PIL import Image
import tempfile


def normalize_instruction(instr):
    """
    Normalize instruction by removing trailing punctuation and converting to lowercase.
    Returns None if instruction is empty after normalization.
    """
    if not instr or not isinstance(instr, str):
        return None
    
    instr = instr.strip().lower()
    # Remove any trailing punctuation (., !, ?)
    import re
    instr = re.sub(r'[.?!]+$', '', instr).strip()
    
    # Return None if instruction is empty after normalization
    if not instr:
        return None
    
    return instr


def extract_bridge_images_and_json(builder_dir, episode_ids, output_folder, 
                                   max_episodes=None, json_filename="instructions.json"):
    """
    Extract Bridge V2 dataset images and instructions, preprocessing images with OpenVLA method.
    
    Args:
        builder_dir (str): Path to the Bridge V2 dataset directory
        episode_ids (list): List of episode IDs to process
        output_folder (str): Path to folder where preprocessed images and JSON will be saved
        max_episodes (int): Maximum number of episodes to process (for debugging)
        json_filename (str): Name of the JSON file to save instructions
    """
    
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")
    
    # Initialize dataset builder
    print(f"Loading Bridge V2 dataset from {builder_dir}...")
    builder = tfds.builder_from_directory(builder_dir=builder_dir)
    
    # Determine episode range
    if episode_ids is None:
        # Get dataset info to determine total number of episodes
        info = builder.info
        total_episodes = info.splits['train'].num_examples
        if max_episodes:
            total_episodes = min(total_episodes, max_episodes)
        episode_ids = list(range(total_episodes))
    else:
        if max_episodes:
            episode_ids = episode_ids[:max_episodes]
    
    print(f"Processing {len(episode_ids)} episodes...")
    
    # Initialize counters and data storage
    sample_counter = 0
    total_episodes_processed = 0
    json_data = []
    
    # Process episodes
    for episode_id in tqdm(episode_ids, desc="Processing episodes"):
        try:
            # Load single episode
            ds = builder.as_dataset(split=f"train[{episode_id}:{episode_id + 1}]")
            episode = next(iter(ds))
            
            # Extract language instruction (assume same for all steps in episode)
            steps = list(episode["steps"])
            if not steps:
                continue
                
            language_instruction = steps[0]["language_instruction"].numpy().decode()
            original_instruction = normalize_instruction(language_instruction)
            
            # Skip episodes with empty or invalid instructions
            if original_instruction is None:
                print(f"Warning: Empty instruction in episode {episode_id}, skipping...")
                continue
            
            # Process each step in the episode
            for step_idx, step in enumerate(steps):
                observation = step["observation"]
                
                # Extract agent view image (Over-the-shoulder RGBD - main view)
                # Try different possible image keys
                agent_view_image = None
                for img_key in ["image_0", "rgb", "image", "camera_0"]:
                    if img_key in observation:
                        agent_view_image = observation[img_key].numpy()
                        break
                
                if agent_view_image is None:
                    print(f"Warning: No agent view image found in episode {episode_id} step {step_idx}, skipping...")
                    continue
                
                # Prepare image for processing
                if agent_view_image.dtype != np.uint8:
                    # Normalize to 0-255 range if needed
                    if agent_view_image.max() <= 1.0:
                        agent_view_image = (agent_view_image * 255).astype(np.uint8)
                    else:
                        agent_view_image = agent_view_image.astype(np.uint8)
                
                # Handle RGBD (4 channels) by taking only RGB
                if agent_view_image.shape[-1] == 4:
                    agent_view_image = agent_view_image[..., :3]
                
                # Save original image temporarily
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                    temp_path = temp_file.name
                    Image.fromarray(agent_view_image).save(temp_path, "JPEG", quality=95)
                
                # Process image with OpenVLA preprocessing
                output_image_path = os.path.join(output_folder, f"{sample_counter}.jpg")
                try:
                    process_image_openvla(temp_path, output_image_path, resize_size=224)
                    
                    # Add to JSON data
                    json_data.append({
                        'sample_id': sample_counter,
                        'instruction': original_instruction
                    })
                    
                    sample_counter += 1
                    
                except Exception as e:
                    print(f"Error processing image for episode {episode_id} step {step_idx}: {e}")
                finally:
                    # Clean up temporary file
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
            
            total_episodes_processed += 1
            
        except Exception as e:
            print(f"Error processing episode {episode_id}: {e}")
            continue
    
    if total_episodes_processed == 0:
        raise ValueError("No valid episodes found in the specified dataset.")
    
    print(f"\nProcessed {total_episodes_processed} episodes successfully.")
    print(f"Generated {sample_counter} preprocessed images and instruction pairs.")
    
    # Save JSON file
    json_path = os.path.join(output_folder, json_filename)
    print(f"Saving instructions JSON to {json_path}...")
    
    with open(json_path, 'w', encoding='utf-8') as jsonfile:
        json.dump(json_data, jsonfile, indent=2, ensure_ascii=False)
    
    print(f"Done! Saved {len(json_data)} instruction pairs to JSON.")
    print(f"Images saved to: {output_folder}")
    print(f"JSON saved to: {json_path}")
    
    return sample_counter, json_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract Bridge V2 dataset images with OpenVLA preprocessing and save instructions as JSON.')
    parser.add_argument('--builder_dir', type=str, default='/root/tensorflow_datasets/bridge_orig/1.0.0',
                        help='Path to the Bridge V2 dataset directory')
    parser.add_argument('--episode_ids', nargs='+', type=int, default=None,
                        help='Specific episode IDs to process (if not specified, processes all)')
    parser.add_argument('--output_folder', type=str, default='bridge_images_openvla',
                        help='Path to folder where preprocessed images and JSON will be saved')
    parser.add_argument('--max_episodes', type=int, default=None,
                        help='Maximum number of episodes to process (for debugging/testing)')
    parser.add_argument('--json_filename', type=str, default='instructions_to_sample_id.json',
                        help='Name of the JSON file to save instructions')
    
    args = parser.parse_args()
    
    # Extract dataset
    sample_count, json_path = extract_bridge_images_and_json(
        builder_dir=args.builder_dir,
        episode_ids=args.episode_ids,
        output_folder=args.output_folder,
        max_episodes=args.max_episodes,
        json_filename=args.json_filename
    )
    
    print(f"\nExtraction complete!")
    print(f"Total samples: {sample_count}")
    print(f"Images folder: {args.output_folder}")
    print(f"JSON file: {json_path}")
