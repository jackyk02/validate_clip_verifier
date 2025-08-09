# Bridge Dataset Extraction

This folder contains 100 extracted datapoints from the Bridge V2 dataset with preprocessing utilities.

## 📁 Folder Structure

```
bridge_dataset_extracted/
├── README.md                                        # This documentation file
├── bridge_samples.json                              # 100 extracted datapoints with metadata
├── bridge_instruction_rephrases_20250809_053826.json # 128 rephrases for each unique instruction
├── extract_bridge_samples.py                       # Main extraction script
├── preprocessing_utils.py                          # Image preprocessing utilities
├── rephrase_bridge_instructions_threaded.py       # Multithreaded instruction rephrasing script
├── bridge_images/                                  # Original images (100 files)
│   ├── 0_clip.jpg
│   ├── 1_clip.jpg
│   └── ... (up to 99_clip.jpg)
└── processed_images/                               # Preprocessed images
    ├── openvla/                                    # OpenVLA preprocessing (224x224)
    │   ├── 0_openvla.jpg
    │   ├── 1_openvla.jpg
    │   └── ... (up to 99_openvla.jpg)
    └── robomonkey/                                 # RoboMonkey preprocessing (224x224)
        ├── 0_robomonkey.jpg
        ├── 1_robomonkey.jpg
        └── ... (up to 99_robomonkey.jpg)
```

## 📊 Dataset Information

- **Total Samples**: 100 datapoints
- **Episodes Processed**: 100 episodes from Bridge V2 dataset
- **Sampling Strategy**: One random timestep per episode
- **Action Dimension**: 7 (position + rotation + gripper)
- **History Length**: 9 previous actions per sample
- **Image Formats**: Original + 2 preprocessed versions (224x224 each)
- **Random Seed**: 42 (for reproducible timestep selection)
- **Instruction Rephrases**: 128 rephrases for each of 93 unique instructions (11,776 total rephrases)

## 📋 Data Structure

Each sample in `bridge_samples.json` contains:

```json
{
  "sample_id": 0,
  "state": {
    "agent_view_image_file": "0_clip.jpg",
    "timestep": 0,
    "episode_id": 0
  },
  "original_instruction": "put small spoon from basket to tray",
  "last_9_history_actions": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], ...],
  "current_ground_truth_action": [x, y, z, rx, ry, rz, gripper],
  "episode_id": 0,
  "timestep": 0
}
```

## 🖼️ Image Preprocessing

- **Original Images** (`bridge_images/`): Raw images from Bridge dataset saved as `{N}_clip.jpg`
- **OpenVLA Format** (`processed_images/openvla/`): Preprocessed using `get_simpler_img` function → `{N}_openvla.jpg`
- **RoboMonkey Format** (`processed_images/robomonkey/`): Preprocessed using `process_image` function → `{N}_robomonkey.jpg`

All preprocessed images are exactly 224×224 pixels.

## 🚀 Usage

To extract more samples:
```bash
python extract_bridge_samples.py --num_samples 200 --output_json new_samples.json
```

To preprocess existing images:
```bash
python preprocessing_utils.py --image_folder path/to/images --output_folder path/to/output
```

To generate rephrases for instructions:
```bash
python rephrase_bridge_instructions_threaded.py
```

## 📝 Notes

- **Unique Episodes**: Each of the 100 samples comes from a different episode
- **Random Timesteps**: Timesteps are randomly selected from each episode (range: 0-88, average: 17.74)
- **Reproducible**: Using seed=42 ensures consistent timestep selection across runs
- **History Padding**: History actions are padded with zeros for early timesteps in episodes
- **High Quality**: Images are saved in high quality JPEG format (95% quality)
- **Preprocessing Fidelity**: All preprocessing maintains the same methods used in the original OpenVLA and RoboMonkey implementations
- **Rephrase Quality**: All rephrases are cleaned to be lowercase and without trailing punctuation
- **Multithreaded Processing**: Uses 8 threads for fast rephrase generation (4.3 minutes vs 40+ minutes)
