"""
Configuration parameters for ETL preprocessing and dataset generation.

Author: Sarvesh Telang
"""
# ==== CONFIGURABLE PARAMETERS ====
MAX_QA_PER_VIDEO = 1 # Max QA entries per video
MAX_FRAMES_PER_VIDEO = 10 # Frames to extract per video


# === INPUT PATHS ===
JSON_PATH = "BDD_instruct_train_main.json" # Instruction JSON file generated through generate_instruction_dataset.ipynb

# Raw video dataset from BDD100K
VIDEO_FOLDERS = [
r"data/bdd100k_videos_train_00/bdd100k/videos/train", # First part of BDD100K training videos
# r"data/bdd100k_videos_train_01/bdd100k/videos/train", # Additional parts can be added if needed
]

# === OUTPUT PATHS ===
OUTPUT_VIDEO_FOLDER = "dataset_train"
OUTPUT_IMAGE_FOLDER = "dataset_train/images"
FILTERED_JSON_PATH = "BDD_instruct_train_filtered.json"

# === HUGGING FACE CONFIG ===
# <Add your Hugging Face token here>
HF_REPO_NAME = "SFT_VLA_Dataset_1.0" # Replace with your desired Hugging Face repo name