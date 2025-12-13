"""
Hugging Face Dataset Generation Script for Vision-Language Assistant Dataset
- Loads filtered JSON metadata
- Builds Hugging Face dataset with images and QA pairs
- Pushes dataset to Hugging Face Hub

Author: Sarvesh Telang
"""

import json
from pathlib import Path
from datasets import Dataset, Features, Sequence, Value, Image, DatasetDict
from huggingface_hub import login
import config


class HFDataGenerator:
    def __init__(self, json_path: str, image_folder: str, max_frames: int):
        self.json_path = Path(json_path)
        self.image_folder = Path(image_folder)
        self.max_frames = max_frames
        self.dataset = None

    def load_data(self):
        """Load JSON data from file."""
        if not self.json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {self.json_path}")
        with open(self.json_path, "r", encoding="utf-8") as f:
            self.qa_items = json.load(f)
        print(f"Loaded {len(self.qa_items)} QA items.")

    def build_dataset(self):
        """Build Hugging Face dataset from JSON data and images."""
        rows = []
        for entry in self.qa_items:
            vid = entry["video_id"].replace(".mov", "")
            imgs = [
                str(self.image_folder / f"{vid}_frame_{i}.jpg")
                for i in range(1, self.max_frames + 1)
                if (self.image_folder / f"{vid}_frame_{i}.jpg").exists()
            ]
            rows.append({
                "images": imgs,
                "prompt": {"content": entry["QA"]["q"], "role": "user"},
                "completion": {"content": entry["QA"]["a"], "role": "assistant"},
            })

        features = Features({
            "images": Sequence(Image()),
            "prompt": {"content": Value("string"), "role": Value("string")},
            "completion": {"content": Value("string"), "role": Value("string")},
        })
        self.dataset = Dataset.from_list(rows, features=features)
        print(f"Dataset built with {len(self.dataset)} rows.")

        print(rows[0])  # Print the first row for verification

    def login_hf(self, token: str = None):
        """Login to Hugging Face Hub."""
        login(token=token)
        print("Logged in to Hugging Face Hub.")

    def push_to_hub(self, repo_name: str, token: str = None):
        """Push dataset to Hugging Face Hub."""
        if self.dataset is None:
            raise ValueError("Dataset not built yet. Call build_dataset() first.")
        self.dataset.push_to_hub(
            repo_name,
            token=token,
        )
        print(f"Dataset pushed to Hugging Face Hub: {repo_name}")


if __name__ == "__main__":
    # Initialize generator
    generator = HFDataGenerator(
        json_path=config.FILTERED_JSON_PATH,
        image_folder=config.OUTPUT_IMAGE_FOLDER,
        max_frames=config.MAX_FRAMES_PER_VIDEO
    )

    # Build dataset
    generator.load_data()
    generator.build_dataset()

    # hugging face login and push to hub
    HF_TOKEN = config.HF_TOKEN
    generator.login_hf(token=HF_TOKEN)
    generator.push_to_hub(repo_name= config.HF_REPO_NAME, token=HF_TOKEN)