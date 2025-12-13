"""
High-level pipeline controller for Video ETL preprocessing, VLA instruction dataset generation, and pushing to Hugging Face Hub.
- Runs ETL preprocessing on raw video dataset
- Builds Hugging Face dataset from filtered metadata and images
- Pushes dataset to Hugging Face Hub

Author: Sarvesh Telang
"""

import config
from etl_preprocessing import VideoETLProcessor
from huggingface_data_gen import HFDataGenerator


def run_etl():
    """Runs the video ETL preprocessing pipeline."""
    processor = VideoETLProcessor()
    processor.run()


def run_dataset_generation():
    """Loads filtered metadata and builds the dataset."""
    generator = HFDataGenerator(
        json_path=config.FILTERED_JSON_PATH,
        image_folder=config.OUTPUT_IMAGE_FOLDER,
        max_frames=config.MAX_FRAMES_PER_VIDEO
    )

    generator.load_data()
    generator.build_dataset()
    return generator


def push_to_huggingface(generator):
    """Logs in and pushes dataset to Hugging Face Hub."""
    hf_token = config.HF_TOKEN
    generator.login_hf(token=hf_token)
    generator.push_to_hub(
        repo_name=config.HF_REPO_NAME,
        token=hf_token
    )


def main():
    """High-level pipeline controller."""
    print("Starting ETL preprocessing...")
    run_etl() # comment out if ETL already done

    print("Building dataset...")
    generator = run_dataset_generation() # comment out if dataset already built

    print("Pushing dataset to Hugging Face Hub...")
    push_to_huggingface(generator) 

    print("Done!")


if __name__ == "__main__":
    main()