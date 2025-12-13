"""
ETL Preprocessing Script for Video Dataset
- Filters JSON entries based on available videos
- Copies videos to output folder
- Extracts frames from videos with resolution stats
- Optionally resizes extracted images

Author: Sarvesh Telang
"""

import os
import json
import shutil
from collections import defaultdict, Counter
import cv2
import glob
from pathlib import Path
from tqdm import tqdm
import config


class VideoETLProcessor:
    """ ETL Processor for Video Dataset """
    def __init__(self):
        self.max_qa_per_video = config.MAX_QA_PER_VIDEO
        self.max_frames_per_video = config.MAX_FRAMES_PER_VIDEO
        self.json_path = config.JSON_PATH
        self.video_folders = config.VIDEO_FOLDERS
        self.output_video_folder = config.OUTPUT_VIDEO_FOLDER
        self.output_image_folder = config.OUTPUT_IMAGE_FOLDER
        self.filtered_json_path = config.FILTERED_JSON_PATH

        os.makedirs(self.output_video_folder, exist_ok=True)
        os.makedirs(self.output_image_folder, exist_ok=True)

    def load_json(self):
        """Load JSON data from the specified path."""
        with open(self.json_path, "r") as f:
            data = json.load(f)
        print(f"Total JSON entries: {len(data)}")
        return data

    def map_videos(self):
        """Map video filenames to their respective folders."""
        video_to_folder = {}
        for folder in self.video_folders:
            if not os.path.exists(folder):
                print(f"WARNING: folder does not exist: {folder}")
                continue
            for v in os.listdir(folder):
                video_to_folder[v] = folder
        print(f"Total videos found across folders: {len(video_to_folder)}")
        return video_to_folder

    def group_entries(self, data, video_to_folder):
        """Group JSON entries by video ID if the video exists in the mapped folders."""
        grouped = defaultdict(list)
        for entry in data:
            vid = entry["video_id"]
            if vid in video_to_folder:
                grouped[vid].append(entry)
        return grouped

    def filter_entries(self, grouped):
        """Filter JSON entries to limit the number of QA pairs per video."""
        filtered_entries = []
        for vid, entries in grouped.items():
            filtered_entries.extend(entries[: self.max_qa_per_video])
        print(f"Max QA entries per video: {self.max_qa_per_video}")
        print(f"Matched entries: {len(filtered_entries)}")

        with open(self.filtered_json_path, "w") as f:
            json.dump(filtered_entries, f, indent=4)
        print(f"Saved filtered JSON → {self.filtered_json_path}")

    def copy_videos(self, grouped, video_to_folder):
        """Copy videos from source folders to the output folder with progress bar."""
        for vid in tqdm(grouped.keys(), desc="Copying videos"):
            src = os.path.join(video_to_folder[vid], vid)
            dst = os.path.join(self.output_video_folder, vid)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
        print(f"Copied {len(grouped)} videos to {self.output_video_folder}/")

    def extract_frames(self):
        """Extract frames from videos, record resolution stats, and save them."""

        video_files = glob.glob(os.path.join(self.output_video_folder, "*.mov"))

        # Stats collectors
        resolution_counter = Counter()
        aspect_ratio_counter = Counter()

        all_saved_images = []  # list of file paths for later resizing

        for video_path in tqdm(video_files, desc=f"Extracting {self.max_frames_per_video} frames per video"):
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                tqdm.write(f"Error opening video: {video_path}")
                continue

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_indices = [
                int(total_frames * i / self.max_frames_per_video) 
                for i in range(self.max_frames_per_video)
            ]

            for idx, frame_num in tqdm(list(enumerate(frame_indices)), total=len(frame_indices),
                                    desc=f"Extracting {video_name}", leave=False):

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()

                if ret:
                    h, w = frame.shape[:2]
                    resolution_counter[(w, h)] += 1
                    aspect_ratio_counter[round(w/h, 4)] += 1

                    save_path = os.path.join(
                        self.output_image_folder, f"{video_name}_frame_{idx+1}.jpg"
                    )
                    cv2.imwrite(save_path, frame)
                    all_saved_images.append(save_path)

            cap.release()

        # Print summary
        print("\n===== Image Resolution Summary =====")
        for (w, h), count in resolution_counter.items():
            print(f"{w}×{h}  →  {count} images")

        print("\n===== Aspect Ratio Summary =====")
        for ratio, count in aspect_ratio_counter.items():
            print(f"{ratio}  →  {count} images")

        # Ask user for resize decision
        target_res = self.ask_resize_target()
        if target_res is not None:
            print(f"\nResizing all images to {target_res} ...")
            self.resize_images(all_saved_images, target_res)
            print("Resizing completed!")
    
    def ask_resize_target(self):
        print("\nEnter target resolution as WIDTH,HEIGHT (recommended --> 640,360)")
        print("Or press ENTER to keep original resolution.")

        user_input = input("Resolution: ").strip()
        if user_input == "":
            print("Keeping original resolution.")
            return None

        try:
            w, h = map(int, user_input.split(","))
            return (w, h)
        except:
            print("Invalid input. No resizing will be applied.")
            return None


    def resize_images(self, image_paths, target_res):
        w, h = target_res
        for img_path in tqdm(image_paths, desc="Resizing images"):
            img = cv2.imread(img_path)
            if img is None:
                continue
            resized = cv2.resize(img, (w, h))
            cv2.imwrite(img_path, resized)

    def verify_no_duplicates(self):
        """Verify that there are no duplicate video filenames in the output folder in case of max_qa_per_video being 1."""
        if self.max_qa_per_video == 1:
            files = [f for f in os.listdir(self.output_video_folder) if f.lower().endswith(".mov")]
            counter = Counter(files)
            duplicates = [name for name, count in counter.items() if count > 1]

            assert len(duplicates) == 0, f"Duplicate video filenames found: {duplicates}"

    def run(self):
        """Run the full ETL preprocessing pipeline."""
        data = self.load_json()
        video_map = self.map_videos()
        grouped = self.group_entries(data, video_map)
        self.filter_entries(grouped)
        self.copy_videos(grouped, video_map)
        self.extract_frames()
        self.verify_no_duplicates()
        print("ETL Preprocessing completed successfully!")


if __name__ == "__main__":
    processor = VideoETLProcessor()
    processor.run()
