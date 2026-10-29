import os
import shutil
import random
import re
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image

# --- PATH CONFIGURATIONS ---
BASE_DIR = Path(__file__).resolve().parent
SOURCE_CARDS_DIR = BASE_DIR / "training_playcards"
CSV_PATH = BASE_DIR / "breakdown-2024.csv"
OUTPUT_DATASET_DIR = BASE_DIR / "sortedcards"

# --- SPLIT RATIOS ---
TRAIN_PROP = 0.70
VAL_PROP = 0.20
TEST_PROP = 0.10

def is_bad_playcard(img_path):
    """Gentle visual filter to drop collapsed tracking images"""
    try:
        img = Image.open(img_path).convert("L")
        img_arr = np.array(img)
        active_pixels = np.where(img_arr > 25)
        if len(active_pixels[0]) < 10: 
            return True  
        if np.std(active_pixels[0]) < 2.5 or np.std(active_pixels[1]) < 2.5:
            return True
        return False
    except Exception:
        return True  

def main():
    print("==================================================")
    print("🎯🏈 CHRONOLOGICAL TACTICAL PLAY SORTING & SPLIT  ")
    print("==================================================")

    if not CSV_PATH.exists():
        print(f"❌ CSV file missing at: {CSV_PATH}")
        return

    try:
        # Read CSV keeping header row intact, auto-detect separator
        df_labels = pd.read_csv(CSV_PATH, sep=None, engine='python')
    except Exception as e:
        print(f"❌ Failed to read CSV: {e}")
        return

    # Targeting specific strategy names from columns
    label_column = "OFF PLAY"  
    fallback_column = "PLAY TYPE"

    # Grab all playcard files and ensure they are sorted alphabetically/chronologically
    all_cards = sorted(list(SOURCE_CARDS_DIR.glob("*_playcard.png")))
    if not all_cards:
        print(f"⚠️ No playcards found in {SOURCE_CARDS_DIR}")
        return

    print(f"[INFO] Found {len(all_cards)} images. Matching to {len(df_labels)} CSV rows sequentially...")

    # --- CLEANUP STEP: Delete old directory if it exists to ensure a fresh build ---
    if OUTPUT_DATASET_DIR.exists():
        print(f"🧹 Deleting existing target folder: '{OUTPUT_DATASET_DIR.name}/'...")
        shutil.rmtree(OUTPUT_DATASET_DIR)
        
    OUTPUT_DATASET_DIR.mkdir(parents=True, exist_ok=True)

    category_buckets = {}
    skipped_bad_cards = 0
    out_of_bounds_cards = 0

    # Match each card to the corresponding row index chronologically
    for index, card_path in enumerate(all_cards):
        if is_bad_playcard(card_path):
            skipped_bad_cards += 1
            continue

        # Prevent out-of-bounds index errors if there are more images than CSV rows
        if index >= len(df_labels):
            out_of_bounds_cards += 1
            continue

        # Extract matching row data sequentially
        row = df_labels.iloc[index]
        
        # Determine play label name with fallback check
        specific_play = row[label_column] if pd.notna(row[label_column]) else row[fallback_column]
        specific_play_name = str(specific_play).strip()

        # Sanitize name string for valid directory syntax
        folder_friendly_name = re.sub(r'[^a-zA-Z0-9_]', '_', specific_play_name)
        folder_friendly_name = re.sub(r'_+', '_', folder_friendly_name).strip('_')

        if not folder_friendly_name or folder_friendly_name.lower() == "nan":
            folder_friendly_name = "Unspecified_Play"

        if folder_friendly_name not in category_buckets:
            category_buckets[folder_friendly_name] = []
        category_buckets[folder_friendly_name].append(card_path)

    print(f"🛑 Filtered out {skipped_bad_cards} bad/collapsed playcards.")
    if out_of_bounds_cards > 0:
        print(f"⚠️ Skipped {out_of_bounds_cards} images because they exceeded the CSV row bounds.")

    # Distribute files into train/val/test split pools
    for play_class, file_list in category_buckets.items():
        random.shuffle(file_list)  
        total_files = len(file_list)
        
        if total_files == 0:
            continue
        
        # If a rare play type doesn't have enough files to split split-percentage wise, 
        # force them all into 'train' instead of breaking the math or dropping data
        if total_files < 3:
            target_dir = OUTPUT_DATASET_DIR / "train" / play_class
            target_dir.mkdir(parents=True, exist_ok=True)
            for source_file in file_list:
                shutil.copy(str(source_file), str(target_dir / source_file.name))
            continue

        train_end = int(total_files * TRAIN_PROP)
        val_end = train_end + int(total_files * VAL_PROP)

        splits = {
            "train": file_list[:train_end],
            "val": file_list[train_end:val_end],
            "test": file_list[val_end:]
        }

        for split_name, split_files in splits.items():
            target_dir = OUTPUT_DATASET_DIR / split_name / play_class
            target_dir.mkdir(parents=True, exist_ok=True)
            for source_file in split_files:
                shutil.copy(str(source_file), str(target_dir / source_file.name))

    print(f"\n🎉 [SUCCESS] Chronological dataset distribution complete!")
    print(f"📂 Output saved to completely clean folder: '{OUTPUT_DATASET_DIR}'")
    for split in ["train", "val", "test"]:
        split_count = len(list((OUTPUT_DATASET_DIR / split).rglob("*.png")))
        print(f"  - Total images inside '{split}': {split_count}")

if __name__ == "__main__":
    main()