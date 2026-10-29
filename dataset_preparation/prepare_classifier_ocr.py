import os
import random
import shutil
from pathlib import Path

def main():
    base_dir = Path(__file__).resolve().parent.parent
    source_dir = base_dir / "datasets" / "cropped_dataset_ocr" / "train"
    output_dir = base_dir / "datasets" / "classifier_dataset_ocr_v2"

    classes = ['legible', 'ilegible']
    files_by_class = {}

    for c in classes:
        class_dir = source_dir / c
        if class_dir.exists():
            files_by_class[c] = [
                path for path in class_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
            ]
        else:
            files_by_class[c] = []
            print(f"Warning: Folder not found -> {class_dir}")

    min_count = min(len(files_by_class['legible']), len(files_by_class['ilegible']))
    if min_count == 0:
        raise RuntimeError(
            f"Not enough images found in {source_dir}. "
            "Verify that both 'legible' and 'ilegible' folders exist with images inside."
        )

    print(f"Balancing dataset to {min_count} images per class...")

    random.seed(42)
    balanced_files = {c: random.sample(files_by_class[c], min_count) for c in classes}

    if output_dir.exists():
        shutil.rmtree(output_dir)

    for split in ['train', 'val']:
        for c in classes:
            (output_dir / split / c).mkdir(parents=True, exist_ok=True)

    split_ratio = 0.8
    train_count = int(min_count * split_ratio)

    for c in classes:
        files = balanced_files[c]
        random.shuffle(files)

        train_files = files[:train_count]
        val_files = files[train_count:]

        for img in train_files:
            shutil.copy(str(img), str(output_dir / 'train' / c / img.name))

        for img in val_files:
            shutil.copy(str(img), str(output_dir / 'val' / c / img.name))

    print(f"Dataset successfully balanced and split!")
    print(f"Ready for training at: {output_dir}")

if __name__ == "__main__":
    main()
